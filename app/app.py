import os
import json
import tarfile
import time
import logging
import io
import boto3
import requests
import urllib.request

import torch
import torchvision.transforms as transforms

from PIL import Image

# load the S3 client when lambda execution context is created
s3 = boto3.client('s3')

# classes for the image classification
classes = []

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# get bucket name from ENV variable
MODEL_BUCKET=os.environ.get('MODEL_BUCKET')
logger.info(f'Model Bucket is {MODEL_BUCKET}')

# get bucket prefix from ENV variable
MODEL_KEY=os.environ.get('MODEL_KEY')
logger.info(f'Model Prefix is {MODEL_KEY}')

# processing pipeline to resize, normalize and create tensor object
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(path):
    """Load a TorchScript model from local path

    Returns:
        torch.jit.ScriptModule: The loaded TorchScript model.
    """
    logger.info(f"Model file is : {path}")
    logger.info("Loading PyTorch model")
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    
    return model

def load_classes(path):
    logger.info(f"Classes file is : {path}")
    with open(path) as f:
        classes = f.read().splitlines()
    logger.info(classes)
    return classes

classes = load_classes('/opt/ml/classes.txt')

# load the model when lambda execution context is created
model = load_model('/opt/ml/model.pt')


def predict(input_tensor, model):
    """Predicts the class from an input image.

    Parameters
    ----------
    input_object: Tensor, required
        The tensor object containing the image pixels reshaped and normalized.

    Returns
    ------
    Response object: dict
        Returns the predicted class and confidence score.
    
    """        
    logger.info("Calling prediction on model")
    start_time = time.time()

    # Perform inference using the TorchScript model
    with torch.no_grad():
        output = model(input_tensor)

    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    
    # Convert the output to a probability distribution
    probs = torch.softmax(output, dim=1)

     # Process the output to obtain the predicted class and its probability
    predicted_prob, predicted_class_idx = torch.max(probs, dim=1)

    predict_class = classes[predicted_class_idx]

    logger.info(f'Predicted class is {predict_class}')
    logger.info(f'Confidence score is {predicted_prob.item()}')
    
    response = {}
    response['class'] = str(predict_class)
    response['confidence'] = predicted_prob.item()

    return response
    
def input_fn(request_body):
    """Pre-processes the input data from JSON to PyTorch Tensor.

    Parameters
    ----------
    request_body: dict, required
        The request body submitted by the client. Expect an entry 'path' containing a URL or S3 path of an image to classify.

    Returns
    ------
    PyTorch Tensor object: Tensor
    
    """    
    logger.info("Getting input path to a image Tensor object")

    if isinstance(request_body, str):
        request_body = json.loads(request_body)
    img_request = request_body['queryStringParameters']['path']

    # check if image path is a URL or S3 path
    if img_request.startswith("http"):
        logger.info(f'Loading image from URL - {img_request}')
        # load image from URL
        with urllib.request.urlopen(img_request) as url:
            image = Image.open(io.BytesIO(url.read()))
    elif img_request.startswith("s3://"):
        logger.info(f'Loading image from s3 - {img_request}')
        # load image from S3
        s3 = boto3.client("s3")
        bucket, key = img_request.split("/")[2], "/".join(img_request.split("/")[3:])
        object_data = s3.get_object(Bucket=bucket, Key=key)
        image = Image.open(io.BytesIO(object_data["Body"].read()))
    else:
        logger.info(f'Unknown image path - {img_request}')
        # # load image from local file path
        # image = Image.open(image_path)

    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor
    
def lambda_handler(event, context):
    """Lambda handler function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    print("Starting event")
    logger.info(event)
    print("Getting input object")
    input_object = input_fn(event)
    print("Calling prediction")
    response = predict(input_object, model)
    print("Returning response")
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }