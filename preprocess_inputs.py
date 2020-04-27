import cv2
import numpy as np

def preprocess_image(image, height, width):
    preprocessed_image = cv2.resize(image, (width, height))
    preprocessed_image = preprocessed_image.transpose(2,0,1)
    preprocessed_image = preprocessed_image.reshape(1,3,height,width)

    return preprocessed_image


def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.

    https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html

    Name: input , shape: [1x3x256x456]. An input image in the [BxCxHxW] format , where:
    B - batch size
    C - number of channels
    H - image height
    W - image width. Expected color order is BGR.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the pose estimation model

    return preprocessed_image


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.

    https://docs.openvinotoolkit.org/latest/_models_intel_text_detection_0004_description_text_detection_0004.html

    name: "input" , shape: [1x3x768x1280] - An input image in the format [BxCxHxW], where:
    B - batch size
    C - number of channels
    H - image height
    W - image width
    Expected color order - BGR.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the text detection model

    return preprocessed_image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.

    https://docs.openvinotoolkit.org/latest/_models_intel_vehicle_attributes_recognition_barrier_0039_description_vehicle_attributes_recognition_barrier_0039.html

    name: "input" , shape: [1x3x72x72] - An input image in following format [1xCxHxW], where:
    - C - number of channels
    - H - image height
    - W - image width.
    Expected color order - BGR.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model

    return preprocessed_image
