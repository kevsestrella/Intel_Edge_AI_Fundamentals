import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.

    The net outputs two blobs with the [1, 38, 32, 57] and [1, 19, 32, 57]
    shapes. The first blob contains keypoint pairwise relations (part affinity
    fields), while the second blob contains keypoint heatmaps.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    heatmaps = output['Mconv7_stage2_L2']
    # TODO 2: Resize the heatmap back to the size of the input
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    for h in range(len(heatmaps[0])):
        # cv2 at it again with it's reverse dimensions, so this just means 
        # upscale the 32 x 57 to 562 x 1000
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.

    name: "model/link\_logits\_/add", shape: [1x16x192x320] - logits related to
            linkage between pixels and their neighbors.
    name: "model/segm\_logits/add", shape: [1x2x192x320] - logits related to
            text/no-text classification for each pixel.
    Refer to PixelLink and demos for details.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    text_notext = output['model/link_logits_/add']
    # TODO 2: Resize this output back to the size of the input
    out_text = np.zeros([text_notext.shape[1], input_shape[0], input_shape[1]])
    h = 0
    for classification in output['model/segm_logits/add'][0]:
        out_text[h] = cv2.resize(classification, input_shape[0:2][::-1])
        h += 1
    return out_text


def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.

    name: "color", shape: [1, 7, 1, 1] - Softmax output across seven color
    classes [white, gray, yellow, red, green, blue, black]
    name: "type", shape: [1, 4, 1, 1] - Softmax output across four type
    classes [car, bus, truck, van]
    '''
    # TODO 1: Get the argmax of the "color" output
    # TODO 2: Get the argmax of the "type" output

    return None


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
