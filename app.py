import argparse
import cv2
import numpy as np
from sys import platform

# Get correct video codec
if platform == "linux" or platform == "linux2":
    CODEC = 0x00000021
elif platform == "darwin":
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Handle an input stream")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args


def capture_stream(args):
    ### TODO: Handle image, video or webcam
    is_image = False
    if args.i == "CAM":
        args.i = 0
    elif args.i.endswith(".jpg") or args.i.endswith(".bmp"):
        is_image = True

    ### TODO: Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # write a video writer
    out = None if is_image else cv2.VideoWriter("out.mp4", CODEC, 30, (100,100))

    # process video until end or process terminated
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Re-size the frame to 100x100
        frame = cv2.resize(frame, (100, 100))

        ### TODO: Add Canny Edge Detection to the frame, 
        ###       with min & max values of 100 and 200
        ###       Make sure to use np.dstack after to make a 3-channel image
        frame = cv2.Canny(frame, 100, 200)
        frame = np.dstack((frame, frame, frame))

        ### TODO: Write out the frame, depending on image or video
        cv2.imwrite('output_image.jpg', frame) if is_image else out.write(frame)
        if key_pressed == 27:
            break

    ### TODO: Close the stream and any windows at the end of the application
    if not is_image:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()
