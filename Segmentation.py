"""
3. Video processing
    3.2. Segmentation (Image processing)

    Note: After execution of the script, one error may prompt, which will not effect our result:
        "Failed to load OpenH264 library: openh264-1.8.0-win64.dll."
        It can be solved by downloading https://github.com/cisco/openh264/releases/tag/v2.0.0 and
        placing the dll file to the current directory.

"""


# importing libraries
import cv2
import os


def segmentation(video_src_path, method, video_tag_path = None):
    """
    :param video_src_path: str
    :param method: str
    :param video_tag_path: str, default path is same to video_src_path
    :return: save/play processed video
    """

    # background remove algorithms
    if method == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(history=60, dist2Threshold=400.0, detectShadows=False)
    elif method == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

    # read video file
    cap = cv2.VideoCapture(video_src_path)

    # assure video file is open
    try:
        assert cap.isOpened() == True
    # print error message
    except AssertionError:
        print("Error opening file, please check the availability of the video source.")

    # Save processed video by reading the original video spec
    h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if video_tag_path == None:
        save_video = cv2.VideoWriter(video_src_path.split('.')[0] + '_processed.mp4', int(fourcc), int(fps), (int(w), int(h)))
    else:
        save_video = cv2.VideoWriter(video_tag_path + '/' + video_src_path.split('.')[0] + '_processed.mp4',
                              int(fourcc), int(fps), (int(w), int(h)))

    # Play video
    while cap.isOpened():
        # read frame by frame
        ret, frame = cap.read()
        if ret == True:
            # apply method to mark background
            fgMask = backSub.apply(frame)
            # apply mask to frame
            proc_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

            # show original video
            cv2.imshow('Original Video', frame)
            # show processed video
            cv2.imshow('Processed Video', proc_frame)
            # save the processed frame
            save_video.write(proc_frame)
        else:
            break

        # Press Q on keyboard to Quit,
        if cv2.waitKey(27) == ord('q'):
            break

    # Release the video capture/writer object
    cap.release()
    save_video.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # run in IDE
    video_src_path = 'video_2.mp4'

    # Create target path if don't exist
    video_tag_path = 'Proc'
    if video_tag_path is not None:
        if not os.path.exists(video_tag_path):
            os.mkdir(video_tag_path)
            print("Directory ", video_tag_path, " Created. ")
        else:
            print("Directory ", video_tag_path, " already exists.")

    # Select/Add new segmentation method
    method = 'KNN'

    # execute main function
    segmentation(video_src_path, method, video_tag_path)
