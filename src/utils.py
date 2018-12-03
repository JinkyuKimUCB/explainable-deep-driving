#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : utils.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20181201
#
# Purpose           : Helper functions
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20181201    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

import  csv
from    sys             import platform

class bcolors:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    GREEN   = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'
    BOLD    = '\033[1m'
    HIGHL   = '\x1b[6;30;42m'
    UNDERLINE = '\033[4m'

class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def csv_dict_reader(file_obj):
    """
    Read a CSV file using csv.DictReader
    """
    return csv.DictReader(file_obj, delimiter=',')

def get_vid_info(file_obj):
    if platform == 'darwin':
        nFrames     = int(file_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width   = int(file_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height  = int(file_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = file_obj.get(cv2.CAP_PROP_FPS)
    else:
        nFrames     = int(file_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width   = int(file_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height  = int(file_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = file_obj.get(cv2.CAP_PROP_FPS)

        # if you have a trouble with these codes,
        # try:
        # pip uninstall opencv-python
        # sudo apt-get install python-opencv
        # OpenCV3 might have an issue with loading a video

    return nFrames, img_width, img_height, fps