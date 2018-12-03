#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : Step0_download_BDDVdata.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20181201
#
# Purpose           : Download .mov files of BDD-V dataset
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20181201    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

from    sys             import platform
from    tqdm            import tqdm
import  os
import  wget
from    src.utils       import *

# Main function
#----------------------------------------------------------------------
if __name__ == "__main__":
    if platform == 'darwin':
        config = dict2(**{
            "annotations": './data/Sample.csv',     # contains (video url, start, end, action, justification)
            "vid_path":    './data/Videos/videos/'})
    else:
        raise NotImplementedError

    if not os.path.exists(config.vid_path): os.makedirs(config.vid_path)

    with open(config.annotations) as f_obj:
        examples = csv_dict_reader(f_obj)

        '''
        Keys:
            1. Input.Video
            2. Answer.1start
            3. Answer.1end
            4. Answer.1action
            5. Answer.1justification
        '''
        non_exist_vidNames    = []
        for item in tqdm(examples):

            vidName  = item['Input.Video'].split("/")[-1][:-4]

            if len(vidName)==0: continue

            print(bcolors.HEADER + "Video: {}".format(vidName) + bcolors.ENDC)

            #--------------------------------------------------
            # Read video clips
            #--------------------------------------------------
            str2read = '%s%s.mov'%(config.vid_path, vidName) # original resolution: 720x1280
            if not os.path.exists(str2read):
                print(bcolors.BOLD + "Download video clips: {}".format(vidName) + bcolors.ENDC)
                wget.download(item['Input.Video'], out=str2read)     
            else:
                print(bcolors.BOLD + "Already downloaded: {}".format(vidName) + bcolors.ENDC)


