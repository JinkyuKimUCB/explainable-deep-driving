#!/usr/bin/env python
#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : Step4_preprocessing_explanation.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20190108
#
# Purpose           : Preprocessing for explanation generator 
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20190108    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

import  os
import  h5py
from    src.config      import  *
from    src.utils_nlp   import *
from    src.utils       import *

def main():
    #-----------------------
    # Parameters
    #-----------------------
    if platform == 'linux':
        param = dict2(**{
            "max_length":       20,    # the maximum length of sentences
            "vid_max_length":   10,    # the maximum length of video sequences
            "size_of_dict":     10000, # the size of dictionary
            "chunksize":        10,    # for h5 format file writing
            "savepath":         'cap',
            "FINETUNING":       False,
            "SAVEPICKLE":       True })
    else:
        raise NotImplementedError
    
    check_and_make_folder(config.h5path+"cap/log/")
    check_and_make_folder(config.h5path+"cap/feat/")

    #-----------------------
    # For each split, collect feats/logs
    #-----------------------
    for split in ['train', 'test', 'val']:
        check_and_make_folder(config.h5path+param.savepath+'/'+split)

        # Step1: Preprocess caption data + refine captions
        caption_file = config.h5path + 'info/captions_BDDX_' + split + '.json' 
        annotations  = process_caption_data(caption_file=caption_file, image_dir=config.h5path+'feat/', max_length=param.max_length)
        if param.SAVEPICKLE: save_pickle(annotations, config.h5path + '{}/{}/{}.annotations.pkl'.format(param.savepath, split, split))
        print(bcolors.BLUE   + '[main] Length of {} : {}'.format(split, len(annotations)) + bcolors.ENDC)

        # Step2: Build dictionary
        if param.FINETUNING:
            with open(os.path.join(config.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, 'train')), 'rb') as f:
                word_to_idx = pickle.load(f)
        else:
            if split == 'train':
                word_to_idx, idx_to_word = build_vocab(annotations=annotations, size_of_dict=param.size_of_dict)
                if param.SAVEPICKLE: save_pickle(word_to_idx, config.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, split))
                if param.SAVEPICKLE: save_pickle(idx_to_word, config.h5path + '{}/{}/idx_to_word.pkl'.format(param.savepath, split))
            else:
                with open(os.path.join(config.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, 'train')), 'rb') as f:
                    word_to_idx = pickle.load(f)

        # Step3: Clustering
        if split == 'train': clusters, ind_cluster = cluster_annotations(annotations=annotations, k=20)

        # Step4: word to index
        captions = build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=param.max_length)
        if param.SAVEPICKLE: save_pickle(captions, config.h5path + '{}/{}/{}.captions.pkl'.format(param.savepath, split, split))

        # Step5: feat & masks
        all_feats4Cap, all_masks4Cap, all_logs, all_attns4Cap = build_feat_matrix( 
                                                           annotations=annotations, 
                                                           max_length=param.vid_max_length, 
                                                           fpath=config.h5path, 
                                                           FINETUNING=param.FINETUNING)

        # Step6: Saving these data into hdf5 format
        feat = h5py.File(config.h5path + "cap/feat/" + split + ".h5", "w")
        logs = h5py.File(config.h5path + "cap/log/"  + split + ".h5", "w")

        dset = feat.create_dataset("/X",     data=all_feats4Cap, chunks=(param.chunksize, param.vid_max_length, 64, 12, 20) ) #fc8
        dset = feat.create_dataset("/mask",  data=all_masks4Cap)
        
        dset = logs.create_dataset("/attn",  data=all_attns4Cap, chunks=(param.chunksize, param.vid_max_length, 240))
        dset = logs.create_dataset("/Caption",      data=captions)
        dset = logs.create_dataset("/timestamp",    data=all_logs['timestamp'])
        dset = logs.create_dataset("/curvature",    data=all_logs['curvature'])
        dset = logs.create_dataset("/accelerator",  data=all_logs['accelerator'])
        dset = logs.create_dataset("/speed",        data=all_logs['speed'])
        dset = logs.create_dataset("/course",       data=all_logs['course'])
        dset = logs.create_dataset("/goaldir",      data=all_logs['goaldir'])
        dset = logs.create_dataset("/pred_accel",   data=all_logs['pred_accel'])
        dset = logs.create_dataset("/pred_courses" ,data=all_logs['pred_courses'])

        if split == 'train': dset = logs.create_dataset("/cluster",      data=ind_cluster)

        print(bcolors.GREEN + '[main] Finish writing into hdf5 format: {}'.format(split) + bcolors.ENDC)

if __name__ == "__main__":
    main()










