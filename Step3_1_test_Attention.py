#!/usr/bin/env python
#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : Step3_1_test_Attention.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20181214
#
# Purpose           : Testing Visual Attention Model 
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20181214    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

import  argparse
import  sys
import  os
import  numpy as np
import  h5py
import  tensorflow        as      tf
from    collections       import namedtuple
from    src.utils         import  *
from    src.preprocessor  import  *
from    src.config        import  *
from    src.VA    	      import  *
from    sys               import platform
from    tqdm              import tqdm


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Path viewer')
  parser.add_argument('--getscore',      type=bool,  default=False, help='get performance scores')
  parser.add_argument('--showvideo',     type=bool,  default=False, help='show video')
  parser.add_argument('--useCPU',        type=bool,  default=False, help='without GPU processing')
  parser.add_argument('--validation',    type=bool,  default=False, help='use validation set')
  parser.add_argument('--gpu_fraction',  type=float, default=0.7,   help='GPU usage limit')
  parser.add_argument('--extractAttn',   type=bool,  default=True,  help='extract attention maps')
  args = parser.parse_args()

  if platform == 'darwin':
    args.model      = "./model/VA/model-0.ckpt"
    args.savepath   = "./result/VA/"
    config.timelen  = 400+3
    timelen         = 400
    config.batch_size = 1
  else:
    raise NotImplementedError

  if args.getscore:    check_and_make_folder(args.savepath)
  if args.extractAttn: check_and_make_folder(config.h5path + "attn/")

  # prepare datasets
  if args.validation: filenames = os.path.join(config.h5path, 'val.txt'  )
  else:               filenames = os.path.join(config.h5path, 'train.txt')

  with open(filenames, 'r') as f:
    fname = ['%s'%x.strip() for x in f.readlines()]

  # Create VA model
  VA_model = VA(alpha_c=config.alpha_c)
  alphas, y_acc, y_course = VA_model.inference()

  if args.useCPU: # Use CPU only
    tfconfig = tf.ConfigProto( device_count={'GPU':0}, intra_op_parallelism_threads=1)
    sess = tf.Session(config=tfconfig)
  else: # Use GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

  # Preprocessor
  pre_processor = PreProcessor_VA(timelen=timelen, phase='test')

  # Load the pretrained model
  saver = tf.train.Saver()
  if args.model is not None:
    saver.restore(sess, args.model)
    print("\rLoaded the pretrained model: {}".format(args.model))

  for dataset in tqdm(fname):
    print(bcolors.HIGHL+"Dataset: {}".format(dataset)+bcolors.ENDC)

    log    = h5py.File(config.h5path + "log/" + dataset + ".h5", "r")
    feats  = h5py.File(config.h5path + "feat/"+ dataset + ".h5", "r")
    cam    = h5py.File(config.h5path + "cam/" + dataset + ".h5", "r")
    nImg   = cam['X'].shape[0]
    nFeat  = feats['X'].shape[0]

    # initialization
    feat_batch      = np.zeros((timelen, 64, 12, 20))
    curvature_batch = np.zeros((timelen, 1))
    accel_batch     = np.zeros((timelen, 1))
    speed_batch     = np.zeros((timelen, 1))
    course_batch    = np.zeros((timelen, 1))
    goaldir_batch   = np.zeros((timelen, 1))
    timestamp_batch = np.zeros((timelen, 1))

    # preprocess logs
    feat_batch[:nFeat]          = feats['X'][:]
    timestamp_batch[:nFeat]     = preprocess_others(log["timestamp"][:],   nImg)[3:]
    curvature_batch[:nFeat]     = preprocess_others(log["curvature"][:],   nImg)[3:] 
    accel_batch[:nFeat]         = preprocess_others(log["accelerator"][:], nImg)[3:] 
    speed_batch[:nFeat]         = preprocess_others(log["speed"][:],       nImg)[3:] 
    course_batch[:nFeat]        = preprocess_course(log["course"][:],      nImg)[3:]      
    goaldir_batch[:nFeat]       = preprocess_others(log["goaldir"][:],     nImg)[3:]

    # Preprocessing for tensorflow
    feat_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(
      		sess=sess, 
      		inImg=np.expand_dims(np.array(feat_batch),0), 
      		course=np.expand_dims(np.array(course_batch),0), 
      		speed=np.expand_dims(np.array(speed_batch),0), 
      		curvature=np.expand_dims(np.array(curvature_batch),0), 
      		accelerator=np.expand_dims(np.array(accel_batch),0), 
      		goaldir=np.expand_dims(np.array(goaldir_batch),0) )

    # Run a model
    feed_dict = {VA_model.features:           feat_p,
                 VA_model.speed:              speed_p,
                 VA_model.goaldir:            goaldir_p}
    alps, pred_accel, pred_courses = sess.run([alphas, y_acc, y_course], feed_dict)
    alps = np.squeeze(alps)

    if args.extractAttn:
      print(config.h5path + "attn/" + dataset + ".h5")
      f     = h5py.File(config.h5path + "attn/" + dataset + ".h5", "w")
      dset  = f.create_dataset("/attn",     data=alps,            chunks=(20,240))
      dset  = f.create_dataset("/timestamp",data=timestamp_batch, chunks=(20,1))
      dset  = f.create_dataset("/curvature",data=curvature_batch, chunks=(20,1))
      dset  = f.create_dataset("/accel",    data=accel_batch,     chunks=(20,1))
      dset  = f.create_dataset("/speed",    data=speed_batch,     chunks=(20,1))
      dset  = f.create_dataset("/course",   data=course_batch,    chunks=(20,1))
      dset  = f.create_dataset("/goaldir",  data=goaldir_batch,   chunks=(20,1))
      dset  = f.create_dataset("/pred_accel",    data=np.expand_dims(pred_accel,1),   chunks=(20,1))
      dset  = f.create_dataset("/pred_courses",  data=np.expand_dims(pred_courses,1), chunks=(20,1))

  # Total Result
  print(bcolors.HIGHL + 'Done' + bcolors.ENDC)



























