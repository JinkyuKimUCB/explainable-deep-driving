#!/usr/bin/env python
#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : Step2_0_train_CNNonly.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20181201
#
# Purpose           : Training CNN+FF (feed forward) model 
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20181201    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

import  os
import  sys
import  argparse
from    server            import  client_generator
from    src.preprocessor  import  *
from    src.NVIDIA_CNN    import  *
from    src.config        import  *
from    src.utils         import  *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host',     type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port',     type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--log_path', type=str, default='./saved/log/')
  args = parser.parse_args()

  # Open a tensorflow session
  gpu_options = tf.GPUOptions() #per_process_gpu_memory_fraction=config.gpu_fraction
  sess        = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

  # Create a CNN+FF model
  USE_SINGLE_FRAME = False      # if False, model use 4 consecutive frames as an input
  NVIDIA_model = NVIDIA_CNN(sess, USE_SINGLE_FRAME=USE_SINGLE_FRAME)

  # Preprocessor
  if USE_SINGLE_FRAME: 
    pre_processor = PreProcessor_CNN()
  else: 
    pre_processor = PreProcessor_CNN_4frame()

  # tensorflow saver
  saver = tf.train.Saver(max_to_keep=20)
  if config.pretrained_model_path is not None:
    saver.restore(sess, config.pretrained_model_path)
    print("\rLoaded the pretrained model: {}".format(config.pretrained_model_path))

  # Train over the dataset
  data_train  = client_generator(hwm=20, host="localhost", port=args.port)
  data_val    = client_generator(hwm=20, host="localhost", port=args.val_port) 

  # create folder
  check_and_make_folder(config.model_path)

  for i in range(config.maxiter):
    # Load new dataset
    X_batch, course_batch, speed_batch, \
    curvature_batch, accelerator_batch, goaldir_batch = next(data_train) 

    # Preprocessing
    Xprep_batch, curvatures, accelerators, speeds, \
    courses, _, goaldirs, _ = pre_processor.process(
      sess, X_batch, course_batch, speed_batch, 
      curvature_batch, accelerator_batch, goaldir_batch )

    l1loss, loss_acc, loss_cur, _ = NVIDIA_model.process(
      sess=sess, 
      x=Xprep_batch, 
      c=courses, 
      a=accelerators,
      s=speeds, 
      g=goaldirs )

    if (i%config.val_steps==0):
      X_val, course_val, speed_val, curvature_val, accelerator_val, goaldir_val = next(data_val)

      # preprocessing
      Xprep_val, curvatures_val, accelerators_val, speeds_val, \
      courses_val, _, goaldirs_val, _ = pre_processor.process(
        sess, X_val, course_val, speed_val, curvature_val, accelerator_val, goaldir_val)

      l1loss_val, l1loss_val_acc, l1loss_val_cur, a_pred = NVIDIA_model.validate(
        sess=sess, 
        x=Xprep_val, 
        c=courses_val, 
        a=accelerators_val, 
        s=speeds_val, 
        g=goaldirs_val )

      print("\rStep {} | train loss: {} | val loss: {} (acc: {}, cur: {})".format( i, l1loss, l1loss_val, l1loss_val_acc, l1loss_val_cur))
      sys.stdout.flush()

    if i%config.save_steps==0:
      checkpoint_path = os.path.join(config.model_path, "model-{}.ckpt".format(i))
      filename        = saver.save(sess, checkpoint_path)
      print("Current model is saved: {}".format(filename))
    
  # End of code

