#!/usr/bin/env python
#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : Step3_train_Attention.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20181214
#
# Purpose           : Training Visual Attention Model 
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20181214    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

import  os
import  sys
import  argparse
import  json
import  numpy        as np
from    server       import  client_generator
from    src.VA       import  *
from    src.preprocessor  import  *
from    src.config        import  *
from    src.utils         import  *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host',     type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port',     type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  args = parser.parse_args()

  # Create a Visual Attention (VA) model
  VA_model         = VA(alpha_c=config.alpha_c)
  loss, alpha_reg  = VA_model.build_model()
  tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE)

  # Exponential learning rate decaying
  global_step           = tf.Variable(0, trainable=False)
  starter_learning_rate = config.lr
  learning_rate         = tf.train.exponential_decay(starter_learning_rate, global_step,
                                         1000, 0.96, staircase=True)

  # train op
  with tf.name_scope('optimizer'):
    optimizer       = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads           = tf.gradients(loss, tf.trainable_variables())
    grads_and_vars  = list(zip(grads, tf.trainable_variables()))
    train_op        = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

  # Preprocessor
  pre_processor = PreProcessor_VA()

  # Open a tensorflow session
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  sess = tf.InteractiveSession(config=tfconfig)
  tf.global_variables_initializer().run()

  # saver
  saver = tf.train.Saver(max_to_keep=10)
  if config.pretrained_model_path is not None:
    saver.restore(sess, config.pretrained_model_path)
    print("\rLoaded the pretrained model: {}".format(config.pretrained_model_path))

  # Train over the dataset
  data_train  = client_generator(hwm=20, host="localhost", port=args.port)
  data_val    = client_generator(hwm=20, host="localhost", port=args.val_port) 

  for i in range(config.maxiter):
    # Load new dataset
    img, course, speed, curvature, acc, goaldir  = next(data_train) 

    # Preprocessing
    img_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(sess, img, course, speed, curvature, acc, goaldir )

    # Training
    feed_dict = {VA_model.features:      img_p,
                 VA_model.target_course: course_p,
                 VA_model.target_acc:    acc_p,
                 VA_model.speed:         speed_p,
                 VA_model.goaldir:       goaldir_p}
    _, l1loss, alpha_reg_loss = sess.run([train_op, loss, alpha_reg], feed_dict)

    print( '\rStep {}, Loss: {} ({})'.format(i, l1loss, alpha_reg_loss) )

    # validation
    if (i%config.val_steps==0):
      img, course, speed, curvature, acc, goaldir = next(data_val)
      img_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(sess, img, course, speed, curvature, acc, goaldir)
      
      feed_dict = {VA_model.features:       img_p,
                   VA_model.target_course:  course_p,
                   VA_model.target_acc:     acc_p,
                   VA_model.speed:          speed_p,
                   VA_model.goaldir:        goaldir_p}
      l1loss_val, alpha_reg_val = sess.run([loss, alpha_reg], feed_dict)

      print("\rStep {} | train loss: {} | val loss: {} (attn reg: {})".format( i, l1loss, l1loss_val, alpha_reg_val ))
      sys.stdout.flush()

    if i%config.save_steps==0:
      checkpoint_path = os.path.join( config.model_path, "model-%d.ckpt"%i )
      filename        = saver.save(sess, checkpoint_path)
      print("Model saved in file: %s" % filename)
    
    

  # End of code

