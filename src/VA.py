#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : VA.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20181214
#
# Purpose           : Tensorflow Visual Attention Model
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20181214    jinkyu      1      initiated
#
# Remark
#|**********************************************************************;

import tensorflow   as      tf
from   src.config   import  *

class VA(object):
  def __init__(self, alpha_c=0.0, dropout=True):
    self.alpha_c     = tf.to_float(alpha_c)
    self.USE_DROPOUT = dropout
    self.SENSORFUSE  = True
    self.timelen     = config.timelen - 3

    # Parameters
    self.T = self.timelen -1
    self.H = config.dim_hidden
    self.L = config.ctx_shape[0]
    self.D = config.ctx_shape[1]
    self.M = config.dim_hidden
    self.V = 1

    # Place holders
    self.features       = tf.placeholder(tf.float32, shape=[None, self.L, self.D])
    self.target_course  = tf.placeholder(tf.float32, shape=[None, 1])
    self.target_acc     = tf.placeholder(tf.float32, shape=[None, 1])
    self.speed          = tf.placeholder(tf.float32, shape=[None, 1])
    self.goaldir        = tf.placeholder(tf.float32, shape=[None, 1])

    # Initializer
    self.weight_initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer  = tf.constant_initializer(0.0)


  def _get_initial_lstm(self, features):
    with tf.variable_scope('initial_lstm'):
      features_mean = tf.reduce_mean(features, 1)

      w_h = tf.get_variable('w_h', [self.D, self.H],  initializer=self.weight_initializer)
      b_h = tf.get_variable('b_h', [self.H],          initializer=self.const_initializer)
      h   = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

      w_c = tf.get_variable('w_c', [self.D, self.H],  initializer=self.weight_initializer)
      b_c = tf.get_variable('b_c', [self.H],          initializer=self.const_initializer)
      c   = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
      return c, h


  def _project_features(self, features):
    with tf.variable_scope('project_features'):
      w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
      features_flat = tf.reshape(features,      [-1, self.D])
      features_proj = tf.matmul( features_flat, w)
      features_proj = tf.reshape(features_proj, [-1, self.L, self.D])

      return features_proj


  def _attention_layer(self, features, features_proj, h, reuse=False):
    with tf.variable_scope('attention_layer', reuse=reuse):
      w     = tf.get_variable('w',     [self.H, self.D], initializer=self.weight_initializer)
      b     = tf.get_variable('b',     [self.D],         initializer=self.const_initializer )
      w_att = tf.get_variable('w_att', [self.D, 1],      initializer=self.weight_initializer)

      h_att      = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    
      out_att    = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])
      alpha      = tf.nn.softmax(out_att)
      alpha_logp = tf.nn.log_softmax(out_att)
      context    = tf.reshape(features * tf.expand_dims(alpha, 2), [-1, self.L*self.D])

      return context, alpha, alpha_logp


  def _decode_lstm(self, h, context, reuse=False, scope='logits'):
    with tf.variable_scope(scope, reuse=reuse):
      w_h   = tf.get_variable('w_h',   [self.H, self.M], initializer=self.weight_initializer)
      b_h   = tf.get_variable('b_h',   [self.M],         initializer=self.const_initializer )
      w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
      b_out = tf.get_variable('b_out', [self.V],         initializer=self.const_initializer )

      if self.USE_DROPOUT: 
        h = tf.nn.dropout(h, 0.5)
      h_logits = tf.matmul(h, w_h) + b_h

      if self.SENSORFUSE:
        w_ctx2out = tf.get_variable(
          name  = 'w_ctx2out', 
          shape = [self.L*self.D+2, self.M], 
          initializer=self.weight_initializer
        )
      else:
        w_ctx2out = tf.get_variable(
          name  = 'w_ctx2out', 
          shape = [self.L*self.D, self.M], 
          initializer=self.weight_initializer
        )
      h_logits += tf.matmul(context, w_ctx2out)

      if self.USE_DROPOUT: 
        h_logits = tf.nn.dropout(h_logits, 0.5)
      out_logits = tf.matmul(h_logits, w_out) + b_out

      return out_logits


  def _batch_norm(self, x, mode='train', name=None):
    return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))


  def build_model(self):
    features    = self.features
    y_acc       = self.target_acc
    y_course    = self.target_course
    batch_size  = tf.shape(features)[0]
    speed       = self.speed 
    goaldir     = self.goaldir

    # apply batch norm to feature vectors
    features      = self._batch_norm(features, mode='train', name='conv_features')

    # Initialize LSTM
    gather_indices_init  = tf.range(config.batch_size) * self.timelen
    features_init        = tf.gather( features, gather_indices_init )
    c, h                 = self._get_initial_lstm(features=features_init)
    lstm_cell            = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

    # Feature projection
    features_proj = self._project_features(features=features)

    # losses
    loss        = 0.0
    alpha_reg   = 0.0
    for t in range(self.T+1):
      gather_indices     = tf.range(config.batch_size) * self.timelen + t
      features_curr      = tf.gather( features,      gather_indices )
      features_proj_curr = tf.gather( features_proj, gather_indices )
      y_acc_curr         = tf.gather( y_acc,         gather_indices )
      y_course_curr      = tf.gather( y_course,      gather_indices )
      speed_curr         = tf.gather( speed,         gather_indices )
      goaldir_curr       = tf.gather( goaldir,       gather_indices )

      context, alpha, alpha_logp = self._attention_layer(features_curr, features_proj_curr, h, reuse=(t!=0))

      # Entropy regularization
      alpha_p    = tf.exp(alpha_logp)
      alpha_ent  = tf.reduce_sum( -alpha_p*alpha_logp ) / tf.to_float(self.L)
      alpha_reg += self.alpha_c * alpha_ent

      if self.SENSORFUSE: context = tf.concat(axis=1, values=[context, speed_curr, goaldir_curr])

      with tf.variable_scope('lstm', reuse=(t!=0)):
        _, (c, h) = lstm_cell(inputs=context, state=[c, h])

      logits_acc    = self._decode_lstm(h, context, reuse=(t!=0), scope='logits_acc'   )
      logits_course = self._decode_lstm(h, context, reuse=(t!=0), scope='logits_course')

      loss  += tf.reduce_sum( tf.abs( tf.subtract( logits_acc,    y_acc_curr    ) ) )
      loss  += tf.reduce_sum( tf.abs( tf.subtract( logits_course, y_course_curr ) ) )

    loss += alpha_reg

    return loss, alpha_reg



  def inference(self):
    features = self.features
    speed    = self.speed 
    goaldir  = self.goaldir

    # apply batch norm to feature vectors
    features = self._batch_norm(features, mode='test', name='conv_features')

    # Initialize LSTM
    gather_indices_init  = tf.range(config.batch_size) * self.timelen
    features_init        = tf.gather( features, gather_indices_init )
    c, h                 = self._get_initial_lstm(features=features_init)
    lstm_cell            = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

    # Feature projection
    features_proj   = self._project_features(features=features)

    y_acc_list, y_course_list, alpha_list = [], [], []
    for t in range(self.T+1):
      gather_indices     = tf.range(config.batch_size) * self.timelen + t 
      features_curr      = tf.gather( features, gather_indices ) 
      features_proj_curr = tf.gather( features_proj, gather_indices ) 
      speed_curr         = tf.gather( speed, gather_indices )
      goaldir_curr       = tf.gather( goaldir, gather_indices )

      context, alpha, _  = self._attention_layer(features_curr, features_proj_curr, h, reuse=(t!=0)) 
      
      if self.SENSORFUSE: context = tf.concat(axis=1, values=[context, speed_curr, goaldir_curr])

      with tf.variable_scope('lstm', reuse=(t!=0)):
        _, (c, h) = lstm_cell(inputs=context, state=[c, h])

      logits_acc    = self._decode_lstm(h, context, reuse=(t!=0), scope='logits_acc')
      logits_course = self._decode_lstm(h, context, reuse=(t!=0), scope='logits_course')

      # accumulation
      y_acc_list.append(logits_acc)
      y_course_list.append(logits_course)
      alpha_list.append(alpha)

    alphas    = tf.transpose(tf.stack(values=alpha_list), (1,0,2))
    ys_acc    = tf.squeeze(y_acc_list)
    ys_course = tf.squeeze(y_course_list)

    return alphas, ys_acc, ys_course



