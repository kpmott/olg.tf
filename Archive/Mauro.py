# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:43:31 2019

@author: mmoretto
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate, Activation
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
tf.config.run_functions_eagerly(True)

from itertools import product
import numpy as np
import pandas as pd
import os 
import time
import datetime

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

class GSE_parallel():
    
    def __init__(self,input_size,output_size,A,M,nodes):
        
        self.input_size = input_size
        self.output_size = output_size
        self.A = A
        self.M = M

        self.n_nodes = nodes
        self.n_layers = len(nodes)
        self.num_epoch = 10
        
        self.create_checkpoint()
        self.learning_rate = 0.001
        self.policy = self.merge_nn_model()
        self.batch_size = 64


    def create_checkpoint(self):
        self.checkpoint_path = 'training/cp.ckpt'
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                           save_weights_only=True,
                                           verbose=False)
        
    def nn_model(self):
        
      init_w = glorot_uniform()
      init_b = tf.constant_initializer(.0)
      
              
      input_state = Input(shape = (self.input_size,))      
      x = Dense(self.n_nodes[0], 
                    activation='tanh', 
                    kernel_initializer=init_w,
                    bias_initializer=init_b)(input_state)      

      for layer in range(1,self.n_layers):
          x = Dense(self.n_nodes[layer], 
                    activation='tanh', 
                    kernel_initializer=init_w,
                    bias_initializer=init_b)(x)
          
      
      output = Dense(self.output_size, activation='linear',
                     kernel_initializer=init_w,
                     bias_initializer=init_b)(x)
       
 #     output = Activation('linear', dtype='float32')(output)

      return input_state, output
  
    def loss(self):#,y_true,y_predict):
        loss = tf.keras.losses.MeanAbsolutePercentageError()#tf.reduce_mean(tf.math.squared_difference(y_true, y_predict))
        return loss

    def merge_nn_model(self):
             
      dict_input = {}
      dict_output = {}

      for age in range(self.A):          
          for m in range(self.M):              
              pol = str(age) + '-' + str(m)
              dict_input[pol], dict_output[pol] = self.nn_model()

      index = list(product(np.arange(self.A),np.arange(self.M)))
      input_merged = [dict_input[str(item[0]) + '-' + str(item[1])] for item in index ]
      output_merged = concatenate([dict_output[str(item[0]) + '-' + str(item[1])] for item in index ])
        
      model = Model(inputs=input_merged, 
                    outputs=output_merged)     

      adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

      metric = tf.keras.metrics.MeanAbsolutePercentageError()
      loss = tf.keras.losses.MeanSquaredError()

      model.compile(loss=loss, 
                    optimizer=adam,
                    metrics=[metric])      
      
      return model
  
    def train_model(self,inputs,outputs):
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        self.policy.fit(inputs, outputs, 
                        epochs=self.num_epoch, 
                        batch_size=self.batch_size, 
                        verbose=True,
                        shuffle=True)
                        #callbacks=[self.cp_callback])

T=1000
input=7
output=1
A=60
M=1
x = GSE_parallel(input,output,A,M,(64,45,32,32))
x.policy.summary()

X = [[np.ones((T,input)) for i in range(A*M)]]
Y = np.zeros((T,A*M))
#y = x.policy.predict(X)

start = time.time()
x.train_model(X,Y)
end = time.time()
print(end-start)

y = x.policy.predict(X)