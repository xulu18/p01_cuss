# -*- coding: utf-8 -*-
import tensorflow as tf

class BaseNet2d(object):
    def __init__(self):
        pass
    
    def print_activations(self,t):
        # e.g. print_activations(conv1)
        #
        # Print the name and size after tf operation
        print(t.get_shape().as_list(),': ',t.op.name)

    def conv_2d(self,datain,kernel,stride,Tflag=True,std=1e-1,pad='SAME'):
        weights=tf.Variable(tf.truncated_normal(kernel, dtype=tf.float32,
                                                mean=0.0,stddev=std),trainable=Tflag)
        conv = tf.nn.conv2d(datain, weights, stride, padding=pad)
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[kernel[-1]]),trainable=Tflag)
        out = tf.nn.bias_add(conv, biases)
        self.print_activations(conv)
        return out

    def conv_2d1(self,datain,kernel,stride,Tflag=True,std=1e-1,pad='SAME'):
        weights=tf.Variable(tf.truncated_normal(kernel, dtype=tf.float32,
                                                mean=0.0,stddev=std),trainable=Tflag)
        conv = tf.nn.conv2d(datain, weights, stride, padding=pad)
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[kernel[-1]]),trainable=Tflag)
        out = tf.nn.bias_add(conv, biases)
        self.print_activations(conv)
        return out,weights

    def aconv_2d(self,datain, kernel,rate,Tflag=True,std=1e-1,pad='SAME'):
        weights = tf.Variable(tf.truncated_normal(kernel, dtype=tf.float32,
                                                  mean=0.0, stddev=std), trainable=Tflag)
        aconv = tf.nn.atrous_conv2d(datain, weights, rate, padding=pad)
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[kernel[-1]]), trainable=Tflag)
        out = tf.nn.bias_add(aconv, biases)
        self.print_activations(aconv)
        return out

    def deconv_2d(self,datain,kernel,stride,outshape,Tflag=True,std=1e-1,pad='SAME'):
        weights = tf.Variable(tf.truncated_normal(kernel,dtype=tf.float32,
                                                mean=0.0,stddev=std),trainable=Tflag)  
        #deconv = tf.nn.conv3d_transpose(datain,weights, [FLAGS.batch_size, 130, 100, 1], [1, 10, 10, 1], 'SAME')
        deconv = tf.nn.conv2d_transpose(datain, filter=weights, output_shape=outshape,
                                        strides=stride , padding=pad)  
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[kernel[-2]]),trainable=Tflag)
        out = tf.nn.bias_add(deconv, biases)
        self.print_activations(deconv)
        return out

    def batch(self, x, n_out, is_training):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
                                            
        mean, var = tf.cond(tf.equal(is_training,True),
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        self.print_activations(normed)
        return normed

    def concat(self,indata,axis):
        concated=tf.concat(indata,axis=axis)
        self.print_activations(concated)
        return concated
        
    def maxpool_2d(self,indata,kernel,stride,pad='SAME'):
        out= tf.nn.max_pool(indata,kernel,stride,padding=pad)
        self.print_activations(out)
        return out

    def IN(self, x):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.add(x, -mean), tf.sqrt(tf.add(var, epsilon)))
 
    def batch_norm(self,x,training):
        y = tf.layers.batch_normalization(x, training=training, momentum=0.9)
        return y

 
    def cr(self, datain, kernel, stride):
        conv = self.conv_2d(datain, kernel, stride)
        #aconv1 = self.batch_norm(conv, training)
        aconv1 = self.IN(conv)
        conv_bn_relu = tf.nn.relu(aconv1)
        return conv_bn_relu

    def cr1(self, datain, kernel, stride, istrain=True):

        conv = self.conv_2d(datain, kernel, stride, Tflag=istrain)
        #aconv1 = self.batch_norm(conv, training)
        aconv1 = self.IN(conv)
        conv_bn_relu = tf.nn.relu(aconv1)
        return conv_bn_relu

    def acr(self, datain, kernel, rate):
        aconv = self.aconv_2d(datain, kernel, rate)
        aconv1 = self.IN(aconv)
        aconv_relu = tf.nn.relu(tf.add(aconv1, aconv))
        return aconv_relu
