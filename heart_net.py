import tensorflow as tf
from BaseNet import BaseNet2d
import numpy as np
from math import ceil

class cu(BaseNet2d):
    def __init__(self, x, y_, batch):
        self.x = x
        self.y_ = y_
        self.batch_size = batch
        self.make()
        self.probs = tf.nn.softmax(self.out)
        self.probs1 = tf.nn.softmax(self.out1)

    def make(self):
        conv0 = self.cr(self.x, [5, 5, 1, 20], [1, 1, 1, 1])
        conv01 = self.cr(conv0, [3, 3, 20, 20], [1, 1, 1, 1])  # 928,640

        pool1 = self.maxpool_2d(conv01, [1, 2, 2, 1], [1, 2, 2, 1])  # 464,320

        conv11 = self.cr(pool1, [3, 3, 20, 20], [1, 1, 1, 1])
        conv12 = self.cr(conv11, [3, 3, 20, 20], [1, 1, 1, 1])
        conv13 = self.cr(conv12, [3, 3, 20, 20], [1, 1, 1, 1])

        pool2 = self.maxpool_2d(conv13, [1, 2, 2, 1], [1, 2, 2, 1])
        conv21 = self.cr(pool2, [3, 3, 20, 20], [1, 1, 1, 1])
        conv22 = self.cr(conv21, [3, 3, 20, 20], [1, 1, 1, 1])
        conv23 = self.cr(conv22, [3, 3, 20, 20], [1, 1, 1, 1])

        pool3 = self.maxpool_2d(conv23, [1, 2, 2, 1], [1, 2, 2, 1])
        conv31 = self.cr(pool3, [3, 3, 20, 20], [1, 1, 1, 1])
        conv32 = self.cr(conv31, [3, 3, 20, 20], [1, 1, 1, 1])
        conv33 = self.cr(conv32, [3, 3, 20, 20], [1, 1, 1, 1])

        pool4 = self.maxpool_2d(conv33, [1, 2, 2, 1], [1, 2, 2, 1])
        conv41 = self.cr(pool4, [3, 3, 20, 20], [1, 1, 1, 1])
        conv42 = self.cr(conv41, [3, 3, 20, 20], [1, 1, 1, 1])
        conv43 = self.cr(conv42, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv1 = self.deconv_2d(conv43, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 32, 32, 20])
        conv71 = self.cr(self.concat([deconv1, conv33], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv72 = self.cr(conv71, [3, 3, 20, 20], [1, 1, 1, 1])
        conv73 = self.cr(conv72, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv2 = self.deconv_2d(conv73, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 64, 64, 20])
        conv81 = self.cr(self.concat([deconv2, conv23], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv82 = self.cr(conv81, [3, 3, 20, 20], [1, 1, 1, 1])
        conv83 = self.cr(conv82, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv3 = self.deconv_2d(conv83, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 128, 128, 20])
        conv91 = self.cr(self.concat([deconv3, conv13], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv92 = self.cr(conv91, [3, 3, 20, 20], [1, 1, 1, 1])
        conv93 = self.cr(conv92, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv4 = self.deconv_2d(conv93, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 256, 256, 20])
        conv101 = self.cr(self.concat([deconv4, conv01], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv102 = self.cr(conv101, [3, 3, 20, 20], [1, 1, 1, 1])
        self.out1 = self.cr(conv102, [3, 3, 20, 8], [1, 1, 1, 1])

        conv01 = self.cr(conv102, [3, 3, 20, 20], [1, 1, 1, 1])
        pool1 = self.maxpool_2d(conv01, [1, 2, 2, 1], [1, 2, 2, 1])
        conv11 = self.cr(self.concat([pool1, conv13], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv12 = self.cr(conv11, [3, 3, 20, 20], [1, 1, 1, 1])
        conv13 = self.cr(conv12, [3, 3, 20, 20], [1, 1, 1, 1])

        pool2 = self.maxpool_2d(conv13, [1, 2, 2, 1], [1, 2, 2, 1])
        conv21 = self.cr(self.concat([pool2, conv23], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv22 = self.cr(conv21, [3, 3, 20, 20], [1, 1, 1, 1])
        conv23 = self.cr(conv22, [3, 3, 20, 20], [1, 1, 1, 1])

        pool3 = self.maxpool_2d(conv23, [1, 2, 2, 1], [1, 2, 2, 1])
        conv31 = self.cr(self.concat([pool3, conv33], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv32 = self.cr(conv31, [3, 3, 20, 20], [1, 1, 1, 1])
        conv33 = self.cr(conv32, [3, 3, 20, 20], [1, 1, 1, 1])

        pool4 = self.maxpool_2d(conv33, [1, 2, 2, 1], [1, 2, 2, 1])
        conv41 = self.cr(self.concat([pool4, conv43], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv42 = self.cr(conv41, [3, 3, 20, 20], [1, 1, 1, 1])
        conv43 = self.cr(conv42, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv1 = self.deconv_2d(conv43, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 32, 32, 20])
        conv71 = self.cr(self.concat([deconv1, conv33], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv72 = self.cr(conv71, [3, 3, 20, 20], [1, 1, 1, 1])
        conv73 = self.cr(conv72, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv2 = self.deconv_2d(conv73, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 64, 64, 20])
        conv81 = self.cr(self.concat([deconv2, conv23], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv82 = self.cr(conv81, [3, 3, 20, 20], [1, 1, 1, 1])
        conv83 = self.cr(conv82, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv3 = self.deconv_2d(conv83, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 128, 128, 20])
        conv91 = self.cr(self.concat([deconv3, conv13], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv92 = self.cr(conv91, [3, 3, 20, 20], [1, 1, 1, 1])
        conv93 = self.cr(conv92, [3, 3, 20, 20], [1, 1, 1, 1])

        deconv4 = self.deconv_2d(conv93, [3, 3, 20, 20], [1, 2, 2, 1], [self.batch_size, 256, 256, 20])
        conv101 = self.cr(self.concat([deconv4, conv01], 3), [3, 3, 40, 20], [1, 1, 1, 1])
        conv102 = self.cr(conv101, [3, 3, 20, 20], [1, 1, 1, 1])

        conv103 = self.cr(conv102, [3, 3, 20, 20], [1, 1, 1, 1])
        conv104 = self.cr(conv103, [3, 3, 20, 20], [1, 1, 1, 1])
        self.out = self.conv_2d(conv104, [3, 3, 20, 8], [1, 1, 1, 1])

