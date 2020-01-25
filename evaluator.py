#-*-coding:utf-8-*-

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy.spatial
class Evaluator(object):
    def __init__(self,gt,pred):
        self.gt=gt
        self.pred=pred
        assert self.gt.shape == self.pred.shape,\
               'gt and pred possess different shape'+gt.shape+'vs'+pred.shape

    def Dice(self,index):
        # Calculate dice coefficient on index
        gt_d=self._normalize_copy(self.gt,index)
        pred_d=self._normalize_copy(self.pred,index)
        dice= np.sum(pred_d[gt_d==1])*2.0 / (np.sum(pred_d)+np.sum(gt_d))       
        return dice

    def Dice1(self,index):
          # Calculate dice coefficient on index
          gt_d=self._normalize_copy2(self.gt,index)
          pred_d=self._normalize_copy2(self.pred,index)
          dice= np.sum(pred_d[gt_d==1])*2.0 / (np.sum(pred_d)+np.sum(gt_d))
          return dice


    def IoU(self,num):
          IoU_sum = 0
          if num is 6:
             for index in[2,3,4,5,6,7]:
             # Calculate dice coefficient on index
                gt_d=self._normalize_copy(self.gt,index)
                pred_d=self._normalize_copy(self.pred,index)
                IoU = np.sum(pred_d[gt_d==1]) / (np.sum(pred_d)+np.sum(gt_d)-np.sum(pred_d[gt_d==1]))
                IoU_sum = IoU_sum+IoU
             mIoU = IoU_sum/num
          else: 
             for index in[0,1,2,3,4,5,6,7]:
             # Calculate dice coefficient on index
                gt_d=self._normalize_copy(self.gt,index)
                pred_d=self._normalize_copy(self.pred,index)
                IoU = np.sum(pred_d[gt_d==1]) / (np.sum(pred_d)+np.sum(gt_d)-np.sum(pred_d[gt_d==1]))
                IoU_sum = IoU_sum+IoU
             mIoU = IoU_sum/num
          return mIoU

    def hd(self,index):
        A=self._normalize_copy2(self.gt,index)
        B=self._normalize_copy2(self.pred,index)
        [hd1, a1, a2] = scipy.spatial.distance.directed_hausdorff(A[0],B[0])
        [hd2, a1, a2] = scipy.spatial.distance.directed_hausdorff(B[0],A[0])
        hd = max(hd1,hd2)
        return hd


    def dice_all(self):
          a = 0
          for i in [1,2,3,4,5,6,7]:
              a = a +self.Dice1(i)
          a = a/7
          return a
    def hf_all(self):
          a = 0
          for i in [1,2,3,4,5,6,7]:
              a = a +self.hd(i)
          a = a/7
          return a
    
    def _normalize_copy(self,data,index):
        temp=data.copy()
        temp[data==index]=1
        temp[data!=index]=0
        return temp

    def _normalize_copy2(self,data,index):
        temp=data.copy()
        if index==6:
           for i in [2,3,4,5,6]:
              temp[data==i]=1
           temp[data==7]=0
           temp[data==0]=0
           temp[data == 1] = 0
        elif index==7:
           for i in [2,3,4,5,6,7]:
              temp[data==i]=1   
           temp[data==0]=0
        else:
           temp[data==index]=1
           temp[data!=index]=0
        return temp
