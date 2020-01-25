#-*-coding:utf-8-*-
from class_path import MyClassPath
import os
import numpy as np
import h5py
from eprogress import LineProgress


class MyDataSet(MyClassPath):
    def __init__(self, str_list_path, str_suffix, list_key_data, list_key_label, flag_read_all=True):
        MyClassPath.__init__(self, str_list_path, str_suffix)
        self._key_data = list_key_data
        self._key_label = list_key_label

        self._index = np.array([])
        self._data_set = {}
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._flag_read_all = flag_read_all
        self._path_now = ''

        self._get_index()
        if flag_read_all is True:
            self._get_dataset()
        else:
            pass
        #print(np.shape(np.where(self._data_set['label'][9] == 0)))
        #print(np.shape(np.where(self._data_set['label'][9] == 1)))


    def _get_dataset(self):
        line_progress = LineProgress(title='Loading files')
        for index, i in enumerate(self._path):
            _temp_dict = self._get_data(i)
            if index == 0:
                for k in self._key_data:
                    temp_shape = list(np.shape(_temp_dict[k]))
                    temp_shape[0] = self._num_files
                    #print(temp_shape)
                    self._data_set[k] = np.zeros(shape=tuple(temp_shape))
                    self._data_set[k][0] = _temp_dict[k]
                for k in self._key_label:
                    temp_shape = list(np.shape(_temp_dict[k]))
                    temp_shape[0] = self._num_files
                    #print(temp_shape)
                    self._data_set[k] = np.zeros(shape=tuple(temp_shape))
                    self._data_set[k][0] = _temp_dict[k]
            else:
                for k in self._key_data:
                    #print(np.shape(self.data_set[k]))
                    #print(np.shape(_temp_dict[k]))
                    self._data_set[k][index] = _temp_dict[k]
                for k in self._key_label:
                    self._data_set[k][index] = _temp_dict[k]

            line_progress.update( (index+1) / self._num_files * 100)
        print()

    def _get_index(self):
        if len(self._index) == 0:
            self._index = [i for i in range(self._num_files)]
        else:
            np.random.shuffle(self._index)

    def _get_data(self, str_file_path):
        if os.path.splitext(str_file_path)[-1] in ['.h5','.mat']:
            ret_dict = {}
            with h5py.File(str_file_path, 'r') as f:
                for i in self._key_data:
                    ret_dict[i]= np.expand_dims(np.array(f[i]),axis=0)
                for i in self._key_label:
                    ret_dict[i] = np.expand_dims(np.array(f[i]),axis=0)
        elif os.path.splitext(str_file_path)[-1] in ['.gz']:
            pass
        else:
            raise Exception( os.path.splitext(str_file_path)[-1] + ' are not supported yet' )
        return ret_dict

    def next_batch(self, batch_size = 1):
        if self._flag_read_all is True:
            assert batch_size <= self._num_files
            ret_dict = {}
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_files:
                #print()
                self._epochs_completed += 1
                self._get_index()
                start = 0
                self._index_in_epoch = batch_size
            end = self._index_in_epoch
            target_index = self._index[start:end]
            if len(target_index)==1:
                for k in self._key_data:
                    ret_dict[k] = np.expand_dims(self._data_set[k][target_index[0]], axis=0)
                for k in self._key_label:
                    ret_dict[k] = np.expand_dims(self._data_set[k][target_index[0]],axis=0)
            else:
                pass
            return ret_dict
        elif self._flag_read_all is False:
            # if batch size is 1
            assert batch_size <= self._num_files
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            ret_dict = self._get_data(self._path[self._index[start]])
            self._path_now = self._path[self._index[start]]
            if self._index_in_epoch >= self._num_files:
                # print()
                self._epochs_completed += 1
                self._get_index()
                self._index_in_epoch = 0
            return ret_dict




'''
a = MyDataSet([r'G:\DataSet\anzhen_hospital\TC_mat\cv6',\
               r'G:\DataSet\anzhen_hospital\TC_mat\cv7',\
               r'G:\DataSet\anzhen_hospital\TC_mat\cv8',\
               r'G:\DataSet\anzhen_hospital\TC_mat\cv9'], r'.mat', ['data'], ['label'],flag_read_all=False)
line_progress = LineProgress(title='Loading files')
max_index = 0
min_index = 100
for i in range(a._num_files):
    temp = a.next_batch()
    line_progress.update((i + 1) / a._num_files * 100)
    if np.size(temp['label']) is not 1:
        print(a._path_now)
    index = temp['label'][0]
    if index >= max_index:
        max_index = index
    if index <= min_index:
        min_index = index
print('max: %g, min: %g'%(max_index,min_index))

'''

