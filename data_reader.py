#-*-coding:utf-8-*-
'''
The DataReader Class is designed for extract data form disk to memory
for traing a model or tesing it.

To use this class, the foramtion of data should be .mat with -V7.3 and
each .mat file should contain complete info of a single piece of data. 
Moreover, tesing and validating data should also contain a label no
matter it is useful or not

Finished in 2017 Sep.
Author: Liu Mingyuan

'''
import os
import h5py
import numpy as np
from random import randint
from data_logger import DataLogger


class DataReader(object):
    def __init__(self,logger=None):
        #list.extend()
        self.file_locations=[]  #Pathes storage absolute location of file
        self.num_files = 0 # Number of files
        self.data_keys =[] # List containing all the keys of hdf5 file
        
        self.data_shape={} # Dictionary containing shape of data ranked by keys      
        self.data={} # Dictionary containing data ranked by keys
        
        self.epochs_completed = 0 # Count how many epoachs has went through
        self.index_in_epoch = 0 # Count data that have been forward
        self.logger=logger
        
    def print(self,str_print,end='\n'):
        
        if self.logger is None:
            print(str_print)
        else:
            self.logger.log_print(str_print,end)
        
    def next_batch(self, batch_size):
        '''
        Forward a batch of data for training of testing
        \batch_size: int, the number of data to forward once.
        '''
        assert batch_size <= self.num_files
        ret_dict={}
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_files:

            self.epochs_completed += 1

            self.randomize_dict(self.data,self.num_files)
            start = 0
            self.index_in_epoch = batch_size
            
            
        end = self.index_in_epoch
        
        for k in self.data_keys:
            ret_dict[k]=self.data[k][start:end]
        return ret_dict

        
    def randomize_dict(self,dict_data,num_data):
        '''
        hyper function for data randomizing, warp of the function randomize_with_model
        \dict_data: dictionary containing data waiting to be shuffled
        \num_data: int data, indicating the number of files
        
        '''
        list_keys=list(dict_data.keys())
        num_keys=len(list_keys)
        permutation = np.random.permutation(num_data)
        for k in list_keys:
            dict_data[k]=self.randomize_with_model(dict_data[k],permutation,True)
       
    def randomize_with_model(self,list_data,list_model=[],flag_modelbased=False):
        '''
        basic function for data randomizing
        \list_data:list of data to be randomize
        \list_model:1 D list as a model for randomize
        \flag_modelbased: False, generate a list of random number for shuffling
        '''
        if flag_modelbased is False:
            permutation = np.random.permutation(list_data.shape[0])
            shuffled_dataset = list_data[permutation]
            return shuffled_dataset,permutation
        else:
            assert len(list_model) is not 0,\
                   'Shuffling model should not be empty'
            assert len(list_model)==list_data.shape[0],\
                   'Shuffling data and length of model dismatch'+str(len(list_model))+'VS'+str(list_data.shape[0])
            shuffled_dataset = list_data[list_model]
            return shuffled_dataset

    def initialize_dataset(self,check=True):
        '''
        Extract data from self.file_locations and self.data_locations
        to self.data and self.label
         
        '''
        self.print('initializing...')
        self.check_locations(self.file_locations)     
        self.num_files=len(self.file_locations)
        self.data_keys,self.data_shape=self.get_hdf5_info(self.file_locations[0])
        self.print('reading   ...   ...')
        for i in range(len(self.data_keys)):
            self.data[self.data_keys[i]]=np.array([h5py.File(str_f)[self.data_keys[i]] for str_f in self.file_locations])
            self.print(self.data_keys[i]+' Done!')
        if check is True:
            self.check_data_distribution(self.data)
        
    def check_data_distribution(self,dict_data):
        '''
        Show batch of basic info of data
        \dict_data: a dictionary containing data
        
        '''
        self.print('- - - - - - - - -')
        for k in list(dict_data.keys()):
            temp_data=dict_data[k]
            self.print(k+':  num:'+str(np.shape(temp_data)[0])+'  shape:',end='')
            self.print(np.shape(temp_data)[1:],end='')
            num_select=randint(0,np.shape(temp_data)[0]-1)

            assert len(np.shape(temp_data)[1:])>0 and len(np.shape(temp_data)[1:])<=3,\
                   'Un defined data shape'
            if len(np.shape(temp_data)[1:]) is 3:            
                self.print('  in file '+str(num_select)+':')
                self.print('    max :'+str(np.max(temp_data[num_select,:,:,:])))
                self.print('    min :'+str(np.min(temp_data[num_select,:,:,:])))
                self.print('    mean:'+str(np.mean(temp_data[num_select,:,:,:])))
                self.print('    var :'+str(np.var(temp_data[num_select,:,:,:])))
                self.print('    cat :'+str(len(np.unique(temp_data[num_select,:,:,:]))))
            elif len(np.shape(temp_data)[1:]) is 2: 
                self.print('  in file '+str(num_select)+':')
                self.print('    max :'+str(np.max(temp_data[num_select,:,:])))
                self.print('    min :'+str(np.min(temp_data[num_select,:,:])))
                self.print('    mean:'+str(np.mean(temp_data[num_select,:,:])))
                self.print('    var :'+str(np.var(temp_data[num_select,:,:])))
                self.print('    cat :'+str(len(np.unique(temp_data[num_select,:,:]))))
            elif len(np.shape(temp_data)[1:]) is 1: 
                self.print('  in file '+str(num_select)+':')
                self.print('    max :'+str(np.max(temp_data[num_select,:])))
                self.print('    min :'+str(np.min(temp_data[num_select,:])))
                self.print('    mean:'+str(np.mean(temp_data[num_select,:])))
                self.print('    var :'+str(np.var(temp_data[num_select,:])))
                self.print('    cat :'+str(len(np.unique(temp_data[num_select,:]))))

         
    def check_locations(self,list_locations):
        '''
        Check whther self.file_locations are legal
        and print the nuber of files to be read
        \list_locations: 1D list containing absolute path of data
        '''
        
        assert len(list_locations)is not 0,\
            'There are no file in the list for initializing'
        num_show=min(len(list_locations),10)
        for f in range(num_show):
            self.print(list_locations[f])
        self.print('...    ...')
        self.print('- - - - - - - - -')
        self.print('%g files waiting to read'%len(list_locations))
          


    def get_hdf5_info(self,str_filename,flag_print=True):
        '''
        View keys of hef5 files
        \str_filename: string, the absolute path of hdf5 file
        '''
        assert h5py.is_hdf5(str_filename),\
               'File %g is not a hdf5 file'
        f = h5py.File(str_filename, 'r')
        f_keys=list(f.keys()) 
        data_keys=f_keys.copy()
        data_shape={}
        if flag_print is True:
          self.print('%g keys in the first file:'%len(data_keys))
        for k in data_keys:
            if flag_print is True:
                self.print(k+':  ',end='')
            data_shape[k]=f[k].shape
            if flag_print is True:
                self.print(data_shape[k])
            
        f.close()
        return data_keys,data_shape


    
            
    def get_files(self,str_path,str_suffix):
        '''
        return file list under path 'str_path'
        
        \str_path:string, absolute target path
        \str_suffix:string, target type of file
        '''  
        assert os.path.exists(str_path),\
               'PathError:no such path: '+str_path
        all_files = os.listdir(str_path)
        files=[os.path.join(str_path,f) for f in all_files if os.path.splitext(f)[1]=='.'+str_suffix]
        return files
    
    def get_folders(self,str_path):
        ''' 
        return folder list under path 'str_path'        
        \str_path:string, absolute target path
        '''  
        assert os.path.exists(str_path),\
               'PathError:no such path: '+str_path
        all_files = os.listdir(str_path)
        folders=[os.path.join(str_path,f)\
                 for f in all_files if os.path.isdir(os.path.join(str_path,f))]
        return folders

        
        
    def list_files(self,list_datapath,str_datasuffix,\
                   flag_report=True):
        '''
        list file names of data and coresponding labels
        \list_datapath:list, contining pathes of dataset
        \str_datasuffix:string, define the target type of data
        \flag_report:bool, if True:list files and labels will be
                     reported to self.file_locations and self.label_locations


        data of the same kind should be settled in a single folder.
        for segmentating question, all training data should be in a folder
        '''

        num_folders=len(list_datapath)
        
        for i in range(num_folders):
            ext_files=self.get_files(list_datapath[i],str_datasuffix)
            self.file_locations.extend(ext_files)

    def result_of_single_file(self,str_file):
        assert h5py.is_hdf5(str_file),\
               'File %g is not a hdf5 file'
        self.data_keys,self.data_shape=self.get_hdf5_info(str_file,flag_print=False)
        self.num_files=1
        for i in range(len(self.data_keys)):
            self.data[self.data_keys[i]]=np.array([h5py.File(str_file)[self.data_keys[i]]])

        
        
        
#b=DataLogger(r'G:\tools\tensor\log')
'''
def print(str_print,end='\n'):
    b.log_print(str_print,end)


a=DataReader()
a.list_files([r'G:\MICCAI_2017\data\train'],'mat',flag_report=True)
a.initialize_dataset()

for i in range(10):
    batch=a.next_batch(1)
    a.check_data_distribution(batch)
'''
    
