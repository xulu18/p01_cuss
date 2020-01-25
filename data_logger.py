#-*-coding:utf-8-*-


import os
import time

class DataLogger(object):
    def __init__(self,str_path):
        '''
        path should be a folder rather than file
        '''
        assert os.path.isdir(str_path),\
               'str_path should be a folder'
        
        self.path_main=str_path
        
        self.path_log=str_path+'/log'
        self.path_model=str_path+'/model'
        self.path_output=str_path+'/output'
        #get time now as month-day-hour-minute
        self.time_stamp=time.strftime('%m_%d_%H_%M',time.localtime(time.time()))

        self.file_log=''
        self.init_folders(self.path_main)        
        self.init_file()

    def init_folders(self,str_path_main):
        assert os.path.exists(str_path_main),\
               'Mainpath for logger do not exist: '+str_path_main
        if not os.path.exists(str_path_main+'/log'):
            os.makedirs(str_path_main+'/log')
            self.file_log=self.get_filepath(str_path_main+'/log')
            self.log_print('Creating new folder: '+str_path_main+'/log')
        else:
            self.file_log=self.get_filepath(str_path_main+'/log')
            
          
        if not os.path.exists(str_path_main+'/model'):
            os.makedirs(str_path_main+'/model')
            self.log_print('Creating new folder: '+str_path_main+'/model')

        if not os.path.exists(str_path_main+'/output'):
            os.makedirs(str_path_main+'/output')
            self.log_print('Creating new folder: '+str_path_main+'/output')
            
            
    def get_filepath(self,str_path):
        '''
        return name of log file
        the naming rule is 'log'+time+'.txt'
        \str_path:string, path of folder prepare for logging
        '''
        if not os.path.isdir(str_path):
            os.makedirs(str_path)
        file_name=os.path.join(str_path,'log-'+self.time_stamp+'.txt')
        f=open(file_name,'w')
        f.close()
        return file_name
        
            
    def init_file(self):
        '''
        print discriptive text into log file
        '''
        self.log_print(time.strftime('%Y_',time.localtime(time.time()))+self.time_stamp)
        
        #print os.path.abspath(__file__)

    
    def log_print(self,str_content,str_end='\n'):
        '''
        print text to both screen and log file
        \str_content:string, content that you would like to print
        '''
        print(str_content,end=str_end)
        with open(self.file_log,'a') as f:
            f.write(str(str_content)+str_end)

#a=DataLogger(r'F:\Liu_my_toolbox\tensor\log')
