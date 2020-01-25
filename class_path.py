#-*-coding:utf-8-*-
import os
'''
A class designed for searching files with particular suffix.
Author :Mingyuan Liu
August, 24, 2018
'''

class MyClassPath(object):
    def __init__(self, str_list_path, str_suffix):
        self._root = []
        self._path = []
        self._num_files = 0
        self._suffix = str_suffix

        self.root(str_list_path)
        self.get_list_path(str_list_path, str_suffix)
        self.print_log()

    def root(self, str_list_root):
        # Get self._root
        self._root = str_list_root

    def num_files(self, length_path):
        self._num_files = length_path

    def get_list_path(self, str_list_path, str_suffix):
        assert isinstance(str_list_path,list), r'str_list_path should be a list'
        self._path = []
        for str_path in str_list_path:

            self.get_path(str_path, str_suffix, flag_clear=False)
        self.num_files(len(self._path))
        assert self._num_files == len(set(self._path)), r'Duplicates, folders may contain within each other!'

    def get_path(self, str_path, str_suffix, flag_clear=True):
        # Load files and subfiles under str_path with str_suffix as ext
        # str_suffix could both start with . & *
        # Refresh self._path & self._num_files
        if flag_clear is True:
            self._path = []

        if str_suffix[0] == r'.':
            for root, dirs, files in os.walk(str_path):
                for file in files:
                    if os.path.splitext(file)[-1] == str_suffix:
                        self._path.append(os.path.join(root, file))
        else:
            length_suffix = len(str_suffix)
            for root, dirs, files in os.walk(str_path):
                for file in files:
                    if file[-length_suffix :] == str_suffix:
                        self._path.append(os.path.join(root, file))
        if flag_clear is True:
            self.num_files(len(self._path))


    def print_log(self):
        # Plot info of MyClassPath
        print(r'# # # # # # #')
        #print(r'# Reading files from :')
        #print(r'# '+ self._root)
        print(self._suffix + ' as suffix')
        print(str(self._num_files) + r' files')
        print(r'e.g. ' + self._path[0])
        print(r'# # # # # # #')

'''
a = MyClassPath([r'G:\DataSet\mni-hsub25\mni-hisub25\mri_dataset\s01',
                 r'G:\DataSet\mni-hsub25\mni-hisub25\mri_dataset\s02',
                 r'G:\DataSet\mni-hsub25\mni-hisub25\mri_dataset\s03'], r'.gz')
'''
