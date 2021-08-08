import numpy as np
import os
import pandas as pd


class ReadData:

    def __init__(self, folder_paths, data_props, use_data_props, window, label, data_overlap):
        self.folder_path = folder_paths
        self.all_data_properties = data_props
        self.use_props = use_data_props
        self.window_size = window
        self.truth_label = label
        self.data_overlap = data_overlap
        self.data_x = list()
        self.data_y = list()

    def load_files(self, file_path='', file_name=''):
        data_x_ = []
        data_y_ = []
        offset = 256 - 256 * self.data_overlap
        file_name_path = os.path.join(file_path, file_name)
        print(file_name_path)
        data_frame = pd.read_csv(file_name_path, sep=',', names=self.all_data_properties)
        # we want to sample or window the data, sample of 5sec will define one feature vector
        for idx in range(0, len(data_frame), offset):
            slice_ = data_frame[idx: idx + 256]
            if slice_.shape[0] < 256:
                break
            y = slice_[self.truth_label].value_counts()[:1].index.tolist()[0]
            if y > 5:  # we want only first 5 activities
                continue
            x = slice_[self.use_props]
            data_x_.append(x)
            data_y_.append(y)
        data_x_ = np.stack(data_x_, axis=0)
        data_y_ = np.vstack(data_y_)
        return data_x_, data_y_

    def read_data(self):
        for directory in os.listdir(self.folder_path):
            train_x, train_y = self.load_file(os.path.join(self.folder_path, directory),
                                              file_name=directory + "dev2.csv")
            self.data_x.append(train_x)
            self.data_y.append(train_y)
        self.data_x = np.vstack(self.data_x)
        self.data_y = np.vstack(self.data_y)
