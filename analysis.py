import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def plot_activity_label(x, y, name):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.plot(x, y)
    ax.xaxis.grid(True, which='major')
    ax.xaxis.set_major_locator(MultipleLocator(256))
    ax.xaxis.set_minor_locator(MultipleLocator(16))
    plt.xticks(rotation=45, ha='right')
    plt.savefig('plots/overlap50percent/' + "overlap_" + name, dpi=500)
    # plt.show()


def plot_data(file_path='', file_name='', no_overlap=False):
    """
    sampling frequency is 51.2 or 51.2 data points per second. we would sample
    data in 5sec block i.e. 256 data points per window.
    """
    count = 0
    offset = 256 if no_overlap else 128
    x_data_plot = list()
    y_data_plot = list()
    file_name_path = os.path.join(file_path, file_name)
    print(file_name_path)
    data_frame = pd.read_csv(file_name_path, sep=',',
                             names=["Device ID", "accelerometer x", "accelerometer y", "accelerometer z",
                                    "gyroscope x", "gyroscope y", "gyroscope z", "magnetometer x",
                                    "magnetometer y",
                                    "magnetometer z", "Timestamp", "Activity Label"])
    # we want to sample or window the data, sample of 5sec will define one feature vector
    for idx in range(0, len(data_frame), offset):
        slice_ = data_frame[idx: idx + 256]
        if slice_.shape[0] < 256:  # There are uneven number of rows in files, for now we don't sample a window if left
            # over data points are les than 256
            count = count + 1
            break
        x_data = [x for x in range(idx, idx + 256)]
        x_data_plot.extend(x_data)
        y_data_plot.extend(slice_['Activity Label'].to_list())
        y = slice_['Activity Label'].value_counts()[:1].index.tolist()[0]
        if y > 5:
            continue
    print("breaking", count)
    print(len(x_data_plot), len(y_data_plot))
    plot_activity_label(x_data_plot[:12800], y_data_plot[:12800], file_name + '.png')


if __name__ == "__main__":
    file_path = 'FORTH_TRACE_DATASET/FORTH_TRACE_DATASET-master'
    loaded_x = list()
    loaded_y = list()
    for directory in os.listdir(file_path):
        plot_data(os.path.join(file_path, directory), file_name=directory + "dev2.csv", no_overlap=True)
