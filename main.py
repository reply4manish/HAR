from data import ReadData

if __name__ == "__main__":
    path = 'FORTH_TRACE_DATASET/FORTH_TRACE_DATASET-master'
    all_props = ["Device ID", "accelerometer x", "accelerometer y", "accelerometer z",
                 "gyroscope x", "gyroscope y", "gyroscope z", "magnetometer x",
                 "magnetometer y", "magnetometer z", "Timestamp", "Activity Label"]
    training_props = [
        "accelerometer x", "accelerometer y", "accelerometer z", "gyroscope x", "gyroscope y",
        "gyroscope z", "magnetometer x", "magnetometer y", "magnetometer z"]
    truth_label = "Activity Label"
    window_size = 256
    overlap = 0.50
    reader = ReadData(path, all_props, training_props, window_size, truth_label, overlap)
    reader.read_data()

