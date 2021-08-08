from data import ReadData
from LSTM import LSTMModel
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TODO some issue with CUDA
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

    # try LSTM
    verbose, epochs, batch_size = 1, 50, 32
    lstm_nn = LSTMModel(reader.data_x, reader.data_y, verbose, epochs, batch_size)
    lstm_nn.model_train("plots/training/" + "lstm_ep50_bs32_layer100_0.67accu.png")
    lstm_nn.model_test()

    # try convLSTM
