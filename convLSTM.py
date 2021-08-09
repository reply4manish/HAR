from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ConvLSTM2D
from keras.layers import Flatten
from keras.layers import Dropout
from utils import plot_progress


class ConvLSTM:
    def __init__(self, data_x, data_y, verbose, epochs, batch_size):
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.30)

    def model_train(self, save_fig_path):
        # convert y to one hot encoding
        # remove zero offset
        self.y_train = self.y_train - 1
        self.y_train = to_categorical(self.y_train)
        # define model
        n_time_steps, n_features, n_outputs = self.x_train.shape[1], self.x_train.shape[2], self.y_train.shape[1]
        # reshape into subsequences (samples, time steps, rows, cols, channels)
        n_steps, n_length = 4, 64
        self.x_train = self.x_train.reshape((self.x_train.shape[0], n_steps, 1, n_length, n_features))
        # define model
        self.model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                                  input_shape=(n_steps, 1, n_length, n_features), return_sequences=True))
        self.model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu'))
        self.model.add(Dropout(0.30))
        self.model.add(Flatten())
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                 verbose=self.verbose)
        plot_progress(save_fig_path, history)

    def model_test(self):
        # convert y to one hot encoding
        # remove zero offset
        self.y_test = self.y_test - 1
        self.y_test = to_categorical(self.y_test)
        n_time_steps, n_features, n_outputs = self.x_test.shape[1], self.x_test.shape[2], self.y_test.shape[1]
        n_steps, n_length = 4, 64
        self.x_test = self.x_test.reshape((self.x_test.shape[0], n_steps, 1, n_length, n_features))
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size, verbose=self.verbose)
        print("Accuracy on test data {}:".format(accuracy))
