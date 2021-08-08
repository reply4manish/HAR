import matplotlib.pyplot as plt


def plot_progress(path_toSave, history):
    plt.plot(history.history['loss'], label='Loss (training data)')
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    print("history loss: ", history.history['loss'])
    plt.title('Training')
    plt.ylabel('value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    plt.savefig(path_toSave)
    plt.show()