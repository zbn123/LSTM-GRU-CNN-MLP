import lstm
import gru
import mlp
import cnnlstm
import lstmclassification
import mlpclassification
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


def plot_results(predicted_data, true_data,quant_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(quant_data, label='Quant Data')
    plt.plot(predicted_data, label='Prediction')

    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in xrange(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()

    plt.show()

# Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs = 20
    seq_len = 20

    print('> Loading data... ')

    # X_train, y_train, X_test, y_test = lstm.load_data()
    # X_train, y_train, X_test, y_test = cnn.load_data()
    # X_train, y_train, X_test, y_test = mlp.load_data()
    X_train, y_train, X_test, y_test, encoder = lstmclassification.load_data()
    # X_train, y_train, X_test, y_test = cnnlstm.load_data()
    # X_train, y_train, X_test, y_test = mlpclassification.load_data()

    # y_test = y_test.astype(np.float32, copy=False)
    print '> Data Loaded. Compiling...'

    # model = lstm.build_model([1, 8, 8, 1])
    # model = gru.build_model([1, 32, 64, 1])
    # model = mlp.build_model([1, 32, 64, 1])
    # model = cnn.build_model()
    # model = cnnlstm.build_model()
    model = lstmclassification.build_model()
    # model = mlpclassification.build_model([1, 16, 32, 64])

    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=epochs, verbose=1,
                     validation_split=0.1, callbacks=[early_stopping])

    #########################
    ## plot loss val_loss
    plt.figure(1)
    axes = plt.gca()
    x_min = hist.epoch[0]
    x_max = hist.epoch[-1] + 1
    axes.set_xlim([x_min, x_max])

    plt.scatter(hist.epoch, hist.history['loss'], color='g')
    plt.plot(hist.history['loss'], color='g', label='Training Loss')

    plt.scatter(hist.epoch, hist.history['val_loss'], color='b')
    plt.plot(hist.history['val_loss'], color='b', label='Validation Loss')

    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss & Validation Loss vs Epochs')
    plt.legend()

    score = model.evaluate(X_test, y_test, show_accuracy=True)
    print("Test Result: score- %.2f, accuracy- %.2f%%" % (score[0], score[1]*100))

    print("Generating test predictions...")
    predictions = model.predict_classes(X_test)
    sio.savemat('predictions_lstm.mat', {'predictions': predictions})
    predicted = encoder.inverse_transform(predictions)

    # predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    # predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    # predicted = lstm.predict_point_by_point(model, X_test)
    # predicted = mlpclassification.predict_point_by_point(model, X_test)
    print 'Training duration (s) : ', time.time() - global_start_time

    print("> Saved model to disk...")
    model.save('my_model_lstm.h5')                  # creates a HDF5 file 'my_model.h5'
    model.save_weights('my_model_weights_lstm.h5')  # save the weights of a model

    # quant_data = np.reshape(X_test[:, 0, 0, 0], (X_test[:, 0, 0, 0].shape[0], 1))
    # quant_data = np.reshape(X_test[:, 0], (X_test[:, 0].shape[0], 1))
    # err = np.subtract(y_test, quant_data)
    # error_energy = np.sum(np.power(err, 2))
    # energy = np.sum(np.power(y_test, 2))
    # print '> Calculate SNR ...'
    # snr = 10*math.log10(energy / error_energy)
    # print '> SNR : ', snr

    # print '> Calculate Gp ...'
    # y_test = y_test.astype(np.float32, copy=False)
    # err = np.subtract(y_test, predicted)
    # error_energy = np.sum(np.power(err, 2))
    # energy = np.sum(np.power(y_test, 2))
    # gp = 10*math.log10(energy / error_energy)
    # print '> Gp : ', gp

    # plot_results(predicted, y_test, X_test[:, 0, 0, 0]/3)
    # plot_results(predicted, encoder.inverse_transform(y_test), X_test[:, 0])
        # plot_results_multiple(predicted, y_test, 50)