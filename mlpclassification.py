import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
import scipy.io as sio
from keras.optimizers import RMSprop
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.optimizers import SGD

warnings.filterwarnings("ignore")


def load_data():
    # f = open(filename, 'rb').read()
    # data = f.split('\n')

    # sequence_length = seq_len + 1
    # result = []
    # for index in range(len(data) - sequence_length):
    #    result.append(data[index: index + sequence_length])

    # if normalise_window:
    #     result = normalise_windows(result)

    # plt.plot(result)
    # plt.show()

    mat_input = 'inputShortV.mat'
    input = sio.loadmat(mat_input)
    input = 3 * input['inputShortV']

    # plt.plot(input)
    # plt.show()

    mat_target = 'targetCodeShortV.mat'
    target = sio.loadmat(mat_target)
    target = target['targetCodeShortV']

    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    # ax.plot(target, label='True Data')
    # plt.plot(input[:,1]/3,label='Quantization')
    # plt.legend()
    # plt.show()

    input = np.array(input)
    target = np.array(target)

    scaler = preprocessing.MinMaxScaler()
    input = scaler.fit_transform(input)

    row = int(round(0.9 * input.shape[0]))
    train_input = input[:row, :]
    train_target = target[:row, :]

    train = np.column_stack((train_input, train_target))
    np.random.shuffle(train)

    x_train = train[:, 0:20]
    y_train = train[:, 20:84]
    # plt.plot(y_train)
    # plt.show()
    train_input = []
    train_target = []
    train = []

    x_test = input[row:, :]
    y_test = target[row:, :]
    input = []
    target = []

    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # (examples, values in sequences, dim. of each value)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))     # (examples, values in sequences, dim. of each value)

    return [x_train, y_train, x_test, y_test]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)

    return normalised_data


def build_model(layers):
    model = Sequential()

    model.add(Dense(layers[1], input_shape=(20,), activation='relu'))
    model.add(Dropout(0.2))  # Dropout overfitting

    # model.add(Dense(layers[2],activation='tanh'))
    # model.add(Dropout(0.2))  # Dropout overfitting

    model.add(Dense(layers[2], activation='relu'))
    model.add(Dropout(0.2))  # Dropout overfitting

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("softmax"))

    model.summary()

    start = time.time()
    # sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss="mse", optimizer=sgd)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) # Nadam RMSprop()
    print "Compilation Time : ", time.time() - start
    return model


def predict_point_by_point(model, data):
    # Predict each time step given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    # predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in xrange(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)

    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in xrange(len(data)/prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in xrange(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    return prediction_seqs