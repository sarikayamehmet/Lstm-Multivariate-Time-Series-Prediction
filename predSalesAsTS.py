import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

my_data= pd.read_csv('salesDataByDay.csv', delimiter=";")
dataDummy = pd.get_dummies(my_data, columns=['month'])
dataDummy = pd.get_dummies(dataDummy, columns=['day'])
testDataCount = 15
nrowDf = dataDummy.shape[0]
train_df = dataDummy.loc[0:(nrowDf - testDataCount - 1)]
test_df = dataDummy.loc[(nrowDf - testDataCount):nrowDf]
# pick a window size of 7
sequence_length = 7
# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

sequence_cols = ['sales', 'bid', 'opp_cpc', 'meanImpr', 'totImpr', 'meanClicks', 'sumClicks', 'meanCost', 'sumCost', 'avg_cpc', 'top_pos_share', 'impr_share', 'outbid_ratio', 'beat', 'meet', 'lose', 'unavailability', 'meanPotential', 'sumPotential', 'meanBookings', 'sumBookings', 'opp_diff', 'ctr', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7']
seq_gen = gen_sequence(train_df, sequence_length, sequence_cols)
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]
label_gen = gen_labels(train_df, sequence_length, ['target'])
label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape
seq_array = seq_array.reshape(label_array.shape[0],sequence_length,len(sequence_cols))
print(label_array.shape)
print(seq_array.shape)
##################################
# Modeling
##################################

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

nb_features = seq_array.shape[2]
nb_out = 1

model = Sequential()
model.add(LSTM( input_shape=(sequence_length, nb_features), units=100, return_sequences=True))
model.add(Dropout(0.01))
model.add(LSTM( units=50, return_sequences=False))
model.add(Dropout(0.01))
model.add(Dense(units=nb_out))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae',r2_keras])
print(model.summary())
history = model.fit(seq_array, label_array, epochs=1000, batch_size=2, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', min_delta=0, patience=10, verbose=0, mode='min')])
# list all data in history
print(history.history.keys())
# summarize history for R^2
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model r^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for MAE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('\nMAE: {}'.format(scores[1]))
print('\nR^2: {}'.format(scores[2]))

##################################
# EVALUATE ON TEST DATA
##################################

# We pick the last sequence for each id in the test data
seq_test = gen_sequence(test_df, sequence_length, sequence_cols)
seq_test_array = np.concatenate(list(seq_test)).astype(np.float32)
label_test = gen_labels(test_df, sequence_length, ['target'])
label_test_array = np.concatenate(label_test).astype(np.float32)
label_test_array.shape
seq_test_array = seq_test_array.reshape(label_test_array.shape[0],sequence_length,len(sequence_cols))
print(label_test_array.shape)
print(seq_test_array.shape)
# test metrics
scores_test = model.evaluate(seq_test_array, label_test_array, verbose=2)
print('\nMAE: {}'.format(scores_test[1]))
print('\nR^2: {}'.format(scores_test[2]))

y_pred_test = model.predict(seq_test_array)
y_true_test = label_test_array
# Plot in blue color the predicted data and in green color the
# actual data to verify visually the accuracy of the model.
fig_verify = plt.figure(figsize=(10, 10))
plt.plot(y_pred_test, color="blue")
plt.plot(y_true_test, color="green")
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()
