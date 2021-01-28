import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras

from . import interp
from . import process

def train_model(train_x, train_y, _epochs=300):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(8, activation='relu'))
  model.add(keras.layers.Dense(20, activation='relu'))
  model.add(keras.layers.Dense(20, activation='relu'))
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
  history = model.fit(train_x, train_y, epochs=_epochs, batch_size=10, shuffle=True)

  plt.plot(history.history['mae'])
  # plt.plot(history.history['val_acc'])
  # plt.title('model accuracy')
  # plt.ylabel('accuracy')
  # plt.xlabel('epoch')
  # plt.legend(['train', 'val'], loc='upper left')
  plt.show()

  return model

def train_model_64(train_x, train_y, validation_x, validation_y, _epochs=300):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dense(20, activation='relu'))
  model.add(keras.layers.Dense(20, activation='relu'))
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
  history = model.fit(
    train_x,
    train_y,
    validation_data=(validation_x, validation_y),
    epochs=_epochs,
    batch_size=64,
    shuffle=True
    )

  plt.plot(history.history['mae'])
  plt.plot(history.history['val_mae'])
  plt.show()

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.show()

  return model

def train_model_dist(train_x, train_y, validation_x, validation_y, _epochs=300):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dense(5, activation='relu'))
  model.add(keras.layers.Dense(5, activation='relu'))
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
  history = model.fit(
    train_x,
    train_y,
    validation_data=(validation_x, validation_y),
    epochs=_epochs,
    batch_size=64,
    shuffle=True
    )

  plt.plot(history.history['mae'])
  plt.plot(history.history['val_mae'])
  plt.show()

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.show()

  return model

def train_model_singlecam(train_x, train_y, validation_x, validation_y, _epochs=300):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dense(100, activation='relu'))
  model.add(keras.layers.Dense(100, activation='relu'))
  model.add(keras.layers.Dense(2, activation='sigmoid'))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
  history = model.fit(
    train_x,
    train_y,
    validation_data=(validation_x, validation_y),
    epochs=_epochs,
    batch_size=64,
    shuffle=True
    )

  plt.plot(history.history['mae'])
  plt.plot(history.history['val_mae'])
  plt.show()

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.show()

  return model

def upscale(_data, _factor):
  out=[]
  for i in _data:
    t=np.array(i)
    t=t.reshape(8,8)
    t=interp.interp2d(t.tolist(), _factor)
    t=np.array(t).flatten()
    out.append(t.tolist())
  return out

def train_model_cnn(train_x, train_y, validation_x, validation_y, _epochs=300):
  print('upscaling test x')
  train_x=upscale(train_x, 5)
  print('upscaling valid x')
  validation_x=upscale(validation_x, 5)

  dim=int(math.sqrt(len(train_x[0])))
  print(f'Dim = {dim}')

  model = keras.Sequential()
  model.add(keras.layers.Reshape((dim, dim, 1),input_shape=(dim**2,)))
  model.add(keras.layers.Conv2D(32, (5, 5), activation='relu'))
  model.add(keras.layers.MaxPooling2D((2, 2)))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(keras.layers.MaxPooling2D((2, 2)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dense(1))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
  history = model.fit(
    train_x,
    train_y,
    validation_data=(validation_x, validation_y),
    epochs=_epochs,
    batch_size=10,
    shuffle=True
    )

  plt.plot(history.history['mae'])
  plt.plot(history.history['val_mae'])
  plt.show()

  return model


def train_knn(train_x, train_y, n_neighbours):
  classifier = KNeighborsRegressor(n_neighbors=n_neighbours)
  classifier.fit(train_x, train_y)
  return classifier

def train_random_forest(train_x, train_y):
  model=RandomForestRegressor()
  model.fit(train_x, train_y)
  return model

def genXYlist(_df):
  _df = _df[_df['angle'] < 30]
  _df = _df[_df['angle'] > -30]

  x = []
  y = []
  for i, row in _df.iterrows():
    d8x8 = process.make8x8FromDF(row.to_frame().T)
    d8x1 = d8x8.mean(1)
    x.append(d8x1.tolist())
    y.append((row['angle']+30)/60)
  
  return x,y

def genXYlist_64(_df):
  _df = _df[_df['angle'] < 30]
  _df = _df[_df['angle'] > -30]

  x = []
  y = []
  for i, row in _df.iterrows():
    d8x8 = process.make8x8FromDF(row.to_frame().T)
    x.append(d8x8.flatten().tolist())
    y.append((row['angle']+30)/60)
  
  return x,y

def genXYlist_dist(_df):
  _df = _df[_df['angle'] < 30]
  _df = _df[_df['angle'] > -30]

  x = []
  y = []
  for i, row in _df.iterrows():
    d8x8 = process.make8x8FromDF(row.to_frame().T)
    x.append(d8x8.flatten().tolist())
    y.append(row['dist']/6)
  
  return x,y

def genXYlist_pos(_df):
  _df = _df[_df['angle'] < 30]
  _df = _df[_df['angle'] > -30]

  x = []
  y = []
  indexes=[]
  for i, row in _df.iterrows():
    d8x8 = process.make8x8FromDF(row.to_frame().T)
    x.append(d8x8.flatten().tolist())
    y.append([
      (row['pos'][0]+2.5)/5,
      row['pos'][2]/5
    ])

    indexes.append(i)
  
  return x,y,indexes