import os
import pickle
import random
import shutil

from numpy.random import seed
from tensorflow import keras
import tensorflow as tf

from . import distModel
from . import loader
from . import utils

MODEL_DIR=os.path.join(loader.ML_MODELS_PATH)

def makeModelPath(_nlayers,_n_pcp,_hlayer,_olayer):
  fname=f'm_{_nlayers}_{_n_pcp}_{_hlayer}_{_olayer}'
  return fname

def modelNameToParams(_modelName):
  params=_modelName.split('_')

  # params[0] *should* always be 'm'
  out={
    'n_layers':int(params[1]),
    'n_pcptns':int(params[2]),
    'h_layer_func':params[3],
    'out_layer_func':params[4]
  }

  return out

def run(_l,_p,_h,_o):
  os.environ['PYTHONHASHSEED']=str(42)
  seed(42)
  random.seed(42)
  tf.random.set_seed(42)

  training_x,training_y,validation_x,validation_y,test_x,test_y=distModel.loadTraining()

  print('Current Model:')
  print('N layers:',_l)
  print('N perceptrons:',_p)
  print('Hidden layer func:',_h)
  print('Output layer func:',_o)

  MODEL_NAME=makeModelPath(_l,_p,_h,_o)
  # make a sub dir to put the model and the history file in
  MODEL_PATH=os.path.join(MODEL_DIR, MODEL_NAME)
  utils.mkdir_p(MODEL_PATH)

  model = keras.models.Sequential()
  model.add(keras.layers.Dense(64, activation=_h))
  for i in range(_l):
    model.add(keras.layers.Dense(_p, activation=_h))
  model.add(keras.layers.Dense(1, activation=_o))

  es = keras.callbacks.EarlyStopping(verbose=1, patience=300)
  mc = keras.callbacks.ModelCheckpoint(os.path.join(MODEL_PATH,MODEL_NAME), monitor='val_mse', mode='min', verbose=1, save_best_only=True)
  
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])
  history = model.fit(
    training_x,
    training_y,
    validation_data=(validation_x, validation_y),
    epochs=1000,
    batch_size=64,
    shuffle=True,
    callbacks=[es, mc]
  )

  model=keras.models.load_model(os.path.join(MODEL_PATH,MODEL_NAME))
  train_acc = model.evaluate(training_x, training_y, verbose=0)
  test_acc = model.evaluate(validation_x, validation_y, verbose=0)
  
  print(f'Train acc: {train_acc}, Valid acc: {test_acc}')

  
  # model.save(MODEL_PATH)
  pickle.dump(history.history, open(os.path.join(MODEL_PATH,f'{MODEL_NAME}_history.sav'), 'wb'))

  # zip.modelAndHistory(MODEL_PATH)
  shutil.make_archive(f'{MODEL_PATH}_out', 'zip', MODEL_PATH)

  return f'{MODEL_PATH}_out.zip'

layers=[1,2,3]
pcptns=[5,50,500]
hidden_l=['linear','sigmoid','relu']
output_l=['linear','sigmoid','relu']

if __name__ == '__main__':
  for a in layers:
    for b in pcptns:
      for c in hidden_l:
        for d in output_l:
          run(a,b,c,d)




