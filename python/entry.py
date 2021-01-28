import json
import shutil
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from . import dataSplit
from . import loader
from . import ml
from . import process

COMBINED_FILENAME='big_SVD.json'
ML_INDEX_FILENAME='big_indexes_SVD.json'
ERRORS_OUTPUT_FILENAME='out_SVD_64.json'
ML_MODEL_NAME='mymodel_SVD_64'
MODEL_PATH = os.path.join(loader.ML_MODELS_PATH, ML_MODEL_NAME)

def loadTraining():
  data=loader.combinedFromFile(COMBINED_FILENAME)
  if data==None:
    print('Data not pre-processed. Combining vive and thermo data...')
    data = loader.combined('big.txt', ['dev1_big.txt', 'dev2_big.txt'])
    process.combinedPostLoad(data)
    loader.combinedToFile(data, COMBINED_FILENAME)
  
  angles1=data['thermo'][0]['angle'].tolist()
  angles2=data['thermo'][1]['angle'].tolist()

  train_i,valid_i,test_i=dataSplit.indexesFromFile(ML_INDEX_FILENAME)
  if train_i == None:
    print('split index file not found, creating...')
    train_i,valid_i,test_i=dataSplit.makeIndexes(angles1, angles2)
    dataSplit.indexesToFile(ML_INDEX_FILENAME,train_i,valid_i,test_i)
 
  training_df = dataSplit.filterDFfromIndexes(data['thermo'][0], data['thermo'][1], train_i)
  validation_df = dataSplit.filterDFfromIndexes(data['thermo'][0], data['thermo'][1], valid_i)
  training_x,training_y = ml.genXYlist_64(training_df)
  validation_x,validation_y = ml.genXYlist_64(validation_df)

  return training_x,training_y,validation_x,validation_y

def loadModel():
  print('Loading angle MLP model')
  try:
    mlp_model=keras.models.load_model(MODEL_PATH)
    return mlp_model
  except IOError:
    print('Model not found')
    return None

def trainModel():
  print('Training angle MLP model')
  training_x,training_y,validation_x,validation_y=loadTraining()
  model = ml.train_model_64(training_x, training_y, validation_x, validation_y)
  model.save(MODEL_PATH)
  return model

def resetModel():
  try:
    shutil.rmtree(MODEL_PATH)
  except FileNotFoundError:
    print('Model already deleted...')

if __name__ == '__main__':
  training_x,training_y,validation_x,validation_y=loadTraining()
  model=loadModel()
  if model is None:
    model=trainModel()
  
  predictions = model.predict(validation_x)
  predictions = (np.array(predictions)*60)-30
  predictions=predictions.flatten()
  validation_y=(np.array(validation_y)*60)-30
  print(predictions.shape, validation_y.shape)
  errors = np.abs(np.subtract(predictions, validation_y))
  print(np.median(errors))

  df_err=pd.DataFrame(
  {
    # 'dist': valid_dists,
    'predict': predictions,
    'actual': validation_y,
    'errors': errors,
    # 'therm': therms
  })

  FILENAME = os.path.join(loader.OUTPUTS_PATH, ERRORS_OUTPUT_FILENAME)
  with open(FILENAME, 'w') as outfile:
    json.dump(df_err.to_dict('records'), outfile)

  df_err=df_err.sort_values('errors')
  print(df_err.head())
  plt.plot(df_err['predict'].values)
  plt.plot(df_err['actual'].values)
  # plt.plot(df_err['dist'].values)
  # plt.plot(df_err['therm'].values)
  plt.show()

  # ecdf
  errors=np.sort(errors)
  n = errors.size
  y = np.arange(1, n+1) / n
  plt.plot(errors,y)
  plt.show()


