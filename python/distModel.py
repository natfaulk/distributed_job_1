import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from . import dataInsights
from . import dataSplit
from . import loader
from . import ml
from . import process
from . import utils

COMBINED_FILENAME='big_SVD.json'
ML_INDEX_FILENAME='big_indexes_SVD.json'
ML_MODEL_NAME='mymodel_dist'
MODEL_PATH = os.path.join(loader.ML_MODELS_PATH, ML_MODEL_NAME)

def loadModel():
  try:
    mlp_model=keras.models.load_model(MODEL_PATH)
    return mlp_model
  except IOError:
    print('Model not found')
    return None

def trainModel():
  training_x,training_y,validation_x,validation_y,test_x,test_y=loadTraining()
  model=ml.train_model_dist(training_x, training_y, validation_x, validation_y)
  model.save(MODEL_PATH)
  return model

def resetModel():
  try:
    shutil.rmtree(MODEL_PATH)
  except FileNotFoundError:
    print('Model already deleted...')

def loadTraining():
  data=loader.combinedFromFile(COMBINED_FILENAME)
  if data==None:
    print('Data not pre-processed. Combining vive and thermo data...')
    data = loader.combined('big.txt', ['dev1_big.txt', 'dev2_big.txt'])
    process.combinedPostLoad(data)
    loader.combinedToFile(data, COMBINED_FILENAME)

  train_i,valid_i,test_i=dataSplit.indexesFromFile(ML_INDEX_FILENAME)
  training_df = dataSplit.filterDFfromIndexes(data['thermo'][0], data['thermo'][1], train_i)
  validation_df = dataSplit.filterDFfromIndexes(data['thermo'][0], data['thermo'][1], valid_i)
  test_df = dataSplit.filterDFfromIndexes(data['thermo'][0], data['thermo'][1], test_i)

  training_x,training_y = ml.genXYlist_dist(training_df)
  validation_x,validation_y = ml.genXYlist_dist(validation_df)
  test_x,test_y = ml.genXYlist_dist(test_df)

  return training_x,training_y,validation_x,validation_y,test_x,test_y

def run():
  training_x,training_y,validation_x,validation_y,test_x,test_y=loadTraining()
  model=loadModel()
  if model is None:
    model = trainModel()

  predictions = model.predict(validation_x)
  predictions = np.array(predictions)*6
  predictions=predictions.flatten()
  validation_y = np.array(validation_y)*6
  errors = np.abs(np.subtract(predictions, validation_y))
  print(np.median(errors))
  print(np.max(errors))
  print(np.percentile(errors,95))
  print(utils.rmse(errors))
  print(errors.shape)

  dataInsights.ecdf(errors)

  plt.plot(predictions)
  plt.plot(validation_y)
  plt.show()

if __name__ == '__main__':
  run()
  