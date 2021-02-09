import os

import numpy as np

from . import dataSplit
from . import loader
from . import process
from . import utils

COMBINED_FILENAME='big_SVD.json'
ML_INDEX_FILENAME='big_indexes_SVD.json'
ML_MODEL_NAME='mymodel_dist'
MODEL_PATH = os.path.join(loader.ML_MODELS_PATH, ML_MODEL_NAME)

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

  training_x,training_y = genXYlist_dist(training_df)
  validation_x,validation_y = genXYlist_dist(validation_df)
  test_x,test_y = genXYlist_dist(test_df)

  return training_x,training_y,validation_x,validation_y,test_x,test_y

