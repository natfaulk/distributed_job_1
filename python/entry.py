import os

from . import dataSplit
from . import loader
from . import process

COMBINED_FILENAME='big_SVD.json'
ML_INDEX_FILENAME='big_indexes_SVD.json'
ERRORS_OUTPUT_FILENAME='out_SVD_64.json'
ML_MODEL_NAME='mymodel_SVD_64'
MODEL_PATH = os.path.join(loader.ML_MODELS_PATH, ML_MODEL_NAME)

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
  training_x,training_y = genXYlist_64(training_df)
  validation_x,validation_y = genXYlist_64(validation_df)

  return training_x,training_y,validation_x,validation_y
