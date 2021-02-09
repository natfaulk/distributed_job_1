import json
import os
import random

from sklearn.ensemble import RandomForestRegressor
from numpy.random import seed
import numpy as np

from . import loader
from . import process
from . import utils
from . import dataSplit
from . import distModel

def loadNF():
  _toload='nathaniel'
  COMBINED_FILENAME=f'{_toload}.json'
  data=loader.combinedFromFile(COMBINED_FILENAME)
  if data==None:
    print('Data not pre-processed. Combining vive and thermo data...')
    data = loader.combined('csi_nathaniel.txt', ['dev1_csi_nathaniel.txt', 'dev2_csi_nathaniel.txt'])
    process.combinedPostLoad(data)
    loader.combinedToFile(data, COMBINED_FILENAME)
  
  data['thermo'][0]['pos']=data['vive']
  data['thermo'][1]['pos']=data['vive']
  ML_INDEX_FILENAME=f'{_toload}_indexes.json'
  train_i,valid_i,test_i=dataSplit.indexesFromFile(ML_INDEX_FILENAME)
  validation_df=dataSplit.filterDFfromIndexes(data['thermo'][0], data['thermo'][1], valid_i)
  validation_x,validation_y = distModel.genXYlist_dist(validation_df)
  return validation_x,validation_y

def checkPredictions(_pred, _yvals, _silent=False):
  predictions = (np.array(_pred)*6)
  predictions=predictions.flatten()
  validation_y=(np.array(_yvals)*6)
  errors = np.abs(np.subtract(predictions, validation_y))

  out={
    'median': np.median(errors),
    '95pcnt': np.percentile(errors,95),
    'max': np.max(errors),
    'rmse': utils.rmse(errors),
    'errors': errors.tolist()
  }

  return out

def makeModelPath(_ne,_md,_mln,_mss,_msl):
  fname=f'm_{_ne}_{_md}_{_mln}_{_mss}_{_msl}'
  return fname

def run(_ne,_md,_mln,_mss,_msl):
  os.environ['PYTHONHASHSEED']=str(42)
  seed(42)
  random.seed(42)

  RF_MODEL_NAME=makeModelPath(_ne,_md,_mln,_mss,_msl)

  # LOAD
  training_x,training_y,validation_x,validation_y,test_x,test_y=distModel.loadTraining()
  testnf_x, testnf_y = loadNF()

  print(f'Training for n_estimators {_ne}, max depth {_md}, max_leaf_nodes {_mln}, min_samples_split {_mss}, min_samples_leaf {_msl}:')
  model=RandomForestRegressor(random_state=42, n_estimators=_ne, max_depth=_md, max_leaf_nodes=_mln, min_samples_split=_mss, min_samples_leaf=_msl)
  model.fit(training_x, training_y)

  predictions_rf=model.predict(validation_x)
  predictions_n=model.predict(testnf_x)

  out={
    'valid':checkPredictions(predictions_rf, validation_y),
    'n_test':checkPredictions(predictions_n, testnf_y)
  }

  out['n_estimators']=_ne
  out['max_depth']=_md
  out['max_leaf_nodes']=_mln
  out['min_samples_split']=_mss
  out['min_samples_leaf']=_msl

  filepath=os.path.join(loader.ML_MODELS_PATH, f'${RF_MODEL_NAME}.json')
  with open(filepath, 'w') as outfile:
    json.dump(out, outfile)
  
  return filepath


if __name__ == '__main__':
  run(1000,None,None,2,1)
  


