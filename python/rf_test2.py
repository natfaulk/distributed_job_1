import os
import random
import pickle

from sklearn.ensemble import RandomForestRegressor
from numpy.random import seed

from . import entry
from . import loader

def makeModelPath(_ne,_md,_mln,_mss,_msl):
  fname=f'm_{_ne}_{_md}_{_mln}_{_mss}_{_msl}'
  return fname

def run(_ne,_md,_mln,_mss,_msl):
  os.environ['PYTHONHASHSEED']=str(42)
  seed(42)
  random.seed(42)

  RF_MODEL_NAME=makeModelPath(_ne,_md,_mln,_mss,_msl)

  # LOAD
  training_x,training_y,validation_x,validation_y=entry.loadTraining()
  print(f'Training for n_estimators {_ne}, max depth {_md}, max_leaf_nodes {_mln}, min_samples_split {_mss}, min_samples_leaf {_msl}:')
  model=RandomForestRegressor(random_state=42, n_estimators=_ne, max_depth=_md, max_leaf_nodes=_mln, min_samples_split=_mss, min_samples_leaf=_msl)
  model.fit(training_x, training_y)

  MODEL_PATH=os.path.join(loader.ML_MODELS_PATH, RF_MODEL_NAME)
  pickle.dump(model, open(MODEL_PATH, 'wb'))
  return MODEL_PATH

if __name__ == '__main__':
  run(1000,None,None,2,1)
  


