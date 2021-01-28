import json
import os
import random

from . import dataInsights
from . import loader
from . import process

def makeIndexes(_angles1, _angles2):
  train_i=[]
  valid_i=[]
  test_i=[]
  TRAIN_SPLIT=0.8
  VALID_SPLIT=0.9
  for i in range(len(_angles1)):
    a1 = process.angleInFov(_angles1[i])
    a2 = process.angleInFov(_angles2[i])
    
    if a1 or a2:
      n = random.random()
      if n < TRAIN_SPLIT:
        train_i.append(i)
      elif n < VALID_SPLIT:
        valid_i.append(i)
      else:
        test_i.append(i)
  
  print('Data split:')
  print(f'Training len: {len(train_i)} ({round(100*len(train_i)/len(_angles1),2)}%)')
  print(f'Validation len: {len(valid_i)} ({round(100*len(valid_i)/len(_angles1),2)}%)')
  print(f'Testing len: {len(test_i)} ({round(100*len(test_i)/len(_angles1),2)}%)')

  
  print('Train FOV info:')
  dataInsights.fovInfo([_angles1[i] for i in train_i], [_angles2[i] for i in train_i])
  print('Valid FOV info:')
  dataInsights.fovInfo([_angles1[i] for i in valid_i], [_angles2[i] for i in valid_i])
  print('Test FOV info:')
  dataInsights.fovInfo([_angles1[i] for i in test_i], [_angles2[i] for i in test_i])

  return train_i,valid_i,test_i

def makeIndexesTimeSeries(_angles1, _angles2):
  train_i=[]
  valid_i=[]
  test_i=[]
  
  for i in range(len(_angles1)):
    a1 = process.angleInFov(_angles1[i])
    a2 = process.angleInFov(_angles2[i])
    
    if a1 or a2:
      if i < len(_angles1)/2:
        train_i.append(i)
      else:
        valid_i.append(i)
        
  return train_i,valid_i,test_i

# _infov is a list with whether each index is in fov [True, False, False.... etc]
def OH_makeIndexes(_infov):
  train_i=[]
  valid_i=[]
  test_i=[]
  TRAIN_SPLIT=0.8
  VALID_SPLIT=0.9
  for i in range(len(_infov)):
    if _infov[i]:
      n = random.random()
      if n < TRAIN_SPLIT:
        train_i.append(i)
      elif n < VALID_SPLIT:
        valid_i.append(i)
      else:
        test_i.append(i)
  
  print('Data split:')
  print(f'Training len: {len(train_i)} ({round(100*len(train_i)/len(_infov),2)}%)')
  print(f'Validation len: {len(valid_i)} ({round(100*len(valid_i)/len(_infov),2)}%)')
  print(f'Testing len: {len(test_i)} ({round(100*len(test_i)/len(_infov),2)}%)')

  return train_i,valid_i,test_i
  

def indexesToFile(_filename,_train_i,_valid_i,_test_i):
  FILENAME=os.path.join(loader.ML_DATA_PATH, _filename)

  out = {
    'train':_train_i,
    'valid':_valid_i,
    'test':_test_i
  }

  with open(FILENAME, 'w') as outfile:
    json.dump(out, outfile)

def indexesFromFile(_filename):
  FILENAME=os.path.join(loader.ML_DATA_PATH, _filename)

  try:
    with open(FILENAME) as f:
      data = json.load(f)

      return data['train'],data['valid'],data['test']
  except IOError:
    print(f'File {FILENAME} does not exist.')
    return None,None,None

def filterDFfromIndexes(_df1, _df2, _indexes):
  out=_df1.iloc[_indexes]
  out=out.append(_df2.iloc[_indexes])
  return out
