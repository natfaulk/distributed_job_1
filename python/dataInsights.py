import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import process

WIDTH_M = 20
HEIGHT_M = 10
SQ_PER_M = 10

#pass list of x,y points
def makeDist(_list):
  dist = np.zeros((WIDTH_M*SQ_PER_M+1, HEIGHT_M*SQ_PER_M+1))

  for i in _list:
    x = i[0]
    y = i[1]
    x = math.floor((10-x)*SQ_PER_M)
    y = math.floor(y*SQ_PER_M)
    dist[x,y]+=1
  
  return dist

def mlListToXY(_list):
  out = []
  for i in _list:
    out.append([i['vive_x'],i['vive_z']])
  return out

def plotAllPixels(_df):
  mask = _df.columns.str.contains('p\d+')
  lines = _df.loc[:,mask].to_numpy()

  # lines=np.array(_df['thermo'].values.tolist())
  time=_df['time'].to_numpy()
  print(lines)
  plt.plot(time, lines)
  plt.show()

def plotD8x8(_d8x8):
  # plt.imshow(_d8x8, vmin=18, vmax=26)
  plt.imshow(_d8x8, vmin=0, vmax=1)
  plt.show()

def fovInfo(_angles1, _angles2):
  none=0
  one=0
  two=0
  both=0
  for i in range(len(_angles1)):
    a1 = process.angleInFov(_angles1[i])
    a2 = process.angleInFov(_angles2[i])

    if a1 and a2:
      both+=1
    elif a1:
      one+=1
    elif a2:
      two+=1
    else:
      none+=1

  print(f'Out of FOV: {none} ({round(100*none/len(_angles1),2)}%)')
  print(f'In FOV of 1: {one} ({round(100*one/len(_angles1),2)}%)')
  print(f'In FOV of 2: {two} ({round(100*two/len(_angles1),2)}%)')
  print(f'In FOV of both: {both} ({round(100*both/len(_angles1),2)}%)')

def ecdf(_errors, _xlabel=None):
  _errors=np.sort(_errors)
  n = _errors.size
  y = np.arange(1, n+1) / n
  plt.figure(figsize=(6,4))
  plt.plot(_errors,y)
  plt.ylabel('ECDF')
  if _xlabel is not None:
    plt.xlabel(_xlabel)
  plt.grid(b=True, which='major')
  plt.tight_layout()
  plt.show()

def allPixels(_df):
  df1 = process.getThermoColumns(_df)
  allframes=[]
  for i, row in df1.iterrows():
    frame=[]
    for i in range(64):
      frame.append(row[f'p{i}'])
    allframes.append(frame)

  allframes=np.array(allframes).T
  plt.imshow(allframes, cmap='gray', interpolation='nearest', aspect='auto')
  plt.show()

  return allframes



# if __name__ == '__main__':
#   train,valid,test = loadTxtFile.mlSplit('big_')

#   print(f'Training len: {len(train)}')
#   print(f'Validation len: {len(valid)}')
#   print(f'Testing len: {len(test)}')

#   train_dist = makeDist(mlListToXY(train))
#   valid_dist = makeDist(mlListToXY(valid))
#   test_dist = makeDist(mlListToXY(test))

#   train_max = train_dist.max()
#   valid_max = valid_dist.max()
#   test_max = test_dist.max()

#   all_max = max(train_max, valid_max, test_max)

#   # plt.figure()
#   f, axs = plt.subplots(1,3)
#   axs[0].imshow(train_dist, cmap='hot', interpolation='nearest', vmin=0, vmax=all_max)
#   axs[1].imshow(valid_dist, cmap='hot', interpolation='nearest', vmin=0, vmax=all_max)
#   axs[2].imshow(test_dist, cmap='hot', interpolation='nearest', vmin=0, vmax=all_max)
#   plt.show()

#   W = 5
#   H = 5
#   all = train+valid+test
#   ims = []
#   for i in range(W*H): 
#     ims.append(random.choice(all))
  
#   df=pd.DataFrame(ims)
#   thermo_df=Process.getThermoColumns(df)
#   maxv=thermo_df.max().max()
#   minv=thermo_df.min().min()

#   f, axs = plt.subplots(W,H)
#   axs = axs.flatten()
#   for i, row in df.iterrows():
#     d8x8 = Process.make8x8FromDF(row.to_frame().T)
#     axs[i].imshow(d8x8, interpolation='nearest', vmin=minv, vmax=maxv)
#   plt.show()








