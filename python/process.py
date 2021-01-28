import copy
import math
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from . import utils as Utils

# x axis is inverted.
# ie looking from lighthouse pespective - +x is left, +y up, +z forward
actuals = [
  [-2.0, 0.0, 1.0 ],
  [-1.0, 0.0, 1.0 ],
  [ 0.0, 0.0, 1.0 ],
  [ 0.0, 0.0, 2.0 ],
  [ 0.0, 0.0, 3.0 ]
]

ANCHOR_COUNT = len(actuals)

def vive(_data, _drawing=False):
  intervals = []
  start = 0
  prev = False
  for i, row in _data.iterrows():
    if row['y'] < -1.0:
      if not prev:
        start = i
        prev = True
    else:
      if prev:
        intervals.append([start, i])
        prev = False

  if prev:
    intervals.append([start, i])
  
  # get midpoints and use them as the cordinates. Might be better to average over a larger section
  mids = []
  count = 1
  # print midpoint
  for i in intervals:
    if count > ANCHOR_COUNT:
      print("Too many anchors - check this!")
      break
    temp=_data.iloc[(i[0]+i[1])//2,:]
    print(count, temp['x'], temp['y'], temp['z'])
    mids.append([temp['x'], temp['y'], temp['z']])
    count+=1

  if len(mids) < ANCHOR_COUNT:
    sys.exit('Not enough calibration points')
  
  THETA = math.atan2(
    (mids[2][2]-mids[0][2]),
    (mids[2][0]-mids[0][0])
  )

  mids_rot = []
  for i in mids:
    mids_rot.append(Utils.rotate(i, THETA, 'y'))

  mids_trans = []
  dX = actuals[0][0]-mids_rot[0][0]
  dY = actuals[0][1]-mids_rot[0][1]
  dZ = actuals[0][2]-mids_rot[0][2]
  for i in mids_rot:
    mids_trans.append(Utils.translate(i, [dX,dY,dZ]))

  # drawing
  if _drawing:  
    actuals_df = pd.DataFrame(actuals)
    actuals_df.columns = ['x', 'y', 'z']
    mids_df = pd.DataFrame(mids)
    mids_df.columns = ['x', 'y', 'z']
    mids_trans_df = pd.DataFrame(mids_trans)
    mids_trans_df.columns = ['x', 'y', 'z']
    mids_rot_df = pd.DataFrame(mids_rot)
    mids_rot_df.columns = ['x', 'y', 'z']

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-4, 4)
    ax.set_ylim3d(-4, 4)
    ax.set_zlim3d(-4, 4)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')

    ax.scatter(actuals_df['x'], actuals_df['z'], actuals_df['y'])
    ax.scatter(mids_df['x'], mids_df['z'], mids_df['y'])
    ax.scatter(mids_trans_df['x'], mids_trans_df['z'], mids_trans_df['y'])
    ax.scatter(mids_rot_df['x'], mids_rot_df['z'], mids_rot_df['y'])
    plt.show()
  
  out = []
  for i, row in _data.iterrows():
    temp = Utils.rotate([row['x'],row['y'],row['z']], THETA, 'y')
    out.append(Utils.translate(temp, [dX,dY,dZ]))
  return out

#thermo to nx65 matrix - n rows 64 columns of pixel data + thermistor
def thermoFromDF(_data):
  out=[]
  for i, row in _data.iterrows():
    row_out = []
    for j in range(64):
      row_out.append(row[f'p{j}'])
    row_out.append(row['thermistor'])

    out.append(row_out)

  return out

def thermoToDF(_data):
  cols = [f'p{i}' for i in range(64)]
  cols.append('thermistor')
  df = pd.DataFrame(_data, columns=cols)

  return df

def make8x8FromDF(_df):
  mask = _df.columns.str.contains('p\d+')
  arr = _df.loc[:,mask].to_numpy()
  arr = arr.reshape((8,8))
  # arr = arr.transpose()
  return arr

# takes a combined data df
def findBackgroundSamples(_df):
  mask = (_df['angle']<-40) | (_df['angle']>40)
  df_back = _df.loc[mask]
  return df_back

def getThermoColumns(_df):
  mask = _df.columns.str.contains('p\d+')
  return _df.loc[:,mask]

def calculateAngles(_vivedata, _campos, _camrotation=0):
  out=[]

  for i in _vivedata:
    # out.append(math.degrees(math.atan2(vv[0], vv[2])))
    xpos = i[0]-_campos[0]
    ypos = i[2]-_campos[1]

    if _camrotation != 0:
      newpt = Utils.rotate([xpos,ypos,0], _camrotation, 'z')
      xpos = newpt[0]
      ypos = newpt[1]

    ang = math.degrees(math.atan2(xpos, ypos))
    # if ang > -30 and ang < 30:
    #   out.append([xpos, ypos])
    out.append(ang)

  return out

def calculateDists(_vivedata, _campos):
  out=[]
  for i in _vivedata:
    xpos = i[0]
    ypos = i[2]

    out.append(math.hypot(xpos-_campos[0], ypos-_campos[1]))
  return out

def normalise(_df):
  # normalise
  thermo_df=getThermoColumns(_df)
  maxv=thermo_df.max().max()
  minv=thermo_df.min().min()
  
  mask = _df.columns.str.contains('p\d+')
  _df.loc[:,mask] = (_df.loc[:,mask] - minv)/(maxv-minv)

def normalise2(_df):
  thermo_df=getThermoColumns(_df)
  mean=thermo_df.stack().mean()
  std=thermo_df.stack().std()
  
  mask = _df.columns.str.contains('p\d+')
  _df.loc[:,mask] = 0.5+(_df.loc[:,mask] - mean)/(std*20)

def backgroundSubtract(_df):
  df_back = findBackgroundSamples(_df).mean().to_frame().T
  df_back = getThermoColumns(df_back).iloc[0]

  # background subtract
  mask = _df.columns.str.contains('p\d+')
  _df.loc[:,mask]=_df.loc[:,mask].subtract(df_back)

def backgroundSubtractSVD(_df):
  df1 = getThermoColumns(_df)
  allframes=[]
  for i, row in df1.iterrows():
    frame=[]
    for i in range(64):
      frame.append(row[f'p{i}'])
    allframes.append(frame)

  print('making array')
  allframes=np.array(allframes).T
  dim_s=1
  dim_e=64

  print('Doing SVD')
  u,s,vh=np.linalg.svd(allframes)

  print('re assembling')
  # @ is matrix multiplication
  foreground = np.array(u[:, dim_s:dim_e]) @ np.diag(s[dim_s:dim_e]) @ np.array(vh[dim_s:dim_e, :])
  # background = np.array(u[:, :dim_s]) @ np.diag(s[:dim_s]) @ np.array(vh[:dim_s, :])
  
  for i in range(64):
    _df[f'p{i}']=foreground[i]

 
KERNEL = [0.1784,0.210431,0.222338,0.210431,0.1784]
# KERNEL = [0.06136,0.24477,0.38774,0.24477,0.06136]
# KERNEL = [0.071303,0.131514,0.189879,0.214607,0.189879,0.131514,0.071303]
# KERNEL = [0.000229,0.005977,0.060598,0.241732,0.382928,0.241732,0.060598,0.005977,0.000229]
# KERNEL = [1/9]*9
# Kernel calculated from here
# http://dev.theomader.com/gaussian-kernel-calculator/
def smoothGaussian(_data):
  thermo_array=thermoFromDF(_data)
  ta_copy=copy.deepcopy(thermo_array)

  for j in range(65):
    for i in range(5,len(thermo_array)-5):
      # v1=KERNEL[0]*thermo_array[i-4][j]
      # v2=KERNEL[1]*thermo_array[i-3][j]
      # v3=KERNEL[2]*thermo_array[i-2][j]
      # v4=KERNEL[3]*thermo_array[i-1][j]
      # v5=KERNEL[4]*thermo_array[i][j]
      # v6=KERNEL[5]*thermo_array[i+1][j]
      # v7=KERNEL[6]*thermo_array[i+2][j]
      # v8=KERNEL[7]*thermo_array[i+3][j]
      # v9=KERNEL[8]*thermo_array[i+4][j]
      # ta_copy[i][j]=v1+v2+v3+v4+v5+v6+v7+v8+v9
      v1=KERNEL[0]*thermo_array[i-2][j]
      v2=KERNEL[1]*thermo_array[i-1][j]
      v3=KERNEL[2]*thermo_array[i][j]
      v4=KERNEL[3]*thermo_array[i+1][j]
      v5=KERNEL[4]*thermo_array[i+2][j]
      ta_copy[i][j]=v1+v2+v3+v4+v5

  return thermoToDF(ta_copy)

def angleInFov(_angle):
  if _angle < -30:
    return False
  if _angle > 30:
    return False
  return True

def getIndexesInFOV_dual(_cam1, _cam2, _indexes):
  out=[]

  for i in _indexes:
    cam1fov = angleInFov(_cam1.iloc[i]['angle'])
    cam2fov = angleInFov(_cam2.iloc[i]['angle'])

    if cam1fov and cam2fov:
      out.append(i)
  
  return out

# Overhead camera check if position in FOV
# _pos is xy position that checking is in FOV
# _camPos is XYZ pos of camera where Z is height above floor
# FOV angle hardcoded to 60 (therefore half angle is 30)
def OH_posInFOV(_pos, _camPos):
  fov_dist=math.tan(math.radians(30))*_camPos[2]
  inFOV_x=abs(_pos[0]-_camPos[0])<fov_dist
  inFOV_y=abs(_pos[1]-_camPos[1])<fov_dist

  return inFOV_x and inFOV_y

def combinedPostLoad(_df):
  print('Gaussian smoothing between frames')
  _df['thermo'][0]=smoothGaussian(_df['thermo'][0])
  _df['thermo'][1]=smoothGaussian(_df['thermo'][1])

  print('calculating angles')
  angles1=calculateAngles(_df['vive'], (0,0))
  angles2=calculateAngles(_df['vive'], (-2.5, 2.5), math.pi/2)
  print('calculating distances')
  dists1=calculateDists(_df['vive'], (0,0))
  dists2=calculateDists(_df['vive'], (-2.5, 2.5))
  print('saving back to dataframe')
  _df['thermo'][0]=_df['thermo'][0].assign(angle=angles1)
  _df['thermo'][1]=_df['thermo'][1].assign(angle=angles2)
  _df['thermo'][0]=_df['thermo'][0].assign(dist=dists1)
  _df['thermo'][1]=_df['thermo'][1].assign(dist=dists2)

  print('background subtract')
  backgroundSubtractSVD(_df['thermo'][0])
  backgroundSubtractSVD(_df['thermo'][1])
  print('normalise')
  normalise(_df['thermo'][0])
  normalise(_df['thermo'][1])

  return angles1,angles2

def combinedPostLoad2(_df):
  print('Gaussian smoothing between frames')
  _df['thermo'][0]=smoothGaussian(_df['thermo'][0])
  _df['thermo'][1]=smoothGaussian(_df['thermo'][1])

  print('calculating angles')
  angles1=calculateAngles(_df['vive'], (0,0))
  angles2=calculateAngles(_df['vive'], (-2.5, 2.5), math.pi/2)
  print('calculating distances')
  dists1=calculateDists(_df['vive'], (0,0))
  dists2=calculateDists(_df['vive'], (-2.5, 2.5))
  print('saving back to dataframe')
  _df['thermo'][0]=_df['thermo'][0].assign(angle=angles1)
  _df['thermo'][1]=_df['thermo'][1].assign(angle=angles2)
  _df['thermo'][0]=_df['thermo'][0].assign(dist=dists1)
  _df['thermo'][1]=_df['thermo'][1].assign(dist=dists2)

  print('background subtract')
  backgroundSubtractSVD(_df['thermo'][0])
  backgroundSubtractSVD(_df['thermo'][1])
  print('normalise')
  normalise2(_df['thermo'][0])
  normalise2(_df['thermo'][1])

  return angles1,angles2
