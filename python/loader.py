import gzip
import json
import os

import numpy as np
import pandas as pd

from . import process
from . import utils

DATA_FOLDER = 'data'
VIVE_DATA_FOLDER = 'vive'
THERMO_DATA_FOLDER = 'thermo'
COMBINED_DATA_FOLDER = 'combined'
ML_DATA_FOLDER = 'ml'
OUTPUTS_FOLDER = 'outputs'
ML_FOLDER = 'models'

DATA_LINE_HEADER = '[data] '

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_PATH = os.path.join(BASE_PATH, DATA_FOLDER)
VIVE_DATA_PATH = os.path.join(DATA_PATH, VIVE_DATA_FOLDER)
THERMO_DATA_PATH = os.path.join(DATA_PATH, THERMO_DATA_FOLDER)
COMBINED_DATA_PATH = os.path.join(DATA_PATH, COMBINED_DATA_FOLDER)
ML_DATA_PATH = os.path.join(DATA_PATH, ML_DATA_FOLDER)
OUTPUTS_PATH = os.path.join(DATA_PATH, OUTPUTS_FOLDER)
ML_MODELS_PATH = os.path.join(BASE_PATH, ML_FOLDER)

utils.mkdir_p(OUTPUTS_PATH)
utils.mkdir_p(ML_MODELS_PATH)

def thermo(filename):
  FILENAME = os.path.join(THERMO_DATA_PATH, filename)
  inputFile = open(FILENAME, 'r') 
  lines = inputFile.readlines() 

  data = []
  for l in lines:
    l_decoded = decodeThermoLine(l)
    if l_decoded:
      data.append(l_decoded)

  df = pd.DataFrame(data)
  df['time'] = pd.to_datetime(df['time'], unit='ms')

  return df

def vive(filename):
  FILENAME = os.path.join(VIVE_DATA_PATH, filename)
  inputFile = open(FILENAME, 'r') 
  lines = inputFile.readlines() 

  data = []
  for l in lines:
    data.append(decodeViveLine(l))
  
  df = pd.DataFrame(data)
  df['time'] = pd.to_datetime(df['time'], unit='s')

  return df

def decodeViveLine(line):
  parsedLine = json.loads(line)

  out = {}
  out['time'] = parsedLine['time']
  out['x'] = parsedLine['pos'][0]
  out['y'] = parsedLine['pos'][1]
  out['z'] = parsedLine['pos'][2]
  out['roll'] = parsedLine['pos'][3]
  out['pitch'] = parsedLine['pos'][4]
  out['yaw'] = parsedLine['pos'][5]

  return out

def decodeThermoLine(line):
  temp = line.split('|')
  if len(temp) < 3:
    return None

  if not temp[2].startswith(DATA_LINE_HEADER):
    return None
  
  parsedLine = json.loads(temp[2][len(DATA_LINE_HEADER):])
  parsedLine['time'] = int(temp[1])

  for i in range(len(parsedLine['data'])):
    parsedLine[f'p{i}'] = parsedLine['data'][i]

  del parsedLine['data']
  del parsedLine['ID']
  return parsedLine

# can pass in list of thermo txt files, everything referenced to first thermo
def combined(_vive, _thermos):
  # if thermo is single string make in to list of len 1
  if type(_thermos) is str:
    _thermos = [_thermos]

  df_vive = vive(_vive)
  vive_data = process.vive(df_vive)
  vive_time = df_vive['time'].tolist()

  df_thermos=[]
  for _thermo in _thermos:
    df_thermo = thermo(_thermo)
    df_thermos.append(df_thermo)
    # thermo_time = df_thermo['time'].tolist()
    # thermistor = df_thermo['thermistor'].tolist()
  
  # remove last item in case trying to interp past the end
  timestamps = df_thermos[0]['time'].tolist()[:-1]
  
  vive_data_out = interpBetweenTimestamps(timestamps, vive_time, vive_data)
  # thermos_out = [process.thermoToDF(process.thermoFromDF(df_thermos[0]), timestamps)]
  thermos_out=[]

  for i in range(len(df_thermos)):
    print('Interping thermo')
    thermo_array=process.thermoFromDF(df_thermos[i])
    interped=interpBetweenTimestamps(timestamps, df_thermos[i]['time'].tolist(), thermo_array)
    thermos_out.append(process.thermoToDF(interped))

  return {
    'timestamps': [int(np.int64(i.to_datetime64())) for i in timestamps],
    'vive': vive_data_out,
    'thermo': thermos_out
  }

def interpBetweenTimestamps(_timestamps, _vivetime, _vivedata):
  out=[]
  j=0
  for t in _timestamps:
    while(_vivetime[j+1]<t):
      j+=1

    #vive time
    vt1=_vivetime[j]
    vt2=_vivetime[j+1]
    
    #time diff
    td1=(t-vt1).total_seconds()
    td2=(vt2-t).total_seconds()

    lerp = td1/(td1+td2)

    # vive val
    vv1=_vivedata[j]
    vv2=_vivedata[j+1]

    # interped vive vals
    vv=[]

    # xyz vals
    for k in range(len(vv1)):
      vv.append(vv1[k]*(1-lerp)+vv2[k]*(lerp))
    
    out.append(vv)
  
  return out

def combinedToFile(_data, _filename):
  FILENAME = os.path.join(COMBINED_DATA_PATH, _filename+'.gz')
  print(f'Saving combined vive thermo data to {FILENAME}')

  out = {
    'timestamps': _data['timestamps'],
    'vive': _data['vive'],
  }

  thermo_list = _data['thermo']
  thermo_list_out = [df_i.to_dict('records') for df_i in thermo_list]
  out['thermo'] = thermo_list_out

  outstring=json.dumps(out)
  with gzip.open(FILENAME, 'wt') as f:
    f.write(outstring)

def combinedFromFile(_filename):
  FILENAME = os.path.join(COMBINED_DATA_PATH, _filename+'.gz')
  print(f'Loading combined vive thermo data from {FILENAME}')

  try:
    with gzip.open(FILENAME,'rt') as f:
      file_content=f.read()
      data = json.loads(file_content)
      
      thermo_list = data['thermo']
      thermo_list_out = [pd.DataFrame(df_i) for df_i in thermo_list]
      data['thermo'] = thermo_list_out

      return data
  except IOError:
    print(f'File {FILENAME} does not exist.')
    return None

