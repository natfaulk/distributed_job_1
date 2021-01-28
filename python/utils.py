import os
import os

import numpy as np
import math

def translate(point, trans):
  point2 = point + [1]
  trans_mat = np.array([[1, 0, 0, trans[0]],
                        [0, 1, 0, trans[1]],
                        [0, 0, 1, trans[2]],
                        [0, 0, 0, 1]])
  return trans_mat.dot(point2).tolist()[:3]

def rotate(point, angle, axis):
  rotMat = []
  if (axis == 'x'):
    rotMat = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])
  if (axis == 'y'):
    rotMat = np.array([[math.cos(angle), 0, math.sin(angle)],[0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
  if (axis == 'z'):
    rotMat = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

  return rotMat.dot(point).tolist()[:3]

def rmse(_errors):
  t=np.square(_errors)
  t=np.mean(t)
  return math.sqrt(t)

def mkdir_p(_path):
  os.makedirs(_path,exist_ok=True)



