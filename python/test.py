import time
import random
import json
import os
from . import loader

def run(_a1, _a2):
  time.sleep(10)
  size=_a1*_a2
  out=[]
  for i in range(size):
    out.append(random.random())
  
  FILENAME = os.path.join(loader.OUTPUTS_PATH, f'job_{_a1}_{_a2}.json')
  with open(FILENAME, 'w') as outfile:
    json.dump(out, outfile)
  
  return FILENAME
