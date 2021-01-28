import time
from .python import mlp_tuning2
# from .python2 import test

def decode(_job):
  _job=_job.split(' ')
  return int(_job[0]),int(_job[1]),_job[2],_job[3]

def run(_job):
  a1,a2,a3,a4=decode(_job)
  return mlp_tuning2.run(a1,a2,a3,a4)
  
# def decode(_job):
#   _job=_job.split(' ')
#   return int(_job[0]),int(_job[1])

# def run(_job):
#   a1,a2=decode(_job)
#   return test.run(a1,a2)
  
