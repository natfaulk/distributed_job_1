from .python import rf_test2

def decode(_job):
  _job=_job.split(' ')

  if (_job[1] == 'None'):
    _job[1] = None
  else:
    _job[1] = int(_job[1])

  if (_job[2] == 'None'):
    _job[2] = None
  else:
    _job[2] = int(_job[2])
  
  return int(_job[0]),_job[1],_job[2],int(_job[3]),int(_job[4])

def run(_job):
  a1,a2,a3,a4,a5=decode(_job)
  return rf_test2.run(a1,a2,a3,a4,a5)
  
# def decode(_job):
#   _job=_job.split(' ')
#   return int(_job[0]),int(_job[1])

# def run(_job):
#   a1,a2=decode(_job)
#   return test.run(a1,a2)
  
