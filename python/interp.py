def interp2d(_buff, nPoints):
  # // pad with zeroes around the outside
  # // ie buf [x,x,x]    ==>    [0,0,0,0,0]
  # //        [x,x,x]           [0,x,x,x,0]
  # //        [x,x,x]           [0,x,x,x,0]
  # //                          [0,x,x,x,0]
  # //                          [0,0,0,0,0]
  # // 
  _buff2 = []
  for i in range(len(_buff)+2):
    _buff2.append([0])

  for i in range(len(_buff[0])):
    _buff2[0].append(0)
    _buff2[-1].append(0)

  for i in range(len(_buff)):
    for j in range(len(_buff[i])):
      _buff2[i+1].append(_buff[i][j])
  
  for i in range(len(_buff)+2):
    _buff2[i].append(0)

  _buff = _buff2

  out = []
  for i in range(len(_buff)):
    out.append([])
    if (i<len(_buff)-1):
      for j in range(nPoints):
        out.append([])
  
  for element in out:
    for i in range(len(_buff[0]) + (len(_buff[0]) - 1) * nPoints):
      element.append(0)

  for i in range(len(_buff)):
    for j in range(len(_buff[0])):
      out[i * (nPoints + 1)][j * (nPoints + 1)] = _buff[i][j]

  # // output filled with all known values
  # // now interpolate across

  for i in range(len(_buff)):
    for j in range(len(_buff[0])-1):
      for k in range(1, nPoints+1):
        tY = i * (nPoints + 1)
        tX = j * (nPoints + 1)
        
        # // if too close to edge use linear interpolation
        # // else use cubic (the cubic needs 4 points whereas lerp only needs 2)
        if (j==0) or (j==len(_buff[0])-2):
          out[tY][tX + k] = lerp(out[tY][tX], out[tY][tX + nPoints + 1], k / (nPoints + 1))
        else:
          out[tY][tX + k] = cubic(
            out[tY][tX - (nPoints + 1)],
            out[tY][tX],
            out[tY][tX + (nPoints + 1)],
            out[tY][tX + 2*(nPoints + 1)],
            k / (nPoints + 1)
            )

  # // now interpolate down
  for j in range(len(out[0])):
    for i in range(len(_buff)-1):
      for k in range(1,nPoints+1):
        tY = i * (nPoints + 1)
        tX = j

        if (i==0) or (i==len(_buff)-2):
          out[tY + k][tX] = lerp(out[tY][tX], out[tY + nPoints + 1][tX], k / (nPoints + 1))
        else:
          out[tY + k][tX] = cubic(
            out[tY - (nPoints + 1)][tX],
            out[tY][tX],
            out[tY + (nPoints + 1)][tX],
            out[tY + 2*(nPoints + 1)][tX],
            k / (nPoints + 1)
            )

  return out

def lerp(_v1, _v2, _mu):
  return _v1*(1-_mu) + _v2*_mu

def cubic(_v0, _v1, _v2, _v3, _mu):
  mu2 = _mu*_mu
  a0 = _v3 - _v2 - _v0 + _v1
  a1 = _v0 - _v1 - a0
  a2 = _v2 - _v0
  a3 = _v1

  return a0*_mu*mu2 + a1*mu2 + a2*_mu + a3

# // ripped off the lerp and cubic from here
# // http://paulbourke.net/miscellaneous/interpolation//
