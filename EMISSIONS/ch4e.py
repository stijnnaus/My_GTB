from pyhdf.SD import *
from pylab import *
from numpy import *
eall = []
for year in range(1988,2009):
   nd = [31,28,31,30,31,30,31,31,30,31,30,31]
   for month in range(1,13):
      nsec = nd[month-1]*24*3600
      ffile = 'flux_CH4_CTL_%4i%2.2i.hdf'%(year,month)
      hdf = SD(ffile)
      e = hdf.select('EMIS').get().sum()
      hdf.end()
      print e*nsec*1e-9, 'Tg/month', year,month
      eall.append(e*nsec*1e-9)
nyear = 0
xx = zeros((12))
for year in range(1988,2009):
   xx += eall[nyear*12:(nyear+1)*12]
   nyear +=1
xx /= nyear
print xx


