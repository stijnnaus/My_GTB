from numpy import *
f = open('mcculloch.dat','r')
y = []
e = []
b = []
for line in f.readlines():
   y.append(int(line.split()[0]))
   e.append(float(line.split()[1]))
   b.append(float(line.split()[2]))
f.close()
# get production
p = [e[0]]
ny = len(y)
for i in range(1,ny):
   p.append(e[i] + b[i] - b[i-1])
# Medium and Rapid Sales (Midgley and McCullloch 1995): from 1970on:
rapid = array(p[0:19]+[135,155,207,259,294,287,387,460,504,486,526,522,489,519,577,568,586,605,657,673,705,585,566,275])
medium = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,20,20,20,20,20,20,20,20,26,26,25,22,24,20,23,22,22,21,20,20,30,8])
slow = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,18])
# emodel:
emodel = [0.75*rapid[0] + 0.25*medium[0]]
for i in range(1,43):
   if i == 1: 
      emodel.append(0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1])
   else:
      emodel.append(0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1] + 0.75*slow[i-2])
# Medium and Rapid Sales (Midgley and McCullloch 1995): from 1970on:
# Prinss alternative: put 5% in stock with release time 10 years:
rapid = array(p[0:19]+[135,155,207,259,294,287,387,460,504,486,526,522,489,519,577,568,586,605,657,673,705,585,566,275])
medium = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,20,20,20,20,20,20,20,20,26,26,25,22,24,20,23,22,22,21,20,20,30,8])
slow = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,18])

stock = (rapid + medium + slow)*0.05
rapid = rapid - stock 

# eprinn:
eprinn = [0.75*rapid[0] + 0.25*medium[0]]
for i in range(1,43):
   if i == 1: 
      emission = 0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1]
   else:
      emission = 0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1] + 0.75*slow[i-2]
   if i < 10:
      for j in range(i-1,-1,-1):
         emission = emission + 0.1*stock[j]
   else:
      for j in range(i-1,i-11,-1):
         emission = emission + 0.1*stock[j]
   eprinn.append(emission)
     
#extend this model:
rapid = p[0:19]+[135,155,207,259,294,287,387,460,504,486,526,522,489,519,577,568,586,605,657,673,705,585,566,275]
medium =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,20,20,20,20,20,20,20,20,26,26,25,22,24,20,23,22,22,21,20,20,30,8]
slow = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,18]
p.append(12.6)  # 2001
p.append(12.9)  # 2002
p.append(12.9)  # 2003
p.append(12.9)  # 2004
for year in range(1,5):
   p.append(12.9*exp(-year/5.0))
for i in range(43,58):
   rapid.append(0.9*p[i])
   medium.append(0.05*p[i])
   slow.append(0.05*p[i])
rapid = array(rapid)
medium = array(medium)
slow = array(slow)
# emodel:
emodel = [0.75*rapid[0] + 0.25*medium[0]]
for i in range(1,58):
   if i == 1: 
      emodel.append(0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1])
   else:
      emodel.append(0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1] + 0.75*slow[i-2])

stock = (rapid + medium + slow)*0.025
rapid = rapid - stock 
print 'sum production', sum(p)
print 'sum production', sum(rapid+medium+slow+stock)

eprinn = [0.75*rapid[0] + 0.25*medium[0]]
for i in range(1,58):
   if i == 1: 
      emission = 0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1]
   else:
      emission = 0.75*rapid[i] + 0.25*rapid[i-1] + 0.25*medium[i] + 0.75*medium[i-1] + 0.25*slow[i-1] + 0.75*slow[i-2]
   if i < 10:
      for j in range(i-1,-1,-1):
         emission = emission + 0.1*stock[j]
   else:
      for j in range(i-1,i-11,-1):
         emission = emission + 0.1*stock[j]
   eprinn.append(emission)
f = open('emissions.dat','w')
i = 0
for year in range(1951,2009):
   f.write('%4i  %6.1f %6.1f  %6.1f  %6.1f  %6.1f \n'%(year,rapid[i],medium[i],slow[i],stock[i],eprinn[i]))
   i += 1
f.close()
