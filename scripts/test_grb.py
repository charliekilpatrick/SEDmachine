from sedmachine import sedmachine
import numpy as np

params = {'E':0.0100,
          'n':1.0e-6,
          'theta_obs':22.5*np.pi/180.}

s = sedmachine()
s.load_grb(parameters=params)

print(s.times)
print(s.Lnu_all)
print(s.Lnu_all.shape)
print(s.times.shape)
for i,t in enumerate(s.times):
    m0 = np.max(s.Lnu_all[i,:])
    m1 = np.min(s.Lnu_all[i,:])
    m2 = np.median(s.Lnu_all[i,:])
    print(t,m0,m1,m2)    
