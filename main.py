import numpy as np
from plt import *

import sys

np.random.seed(0)
q=np.random.normal(0,0.3,(10000,2))


from raster import raster

from rsurf import rsurf,plot

from pyod.models.iforest import IForest


mn,mx=-1.5,1.5

query=raster(2,mn,mx,100)

clf=IForest(n_estimators=100,random_state=0)
clf.fit(q)

border=np.quantile(clf.decision_function(q),0.9)

vol,surf,m_s,m_v=rsurf(clf.decision_function,border,xmin=mn,xmax=mx,dim=2,steps=2)


plt.figure(figsize=(8,8))


plot(m_s,m_v,"maroon",True,"90% decision boundary")
plt.plot(q[:,0],q[:,1],'.',alpha=0.15,color="black",label="data")
plt.legend(frameon=True,framealpha=0.8)

plt.xlim(mn,mx)
plt.ylim(mn,mx)


plt.show()




