# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:58:48 2016

@author: 111
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,5,0.1)
y = np.sin(x)
a,subp = plt.subplots(2)
subp[0].plot(x,y)
subp[1].plot(x,y)
plt.show()

#lt.plot(x,y)