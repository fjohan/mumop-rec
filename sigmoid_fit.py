# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# good discussion here:  http://stackoverflow.com/questions/4308168/sigmoidal-regression-with-scipy-numpy-python-etc
# curve_fit() example from here: http://permalink.gmane.org/gmane.comp.python.scientific.user/26238
# other sigmoid functions here: http://en.wikipedia.org/wiki/Sigmoid_function

import numpy as np
import pylab
from scipy.optimize import curve_fit

def sigmoid_rf(x, d, c1, a1, c2, a2, b):
     #y = d - (c1 / (1 + np.exp(-a1*(b-x)+g))) - (c2 / (1 + np.exp(-a2*(x-b)+g)))
     #y = (c1 / (1 + np.exp(-a1*(b-x)+g)))
     y = (c2 / (1 + np.exp(-a2*(x-b))))
     return y

def sigmoid2(x, a1, a2, b, c1, c2, d):
     y = d - (c1 / (1 + np.exp(-a1*(b-x)))) - (c2 / (1 + np.exp(-a2*(x-b))))
     return y
    
def sigmoid1(x, x0, a1, c1, d):
     y = c1 / (1 + np.exp(-a1*(x-x0)))
     return y

def sigmoid_sum(x, *args):
    d, b, g, c1, c2, k1, k2 = args
    #ret = d
    #ret -= c1 / (1 + np.exp(-k1*(b-x)+g))
    #ret -= c2 / (1 + np.exp(-k2*(x-b)+g))
    ret = d - (c1 / (1 + np.exp(-k1*(b-x)+g))) - (c2 / (1 + np.exp(-k2*(x-b)+g)))
    #ret = 1
    #ret += -1 / (1 + np.exp(-(-1.3)*(x-7.2)))
    #ret += c2 / (1 + np.exp(-k2*(x+18)))
    return ret

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

xdata1 = np.array([ 0.0,  1.0,  3.0,  4.3,  7.0,  8.0,  8.5, 10.0, 12.0])
xdata2 = np.array([23.0, 25.0, 27.0, 28.0, 30.7, 34.0, 35.0])
ydata1 = np.array([0.01, 0.15, 0.04, 0.11, 0.43,  0.7, 0.89, 0.95, 0.99])
ydata2 = np.array([0.99, 0.89,  0.7, 0.43, 0.11, 0.02, 0.01])
xdata3 = np.append(xdata1, xdata2)
ydata3 = np.append(ydata1, ydata2)

params = [1, 12, 6, 1, 1, 1.3, 1.3]

popt, pcov = curve_fit(sigmoid_sum, xdata3, ydata3, p0=params)
print popt

#x = np.linspace(-1, 15, 50)
#x = np.linspace(12, 28, 50)
x = np.linspace(-1, 50, 50)
y = sigmoid_sum(x, *popt)

pylab.plot(xdata3, ydata3, 'o', label='data')
pylab.plot(x,y, label='fit')
pylab.ylim(0, 2.05)
pylab.legend(loc='best')
pylab.show()