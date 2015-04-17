import sys
import os
import numpy as np
from pylab import *
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import math
from scipy.optimize import curve_fit
import random
from array import *
import numpy
from random import randrange, uniform

#Code to compute MCMC for linear regression.
#Aisha Mahmoud-Perez

#Read data file.
data = np.genfromtxt('probset5.dat')
x = data[:,0]
y = data[:,1]
sigmay = data[0:,2] 

#apply a linear fit
def fit_func(x,a,b):
	return a*x +b


params = curve_fit(fit_func, x, y)
[a,b]=params[0]
print('Slope and intercept of given data: ',a,b)

#calculate confidence interval, assuming normal distribution. 
def mean_confidence_interval(data, confidence):
	d = 1.0*np.array(data)
	n =  len(d)
	m, se = np.mean(d), scipy.stats.sem(d)
	h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
	return m, m-h, m+h

#Define variables and arrays.
a = 1.5
b = 30.0
varx = 0.03
vary = 2.0
length=5000
n =  len(x)
ite=[]
at_tot = []
bt_tot = []

for i in range(length):
	gauss = random.gauss(0,1)
	gauss2 = random.gauss(0,1)
	at = a + varx*gauss
	bt = b + vary*gauss2
	sum_chi_a = 0
	sum_chi_at = 0
	alpha = 0
	for j in range(n):
		chi_a = ((y[j]-(a*x[j]+b))/sigmay[j])**2
		chi_at = ((y[j]-(at*x[j]+bt))/sigmay[j])**2
		sum_chi_a += chi_a
		sum_chi_at += chi_at
	alpha = min(1, exp((sum_chi_a-sum_chi_at)/2.0))
	u = random.uniform(0, 1)
	if u <= alpha:
		ite.append(i)
		at_tot.append(at)
		bt_tot.append(bt)
		a = at
		b = bt

arr = mean_confidence_interval(at_tot, 0.95)
brr = mean_confidence_interval(bt_tot, 0.95)
print ('chain length = ', len(at_tot))
print ('ratio = ', len(at_tot)/5000.)
print(arr, brr)
plt.figure(1)
plt.plot(ite,at_tot,color='black', linestyle=' ', marker='.')
plt.xlabel('Chain Element')
#plt.xlim(0,1000)
plt.ylabel('a(slope)')

plt.figure(2)
plt.plot(ite,bt_tot,color='black', linestyle=' ', marker='.')
plt.xlabel('Chain Element')
#plt.xlim(0,1000)
plt.ylabel('b(intercept)')

plt.figure(3)
plt.plot(bt_tot,at_tot,color='black', linestyle=' ', marker='.')
plt.xlabel('b(intercept)')
plt.ylabel('a(slope)')

plt.show()


