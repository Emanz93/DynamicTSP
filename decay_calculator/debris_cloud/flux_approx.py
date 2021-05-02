#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


load = np.load('flux.npz')
y = load['f10']
load.close()
x = np.array(range(len(y)))

x = x
y = y

f = interpolate.interp1d(x, y, kind='cubic')

xnew = np.arange(0, len(y) - 1, 1 / 12)
ynew = f(xnew)

plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()

print('f(30) = {}'.format(f(30)))

# dasf
