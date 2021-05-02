#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


x = np.array(range(66))
load = np.load('Ap.npz')
y = load['Ap']
load.close()

x = x
y = y

f = interpolate.interp1d(x, y, kind='cubic')

xnew = np.arange(0, 65, 1 / 12)
ynew = f(xnew)

plt.plot(x, y, 'o', xnew, ynew, '-')
# plt.show()

print('f(30) = {}'.format(f(30))
#
# Ap_func = np.polyfit(x, y, 100)
# p = np.poly1d(Ap_func)

# Ap_func2 = np.polyfit(x, y, 100)
# p2 = np.poly1d(Ap_func2)

# plt.plot(x, y, '.', xp, p(xp), '--')
# plt.show()
