#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi, pow, sqrt, exp

M = 100.0
A = 1.0
H = 300.0
F10 = 70
Ap = 0
print(" TIME HEIGHT PERIOD MEAN MOTION DECAY")
print("(days) (km) (mins) (rev/day) (rev/day^2)")
Re = 6378000  # Earth radius [m]
Me = 5.98E+24  # Earth mass [kg]
G = 6.67E-11  # Universal constant of gravitation
T = 0  # time
dT = .1  # time increment are in days
D9 = dT * 3600 * 24  # put time increment into seconds
H1 = 10  # H1=print height increment
H2 = H  # H2=print height,
R = Re + H * 1000  # R is orbital radius in metres
P = 2 * pi * sqrt(R * R * R / Me / G)  # P is period in seconds
# now iterate satellite orbit with time
while H < 180:
    SH = (900 + 2.5 * (F10 - 70) + 1.5 * Ap) / (27 - .012 * (H - 200))
    DN = exp(-(H - 175) / SH)  # atmospheric density
    dP = 3 * pi * A / M * R * DN * D9
    dP = 6E-10 * dP  # decrement in orbital period
    if H <= H2:  # test for print
        Pm = P / 60  # period in minutes
        MM = 1440 / Pm  # mean motion
        nMM = 1440 / ((P - dP)) / 60  # print units
        Decay = dP / dT / P * MM  # rev/day/day
        #print("{} {} {} {} {}".format(T, H, Pm, MM, Decay))
        H2 = H2 - H1  # decrement print height

    P = P - dP
    T = T + dT  # compute new values
    R = pow((G * Me * P * P / 4 / pi / pi), .33333)  # new orbital radius
    H = (R - Re) / 1000  # new altitude (semimajor axis)

# now print estimated lifetime of satellite
print("Re-entry after {} days ( {} years)".format(T, T / 365))
