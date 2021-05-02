# -*- coding: utf-8 -*-
from math import pi, sqrt, pow, sin, cos
from os import listdir

import numpy as np
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

from Constants import RE, MU
from Orbit import Orbit

# LOCAL CONSTANTS
Re = 6378.137  # expressed in km
Mu = 398600.4418
Cd = 1
F107 = 70  # solar radio flux at 10.7 cm. From 65 to 300. Good value is 70.
# 300 is used during solar storms
Ap = 0
PI2 = 2 * pi
CMP = pow(Mu / (4 * pi * pi), 1 / 3)
CMP_m = pow(MU / (4 * pi * pi), 1 / 3)

MIN_ORBIT_TIME = 10 * 365 * 24 * 60 * 60
MAX_ORBIT_TIME = 200 * 365 * 24 * 60 * 60
# limit of the timestamp. A debries removed each week.
WEEK_LIM = 7 * 24 * 60 * 60


# get the areas from the file.
def get_areas(filename):
    return np.loadtxt(filename, dtype=np.float, delimiter=",")


def get_masses(areas, total_mass):
    m = np.zeros(len(areas))
    total_area = sum(areas)
    for i in range(len(areas)):
        m[i] = total_mass * areas[i] / total_area
    return m


# get orbits from tle file.
def get_orbits(filename, out_dir, areas, masses):
    f = open(filename)
    i = 0
    orbits = []

    line0 = f.readline()

    already_computed = []
    files = listdir(out_dir)
    for file in files:
        already_computed.append(int(file.split('.')[0]))

    while line0 != '' and i != len(areas):

        # name
        name = line0.strip('\n')

        line1 = f.readline()
        line2 = f.readline()
        satellite = twoline2rv(line1, line2, wgs84)
        satellite.propagate(2017, 12, 1, 12, 00, 00)

        satellite_number = satellite.satnum
        inclination = satellite.inclo
        raan = satellite.nodeo
        eccentricity = satellite.ecco
        w = satellite.argpo
        mean_anomaly = satellite.mo
        mean_motion = satellite.no
        o = Orbit(name, satellite_number, inclination, raan, eccentricity, w, mean_anomaly, mean_motion,
                  satellite.a * RE, areas[i], masses[i])

        # orbital decay
        print("Computing satellite {}".format(o.satnum))

        if not o.satnum in already_computed:
            P, t = compute_orbital_decay(
                orbits, o.a / 1000, o.eccentricity, o.area, o.mass)
            is_loaded = False
        else:
            print("  P loaded from {}.npz".format(o.satnum))
            loaded = np.load(out_dir + str(o.satnum) + ".npz")
            P = loaded["P"]
            t = loaded["t"]
            loaded.close()
            is_loaded = True

        if t[-1] < MIN_ORBIT_TIME:
            print(" skipping satellite: {}".format(o.satnum))
            del P, t, o
        else:
            if not is_loaded:
                np.savez_compressed(
                    out_dir + "{}.npz".format(o.satnum), P=P, t=t)
                print("  saved file: {}.npz".format(o.satnum))
            else:
                print("  already saved file: {}.npz".format(o.satnum))
            i = i + 1
            o.P = P
            orbits.append(o)

        line0 = f.readline()

    f.close()
    return np.array(orbits)


# Compute the circular height(km)
def h(P):
    return pow(P, 2 / 3) * CMP  # Compute the circular height [km]


def he(a, e):
    # Compute the effective height [km]
    return a * (1 - e) + 900 * pow(e, 0.6) - Re


# Equivilent height in km
def H(h):
    return (900 + 2.5 * (F107 - 70) + 1.5 * Ap) / (27 - 0.012 * (h - 200))


def rho(h):
    return 6e-10 * np.exp(-(h - 175) / H(h))  # Density(kg m ^ -3)


def dP(hh, Ae, dt, m0):
    return (-3 * pi * (hh + Re) * 1000 * rho(hh) * Ae * dt) / m0


def compute_orbital_decay(orbits, a, e, A, m0):
    # a, e, A, m0 are scalar.
    P = [2 * pi * sqrt(pow(a, 3) / Mu)]
    Ae = A * Cd
    dt = 0.5 * P[0]
    t = [0]
    p_min = 2 * pi * sqrt(pow(Re + 180, 3) / Mu)

    # Iterate satellite orbit with time:
    hh = 181
    while P[-1] > p_min and hh >= 181:
        hh = he(h(P[-1]), e)
        P.append(P[-1] + dP(hh, Ae, dt, m0))
        t.append(t[-1] + dt)

        if t[-1] >= MAX_ORBIT_TIME:
            break
    return np.array(P), np.array(t)


def shrink(o):
    n = len(o.P)
    indexes = [0]
    # keep only t indexes.
    i = 0
    k = i + 1
    # tmp = t[i]
    tmp = 0

    while k < n:
        tmp = tmp + o.P[k]
        if tmp >= WEEK_LIM:
            i = k
            tmp = 0
            indexes.append(i)
        k = k + 1

    return o.P[indexes]


def update_orbits(orbits, t):
    for o in orbits:
        o.a = pow(o.P[t], 2 / 3) * CMP_m
        o.periapsis = o.a * (1 - o.eccentricity)
        o.apoapsis = o.a * (1 + o.eccentricity)
        # Velocity at apoapsis[m s - 1]
        o.v_a = sqrt(MU * (2 / o.apoapsis - 1 / o.a))
        # Velocity at periapsis[m s - 1]
        o.v_p = sqrt(MU * (2 / o.periapsis - 1 / o.a))
    return orbits


def manouver(o_s, o_f):
    # First manouver CASE 1
    gamma_s = o_s.raan + o_s.w
    gamma_f = o_f.raan + o_f.w
    diff = abs(gamma_f - gamma_s)

    if diff >= pi / 2 and diff <= 3 * pi / 2:  # case 2
        # first man
        a = (o_s.periapsis + o_f.apoapsis) / 2
        v_ta = sqrt(MU * (2 / o_f.apoapsis - 1 / a))
        delta_1 = abs(v_ta - o_s.v_p)

        # second man: circularization
        delta_2 = abs(v_ta - o_f.v_a)

        # inclination change:
        cos_delta_i = cos(o_s.inclination) * cos(o_f.inclination) + sin(o_s.inclination) * sin(o_f.inclination) * cos(o_s.raan) * cos(o_f.raan) + sin(
            o_s.inclination) * sin(o_f.inclination) * sin(o_s.raan) * sin(o_f.raan)
        delta_i = sqrt(pow(o_f.v_a, 2) + pow(o_f.v_p, 2) -
                       2 * o_f.v_a * o_f.v_p * cos_delta_i)

    else:  # case 1
        # first man
        a = (o_s.periapsis + o_f.periapsis) / 2
        v_ta = sqrt(MU * (2 / o_f.periapsis - 1 / a))
        delta_1 = abs(v_ta - o_s.v_a)

        # second man: circularization
        delta_2 = abs(v_ta - o_f.v_p)

        # inclination change:
        cos_delta_i = cos(o_s.inclination) * cos(o_f.inclination) + sin(o_s.inclination) * sin(o_f.inclination) * cos(o_s.raan) * cos(o_f.raan) + sin(
            o_s.inclination) * sin(o_f.inclination) * sin(o_s.raan) * sin(o_f.raan)
        delta_i = sqrt(pow(o_f.v_a, 2) + pow(o_f.v_p, 2) -
                       2 * o_f.v_a * o_f.v_p * cos_delta_i)

    return delta_1 + delta_2 + delta_i


def compute_manouvers(orbits, timestamp):
    n = len(orbits)
    m = np.zeros((n, n))
    orbits = update_orbits(orbits, timestamp)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = manouver(orbits[i], orbits[j])
            m[j, i] = m[i, j]
    return m


# compute the manouvers matricies for each satellite with sufficient high orbit.
def compute_manouvers_matricies(savefilename, orbits):
    print("Computing manouver matrix")
    min_len = len(orbits[0].P)
    # Loop for each satellite filter the timestamps once a week and compute the manouvers.
    for o in orbits:
        o.P = shrink(o)

        l = len(o.P)
        if min_len > l:
            min_len = l

    # For each significant period, compute it's matrix.
    m = []
    for i in range(min_len):
        m.append(compute_manouvers(orbits, i))
        print("  computed {}-th matrix.".format(i))
    m = np.array(m)
    shape = np.shape(m)
    np.savez_compressed(savefilename, m=m)
    del m
    print("Final shape = {}\n  saved: {}".format(shape, savefilename))


if __name__ == '__main__':
    filename_areas = "./debris_cloud/fengyun1C_areas.txt"
    filename_tles = "./debris_cloud/fengyun1C_tle.txt"
    filename_weight = "./debris_cloud/fengyun1C_weight.txt"
    out_dir = "./fengyun/"
    savefilename = "fengyun1C_weight_matrix.npz"

    # get the areas.
    areas = get_areas(filename_areas)
    total_mass = float(np.loadtxt(filename_weight))
    masses = get_masses(areas, total_mass)

    orbits = get_orbits(filename_tles, out_dir, areas, masses)

    print("total number of satellites = {}".format(len(orbits)))
    compute_manouvers_matricies(savefilename, orbits)
# tua
