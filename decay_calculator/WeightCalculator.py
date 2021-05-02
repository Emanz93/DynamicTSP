# -*- coding: utf-8 -*-
import threading
from math import pi, sqrt, pow, sin, cos, exp
from os import listdir

import numpy as np
from scipy import interpolate
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

from Constants import RE, MU, WEEK, YEAR, MONTH
from Orbit import Orbit

# LOCAL CONSTANTS

PI2 = 2 * pi
PI24 = 4 * pi * pi

# CMP = pow(Mu / (4 * pi * pi), 1 / 3)
CMP_m = pow(MU / (4 * pi * pi), 1 / 3)


class OrbitalDecayer():
    # P, t = compute_orbital_decay(o.a / 1000, o.eccentricity, o.area, o.mass, ap)
    def __init__(self, a, e, A, m0, f10_data, aps_data):
        self.a = a  # Semi Major Axis
        self.e = e  # Eccentricity
        self.periapsis_height = a * (1 - e)
        self.A = A  # Cross section area of the debris
        self.m0 = m0  # mass of the debris
        # Geomagnetic Index. Function of the time [years] ellapsed.
        self.Ap = self.init_ap(aps_data)
        self.f10 = self.init_f10(f10_data)  # solar flux F107
        self.AP_MAX = aps_data[-1]
        self.F10_MAX = f10_data[-1]
        self.P = None  # Periods' array.
        self.t = None  # Timestamps.

    def init_ap(self, aps_data):
        """Init the Ap function. Interpolate the 66 values of the Ap given values.
            H. S. Ahluwalia. "The predicted size of cycle 23 based on the inferred three-cycle
            quasi-periodicity of the planetary index Ap". JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 103,
            NO. A6, PAGES 12,103-12,109, JUNE 1, 1998.
        """
        x = np.array(range(len(aps_data)))
        return interpolate.interp1d(x, aps_data, kind='cubic')

    def init_f10(self, f10_data):
        """
        Init the f10 solar radio flux function. Monthly averages of the adjusted (scaled for
        an Earth-Sun distance of 1 AU) F 10.7 values since 1947.
        Source: "The 10.7 cm solar radio flux (F 10.7 )." K. F. Tapping.
        SPACE WEATHER, VOL. 11, 394–406, doi:10.1002/swe.20064, 2013

        :param f10_data:
        :return:
        """
        x = np.array(range(len(f10_data)))
        return interpolate.interp1d(x, f10_data, kind='cubic')

    def Cd(self, h):
        if h >= 300:
            return 2.4
        elif h >= 250:
            return 2.3
        elif h >= 200:
            return 2.2
        else:
            return 2.1

    def compute_orbital_decay(self):
        """Compute orbital decay starting from the periapsis height. The step is one week."""
        h = (self.periapsis_height - RE) / 1000

        r = RE + h * 1000
        self.P = [2 * pi * sqrt(pow(r, 3) / MU)]  # initial Period.
        self.t = [0]

        dt = WEEK  # a week in seconds

        # Iterate satellite orbit with time:

        while (h >= 181):
            f10_param = self.f10((self.t[-1] / MONTH) % (self.F10_MAX))
            ap_param = self.Ap((self.t[-1] / YEAR) % (self.AP_MAX))
            # print("f10={}".format(f10_param))
            # print("ap={}".format(ap_param))
            # cd_param = atan(x - 2.22) + h

            # SH = (900 + 2.5 * (F10 - 70) + 1.5 * Ap) / (27 - .012 * (H - 200))
            sh = (900 + 2.5 * (f10_param - 70) + 1.5 *
                  ap_param) / (27 - 0.012 * (h - 200))
            # dP = 3 * pi * A / M * R * D9 * exp(-(H - 175) / SH)
            dP = 3 * pi * self.A * \
                self.Cd(h) / self.m0 * r * dt * exp(-(h - 175) / sh)
            # dP = 6E-10 * dP  # decrement in orbital period
            dP = dP * 6e-10
            # P = P - dP
            self.P.append(self.P[-1] - dP)
            # T = T + dT  # compute new values
            self.t.append(self.t[-1] + dt)
            # R = pow((MU * pow(P, 2) / 4 / pow(pi, 2)), .3333)  # new orbital radius
            p_2 = self.P[-1] * self.P[-1]
            r = pow(MU * (p_2 / PI24), 0.3333)
            # H = (R - RE) / 1000  # new altitude (semimajor axis)
            h = (r - RE) / 1000
        return np.array(self.P), np.array(self.t)


# get the areas from the file.
def get_areas(filename):
    return np.loadtxt(filename, dtype=np.float, delimiter=",")


def get_aps(filename_aps):
    loaded = np.load(filename_aps)
    aps = loaded['Ap']
    loaded.close()
    return aps


def get_flux(filename_flux):
    loaded = np.load(filename_flux)
    flux = loaded['f10']
    loaded.close()
    return flux


def get_masses(areas, total_mass):
    m = np.zeros(len(areas))
    total_area = sum(areas)
    for i in range(len(areas)):
        m[i] = total_mass * areas[i] / total_area
    return m


# get orbits from tle file.
def get_orbits(filename, areas, masses):
    f = open(filename)
    i = 0
    orbits = []

    line0 = f.readline()
    max_len = len(areas)

    while line0 != '' and i != max_len:
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
        orbits.append(o)
        line0 = f.readline()
        i = i + 1

    f.close()
    return np.array(orbits)


# Compute decay info. Called only from the daemon.
def t_worker(dir_of_npz, orbits, start, stop, aps_data, flux_data):
    i = start

    done = listdir(dir_of_npz)

    while i < stop:
        o = orbits[i]

        if not str(o.satnum) + '.npz' in done:
            print('{}: computing {} satellite'.format(
                threading.currentThread().getName(), o.satnum))
            od = OrbitalDecayer(o.a, o.eccentricity, o.area,
                                o.mass, aps_data, flux_data)
            P, t = od.compute_orbital_decay()
            np.savez_compressed(
                dir_of_npz + "{}.npz".format(o.satnum), P=P, t=t)
            print("{} saved file: {}.npz. Len={}".format(
                threading.currentThread().getName(), o.satnum, len(P)))
            del P
            del t
        del o
        i = i + 1
    print("Computed all orbital decay...")


def update_orbits(orbits, t):
    for o in orbits:
        try:
            o.a = pow(o.P[t], 2 / 3) * CMP_m
        except TypeError:
            print("o={}".format(o))
            print("o.P={}".format(o.P))
            print("t={}".format(t))
            # print("CMP_m={}".format(CMP_m))
            # print("o={}".format())
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
        cos_delta_i = cos(o_s.inclination) * cos(o_f.inclination) + sin(o_s.inclination) * sin(o_f.inclination) * cos(
            o_s.raan) * cos(o_f.raan) + sin(
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
        cos_delta_i = cos(o_s.inclination) * cos(o_f.inclination) + sin(o_s.inclination) * sin(o_f.inclination) * cos(
            o_s.raan) * cos(o_f.raan) + sin(
            o_s.inclination) * sin(o_f.inclination) * sin(o_s.raan) * sin(o_f.raan)
        delta_i = sqrt(pow(o_f.v_a, 2) + pow(o_f.v_p, 2) -
                       2 * o_f.v_a * o_f.v_p * cos_delta_i)

    return delta_1 + delta_2 + delta_i


def compute_manouvers(orbits, t):
    n = len(orbits)
    m = np.zeros((n, n))
    orbits = update_orbits(orbits, t)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = manouver(orbits[i], orbits[j])
            m[j, i] = m[i, j]
    return m


def compute_manouvers_matricies(dir_of_npz, filename_save, tmp_dir, orbits):
    """compute the manouvers matricies for each satellite with sufficient high orbit. Hence remove those satellite that
    naturally decay themself"""
    MIN_ORBIT_TIME = len(orbits) * WEEK
    print("Update orbits...")
    # Loop for each satellite filter the timestamps once a week and compute the manouvers.
    i = 0
    while i < len(orbits):
        o = orbits[i]
        loaded = np.load(dir_of_npz + str(o.satnum) + ".npz")
        t = loaded["t"]
        P = loaded["P"]
        loaded.close()

        if t[-1] < MIN_ORBIT_TIME:
            print("Warning. Satellite {} not pass MIN_ORBIT_TIME={} weeks.".format(
                o.satnum, MIN_ORBIT_TIME / WEEK))
            orbits = np.delete(orbits, i)
            del P, t
        else:
            orbits[i].P = P[:len(orbits)]
            i = i + 1

    # For each significant period, compute it's matrix.
    save_in_npz(orbits, filename_save, tmp_dir)


def save_in_npz(orbits, filename_save, tmp_dir):
    """Save the final matricies of weights into a npz file."""
    if tmp_dir == "" or tmp_dir is None:
        m = []
        for i in range(len(orbits)):
            m.append(compute_manouvers(orbits, i))
            print("computed manouvers for t={}".format(i))
            np.savez_compressed(filename_save, w=np.array(m))
    else:
        # The debris are too much. I need to use local storage.
        files = listdir(tmp_dir)
        for i in range(len(orbits)):
            if not str(i) + ".txt" in files:
                np.savetxt(tmp_dir + '{}.txt'.format(i),
                           compute_manouvers(orbits, i), delimiter=',')
                print("computed manouvers for t={}".format(i))
        print("computed all matricies.")
        m = []
        # the file is too big. Genration of multiple npz files.
        if len(files) > 500:
            step = 100
            times = 1
            files = sorted(files)
            for i in range(len(files)):
                file = str(i) + ".txt"
                print("appending {} ...".format(file))
                m.append(np.loadtxt(tmp_dir + file, delimiter=','))
                if i == step * times - 1:
                    print('  saving...')
                    newfilename = filename_save.split(
                        '.n')[0] + "_" + str(times) + ".npz"
                    np.savez_compressed(newfilename, w=np.array(m))
                    print("  saved {} ...".format(newfilename))
                    m = []
                    times = times + 1
        else:
            for file in files:
                print("appending {} ...".format(file))
                m.append(np.loadtxt(tmp_dir + file, delimiter=','))
            np.savez_compressed(filename_save, w=np.array(m))

    print("Weight file saved in {}".format(filename_save))


if __name__ == '__main__':
    # filename_areas = "./debris_cloud/iridium33_areas.txt"
    # filename_tles = "./debris_cloud/iridium33_tle.txt"
    # filename_weight = "./debris_cloud/iridium33_weight.txt"
    # filename_aps = "./debris_cloud/Ap.npz"
    # filename_flux = "./debris_cloud/flux.npz"
    # dir_of_npz = './raw_data_npz/iridium/'
    # filename_save = "./fengyun_matricies_weights.npz"
    # tmp_dir = ""

    # filename_areas = "./debris_cloud/cosmos2251_areas.txt"
    # filename_tles = "./debris_cloud/cosmos2251_tle.txt"
    # filename_weight = "./debris_cloud/cosmos2251_weight.txt"
    # filename_aps = "./debris_cloud/Ap.npz"
    # filename_flux = "./debris_cloud/flux.npz"
    # dir_of_npz = './raw_data_npz/cosmos/'
    # filename_save = "./cosmos_matricies_weights.npz"
    # tmp_dir = ""

    filename_areas = "./debris_cloud/fengyun1C_areas.txt"
    filename_tles = "./debris_cloud/fengyun1C_tle.txt"
    filename_weight = "./debris_cloud/fengyun1C_weight.txt"
    filename_aps = "./debris_cloud/Ap.npz"
    filename_flux = "./debris_cloud/flux.npz"
    dir_of_npz = "./raw_data_npz/fengyun/"
    filename_save = "./fengyun_matricies_weights.npz"
    tmp_dir = "./fengyun_txt/"

    # get the areas.
    aps = get_aps(filename_aps)
    areas = get_areas(filename_areas)
    total_mass = float(np.loadtxt(filename_weight))
    masses = get_masses(areas, total_mass)
    aps_data = get_aps(filename_aps)
    flux_data = get_flux(filename_flux)

    orbits = get_orbits(filename_tles, areas, masses)
    print("Loaded all basic informations.")

    n = len(areas)

    t_worker(dir_of_npz, orbits, 0, n, aps_data, flux_data)

    compute_manouvers_matricies(dir_of_npz, filename_save, tmp_dir, orbits)
