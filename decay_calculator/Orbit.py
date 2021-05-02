# -*- coding: utf-8 -*-

from math import sqrt
from Constants import MU


class Orbit:
    def __init__(self, name, n, i, raan, e, w, m_a, n_o, a, area, mass):
        self.name = str(name)  # satellite name
        self.satnum = n  # satellite number
        self.inclination = i  # inclination
        self.raan = raan  # Right Ascension of the Ascending Node[Degrees]
        self.eccentricity = e  # Eccentricity
        self.w = w  # Argument of Perigee[Degrees]
        self.mean_anomaly = m_a  # Mean Anomaly[Degrees]
        self.mean_motion = n_o  # Mean Motion[Revs per day] a # semi
        self.a = a  # Semi major axis[m]
        self.apoapsis = max(self.a * (1 - self.eccentricity),
                            self.a * (1 + self.eccentricity))
        self.periapsis = min(self.a * (1 - self.eccentricity),
                             self.a * (1 + self.eccentricity))
        # Velocity at apoapsis[m s - 1]
        self.v_a = sqrt(MU * (2 / self.apoapsis - 1 / self.a))
        # Velocity at periapsis[m s - 1]
        self.v_p = sqrt(MU * (2 / self.periapsis - 1 / self.a))
        self.area = area
        self.mass = mass
        self.P = None
