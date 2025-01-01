from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from model import Vector3
from kml import KML
from rinex4 import RINEXNavigationData, RINEXObservationData, SatelliteSystem
from utils import ecef_to_lla, ecef_to_enu, lla_to_ecef, Omegae_dot, c
x = np.zeros(4)

# x[0]=-2179073.8021354964
# x[1]=4388292.52747375
# x[2]=4069831.4224592056
# x[0]=-289831.858956647
# x[1]=-2756507.82903336
# x[2]=5725124.85444711



x[0]=-285338.169855
x[1]=466768.45343499
x[2]= 326851.29643225
#print(x)
estimated_lla = ecef_to_lla(Vector3(*x[:3]))
print(estimated_lla)

'''
def ecef_to_lla(ecef: Vector3) -> Vector3:
    """
    ecef_to_lla Convert ECEF coordinates to latitude, longitude, altitude.

    Parameters:
    ecef : Vector3
        ECEF coordinates [x, y, z].

    Returns:
    lla : Vector3
        Latitude, longitude in degrees and altitude in meters [lat, lon, alt].
    """
    eps = 1e-6
    lla = Vector3.zero()

    if ecef[0] >= 0:
        lla[1] = np.arctan(ecef[1] / ecef[0])
    else:
        if ecef[1] >= 0:
            lla[1] = np.pi + np.arctan(ecef[1] / ecef[0])
        else:
            lla[1] = -np.pi + np.arctan(ecef[1] / ecef[0])
    lla[0] = np.arctan(
        ecef[2] / np.sqrt(ecef[0] ** 2 + ecef[1] ** 2) * (1 + e2 / (1 - e2))
    )
    tmp = lla[0] + 10 * eps

    h = 0
    print(ecef)
    lla[0] =0
    while abs(lla[0] - tmp) > eps:
        tmp = lla[0]
        # N = a / np.sqrt(1 - e2 * np.sin(lla[0]) ** 2)
        # lla[0] = np.arctan(
        #     ecef[2]
        #     / np.sqrt(ecef[0] ** 2 + ecef[1] ** 2)
        #     * (1 + e2 * N * np.sin(lla[0]) / ecef[2])
        # )
        
        N = a / np.sqrt(1 - e2 * (np.sin(lla[0]) ** 2))
        h = np.sqrt(ecef[0] ** 2 + ecef[1] ** 2)/ np.cos(lla[0]) - N
        
        lla[0] = np.arctan(
            ecef[2]
            / np.sqrt(ecef[0] ** 2 + ecef[1] ** 2)
            / (1 - e2 * N /(N+h) )
        )
        print("h=",h,"phi=",lla[0],"erro=",abs(lla[0] - tmp))

    # lla[2] = (
    #     np.sqrt(ecef[0] ** 2 + ecef[1] ** 2) * np.cos(lla[0])
    #     + ecef[2] * np.sin(lla[0])
    #     - a * np.sqrt(1 - e2 * np.sin(lla[0]) ** 2)
    # )
    lla[2] =h
    
    lla[0] = np.degrees(lla[0])
    lla[1] = np.degrees(lla[1])

    lla = Vector3.from_list(lla)
    return lla
'''