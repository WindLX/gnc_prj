import os
import gzip
import shutil
from typing import List, Tuple

from gps_constants import GpsConstants
from gnss_thresholds import GnssThresholds
from day_of_year import day_of_year
from utc_to_gps import utc_to_gps
from read_rinex_nav import read_rinex_nav


def get_nasa_hourly_ephemeris(
    utcTime: List[int], dirName: str
) -> Tuple[List[dict], List[dict]]:
    allGpsEph = []
    allGloEph = []

    bOk, dirName = check_inputs(utcTime, dirName)
    if not bOk:
        return allGpsEph, allGloEph

    yearNumber4Digit = utcTime[0]
    yearNumber2Digit = utcTime[0] % 100
    dayNumber = day_of_year(utcTime)

    hourlyZFile = f"hour{dayNumber:03d}0.{yearNumber2Digit:02d}n.Z"
    ephFilename = hourlyZFile[:-2]  # Remove .Z extension
    fullEphFilename = os.path.join(dirName, ephFilename)

    # Check if ephemeris file already exists locally
    bGotGpsEph = False
    if os.path.exists(fullEphFilename):
        print(f"Reading GPS ephemeris from '{ephFilename}' file in local directory")
        allGpsEph = read_rinex_nav(fullEphFilename)
        fctSeconds = utc_to_gps(utcTime)[1]
        ephAge = [
            eph["GPS_Week"] * GpsConstants.WEEKSEC + eph["Toe"] - fctSeconds
            for eph in allGpsEph
        ]

        # Get index into fresh and healthy ephemeris (health bit set => unhealthy)
        iFreshAndHealthy = [
            i
            for i, eph in enumerate(allGpsEph)
            if abs(ephAge[i]) < GpsConstants.EPHVALIDSECONDS and not eph["health"]
        ]
        goodEphSvs = set([allGpsEph[i]["PRN"] for i in iFreshAndHealthy])

        if len(goodEphSvs) >= GnssThresholds.MINNUMGPSEPH:
            bGotGpsEph = True

    if not bGotGpsEph:
        print(f"\nGetting GPS ephemeris '{hourlyZFile}' from local directory ...")
        # If file is compressed, unzip it
        if hourlyZFile.endswith(".Z"):
            with gzip.open(os.path.join(dirName, hourlyZFile), "rb") as f_in:
                with open(fullEphFilename, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Successfully uncompressed ephemeris file '{ephFilename}'")

    allGpsEph = read_rinex_nav(fullEphFilename)

    return allGpsEph, allGloEph


def check_inputs(utcTime: List[int], dirName: str) -> Tuple[bool, str]:
    if len(utcTime) != 6:
        raise ValueError("utcTime must be a (1x6) vector")

    bOk = True
    if not dirName:
        return bOk, dirName

    if not os.path.isdir(dirName):
        bOk = False
        print(f"Error: directory '{dirName}' not found")
    else:
        if not dirName.endswith(os.sep):
            dirName = dirName + os.sep
    return bOk, dirName
