import numpy as np


def read_rinex_nav(file_name):
    """
    Read GPS ephemeris and iono data from an ASCII formatted RINEX 2.10 Nav file.
    Input:
       file_name - string containing name of RINEX formatted navigation data file
    Output:
    gps_eph: list of ephemeris data, each element is an ephemeris dictionary
    iono: ionospheric parameter dictionary
    """

    def count_eph(fid_eph):
        num_hdr_lines = 0
        b_found_header = False
        while not b_found_header:
            num_hdr_lines += 1
            line = fid_eph.readline()
            if not line:
                break
            if "END OF HEADER" in line:
                b_found_header = True
                break
        if not b_found_header:
            raise ValueError(
                f"Error reading file: {file_name}\nExpected RINEX header not found"
            )

        num_eph = -1
        while True:
            num_eph += 1
            line = fid_eph.readline()
            if not line:
                break
            if len(line) != 79:
                raise ValueError("Incorrect line length encountered in RINEX file")

        if num_eph % 8 != 0:
            raise ValueError(
                f"Number of nav lines in {file_name} should be divisible by 8"
            )

        return num_eph // 8, num_hdr_lines

    def read_iono(fid_eph, num_hdr_lines):
        iono = {}
        b_iono_alpha = False
        b_iono_beta = False

        for _ in range(num_hdr_lines):
            line = fid_eph.readline()
            if "ION ALPHA" in line:
                iono["alpha"] = list(map(float, line.split()[:4]))
                b_iono_alpha = len(iono["alpha"]) == 4
            if "ION BETA" in line:
                iono["beta"] = list(map(float, line.split()[:4]))
                b_iono_beta = len(iono["beta"]) == 4

        if not (b_iono_alpha and b_iono_beta):
            iono = {}

        return iono

    def initialize_gps_eph():
        return {
            "PRN": 0,
            "Toc": 0,
            "af0": 0,
            "af1": 0,
            "af2": 0,
            "IODE": 0,
            "Crs": 0,
            "Delta_n": 0,
            "M0": 0,
            "Cuc": 0,
            "e": 0,
            "Cus": 0,
            "Asqrt": 0,
            "Toe": 0,
            "Cic": 0,
            "OMEGA": 0,
            "Cis": 0,
            "i0": 0,
            "Crc": 0,
            "omega": 0,
            "OMEGA_DOT": 0,
            "IDOT": 0,
            "codeL2": 0,
            "GPS_Week": 0,
            "L2Pdata": 0,
            "accuracy": 0,
            "health": 0,
            "TGD": 0,
            "IODC": 0,
            "ttx": 0,
            "Fit_interval": 0,
        }

    def utc2gps(utc_time):
        # Placeholder for actual UTC to GPS time conversion
        return utc_time

    with open(file_name, "r") as fid_eph:
        num_eph, num_hdr_lines = count_eph(fid_eph)
        fid_eph.seek(0)
        iono = read_iono(fid_eph, num_hdr_lines)

        gps_eph = [initialize_gps_eph() for _ in range(num_eph)]

        for j in range(num_eph):
            line = fid_eph.readline()
            gps_eph[j]["PRN"] = int(line[0:2])
            year = int(line[2:6])
            year = 2000 + year if year < 80 else 1900 + year
            month = int(line[6:9])
            day = int(line[9:12])
            hour = int(line[12:15])
            minute = int(line[15:18])
            second = float(line[18:22])
            gps_time = utc2gps([year, month, day, hour, minute, second])
            gps_eph[j]["Toc"] = gps_time[1]
            gps_eph[j]["af0"] = float(line[22:41])
            gps_eph[j]["af1"] = float(line[41:60])
            gps_eph[j]["af2"] = float(line[60:79])

            line = fid_eph.readline()
            gps_eph[j]["IODE"] = float(line[3:22])
            gps_eph[j]["Crs"] = float(line[22:41])
            gps_eph[j]["Delta_n"] = float(line[41:60])
            gps_eph[j]["M0"] = float(line[60:79])

            line = fid_eph.readline()
            gps_eph[j]["Cuc"] = float(line[3:22])
            gps_eph[j]["e"] = float(line[22:41])
            gps_eph[j]["Cus"] = float(line[41:60])
            gps_eph[j]["Asqrt"] = float(line[60:79])

            line = fid_eph.readline()
            gps_eph[j]["Toe"] = float(line[3:22])
            gps_eph[j]["Cic"] = float(line[22:41])
            gps_eph[j]["OMEGA"] = float(line[41:60])
            gps_eph[j]["Cis"] = float(line[60:79])

            line = fid_eph.readline()
            gps_eph[j]["i0"] = float(line[3:22])
            gps_eph[j]["Crc"] = float(line[22:41])
            gps_eph[j]["omega"] = float(line[41:60])
            gps_eph[j]["OMEGA_DOT"] = float(line[60:79])

            line = fid_eph.readline()
            gps_eph[j]["IDOT"] = float(line[3:22])
            gps_eph[j]["codeL2"] = float(line[22:41])
            gps_eph[j]["GPS_Week"] = float(line[41:60])
            gps_eph[j]["L2Pdata"] = float(line[60:79])

            line = fid_eph.readline()
            gps_eph[j]["accuracy"] = float(line[3:22])
            gps_eph[j]["health"] = float(line[22:41])
            gps_eph[j]["TGD"] = float(line[41:60])
            gps_eph[j]["IODC"] = float(line[60:79])

            line = fid_eph.readline()
            gps_eph[j]["ttx"] = float(line[3:22])
            gps_eph[j]["Fit_interval"] = float(line[22:41])

    return gps_eph, iono
