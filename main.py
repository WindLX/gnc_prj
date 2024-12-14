import matplotlib.pyplot as plt
from rinex2 import read_rinex_nav, rinex_to_sv, rinex_to_ecef
from datetime import datetime

if __name__ == "__main__":
    rinex_file_path = "./brdc3420.24n"
    rinex_head, rinex_data = read_rinex_nav(rinex_file_path)
    rinex_data_list = list(filter(lambda x: x.svprn == 19, rinex_data))

    rinex_coords = []

    for rinex_data in rinex_data_list:
        print("RINEX Data:")
        print(rinex_data)
        date = rinex_data.date
        print()

        rinex_sv = rinex_to_sv(rinex_data, date)
        print("SV Data from RINEX:")
        print(rinex_sv)
        print()

        rinex_ecef = rinex_to_ecef(rinex_data, date)
        print("ECEF Data from RINEX:")
        print(rinex_ecef)
        print()

        rinex_coords.append((rinex_ecef.x, rinex_ecef.y, rinex_ecef.z))

    rinex_x, rinex_y, rinex_z = zip(*rinex_coords)
