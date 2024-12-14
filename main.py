import numpy as np
import matplotlib.pyplot as plt
import os

from set_data_filter import set_data_filter
from read_gnss_logger import read_gnss_logger
from gps_to_utc import gps_to_utc
from get_nasa_hourly_ephemeris import get_nasa_hourly_ephemeris
from process_gnss_meas import process_gnss_meas
from gps_wls_pvt import gps_wls_pvt

# Define the directory and file names
dir_name = "./demoFiles"
pr_file_name = "gnss_log_2024_12_13_11_31_47.txt"

# Parameters (example: known true position in WGS84 coordinates)
lla_true_deg_deg_m = [37.422578, -122.081678, -28]  # Charleston Park Test Site

# Set the data filter
data_filter = set_data_filter()

# Read the GNSS logger file and extract data
gnss_raw, gnss_analysis = read_gnss_logger(dir_name, pr_file_name, data_filter)
if not gnss_raw:
    raise ValueError("No data found in the GNSS raw file.")

# Compute UTC time from the GNSS data
fct_seconds = 1e-3 * float(
    gnss_raw["allRxMillis"][-1]
)  # Convert milliseconds to seconds
utc_time = gps_to_utc([], fct_seconds)

# Get the online ephemeris from NASA FTP
all_gps_eph = get_nasa_hourly_ephemeris(utc_time, dir_name)
if not all_gps_eph:
    raise ValueError("Failed to get GPS ephemeris data.")

# Process raw GNSS measurements
gnss_meas = process_gnss_meas(gnss_raw)

# Plot pseudoranges and pseudorange rates
# Figure 1: Pseudoranges
# plt.figure()
# colors = PlotPseudoranges(gnss_meas, pr_file_name)

# # Figure 2: Pseudorange rates
# plt.figure()
# PlotPseudorangeRates(gnss_meas, pr_file_name, colors)

# # Figure 3: C/No (Carrier to Noise Density Ratio)
# plt.figure()
# PlotCno(gnss_meas, pr_file_name, colors)

# Compute the Weighted Least Squares (WLS) position and velocity
gps_pvt = gps_wls_pvt(gnss_meas, all_gps_eph)

# # Plot PVT results
# plt.figure()
# ts = "Raw Pseudoranges, Weighted Least Squares solution"
# PlotPvt(gps_pvt, pr_file_name, lla_true_deg_deg_m, ts)

# plt.figure()
# PlotPvtStates(gps_pvt, pr_file_name)

# # Plot accumulated delta range if available
# if np.any(np.isfinite(gnss_meas["AdrM"])) and np.any(gnss_meas["AdrM"] != 0):
#     gnss_meas = ProcessAdr(gnss_meas)
#     plt.figure()
#     PlotAdr(gnss_meas, pr_file_name, colors)

#     adr_resid = GpsAdrResiduals(gnss_meas, all_gps_eph, lla_true_deg_deg_m)
#     plt.figure()
#     PlotAdrResids(adr_resid, gnss_meas, pr_file_name, colors)

# plt.show()
