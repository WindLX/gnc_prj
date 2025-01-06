from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from model import Vector3
from utils import ecef_to_lla
from estimation import PositionEstimation
from map import Map, PathStyle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D


def evalution(df):
    mean_position = df[["ecef_x", "ecef_y", "ecef_z"]].mean()
    variance_position = df[["ecef_x", "ecef_y", "ecef_z"]].var()

    mean_errors = df[
        [
            "norm_error",
            "horizontal_error_lat",
            "horizontal_error_lon",
            "vertical_error",
        ]
    ].mean()
    variance_errors = df[
        [
            "norm_error",
            "horizontal_error_lat",
            "horizontal_error_lon",
            "vertical_error",
        ]
    ].var()

    print("Mean Estimated Position (ECEF):")
    print(mean_position)
    print("\nVariance of Estimated Position (ECEF):")
    print(variance_position)

    print("\nMean Errors:")
    print(mean_errors)
    print("\nVariance of Errors:")
    print(variance_errors)

    gdop_mean = df["GDOP"].mean()
    print(f"\nMean GDOP: {gdop_mean}")
    vdop_mean = df["VDOP"].mean()
    print(f"\nMean GDOP: {vdop_mean}")
    gdop_variance = df["GDOP"].var()
    print(f"Variance of GDOP: {gdop_variance}")

    log_path = Path(result_file) / "summary_results.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as f:
        f.write("Mean Estimated Position (ECEF):\n")
        f.write(f"  X: {mean_position['ecef_x']} meters\n")
        f.write(f"  Y: {mean_position['ecef_y']} meters\n")
        f.write(f"  Z: {mean_position['ecef_z']} meters\n")
        f.write("\nVariance of Estimated Position (ECEF):\n")
        f.write(f"  X: {variance_position['ecef_x']} meters^2\n")
        f.write(f"  Y: {variance_position['ecef_y']} meters^2\n")
        f.write(f"  Z: {variance_position['ecef_z']} meters^2\n")
        f.write("\nMean Errors:\n")
        f.write(f"  Norm Error: {mean_errors['norm_error']} meters\n")
        f.write(
            f"  Horizontal Error Latitude: {mean_errors['horizontal_error_lat']} meters\n"
        )
        f.write(
            f"  Horizontal Error Longitude: {mean_errors['horizontal_error_lon']} meters\n"
        )
        f.write(f"  Vertical Error: {mean_errors['vertical_error']} meters\n")
        f.write("\nVariance of Errors:\n")
        f.write(f"  Norm Error: {variance_errors['norm_error']} meters^2\n")
        f.write(
            f"  Horizontal Error Latitude: {variance_errors['horizontal_error_lat']} meters^2\n"
        )
        f.write(
            f"  Horizontal Error Longitude: {variance_errors['horizontal_error_lon']} meters^2\n"
        )
        f.write(f"  Vertical Error: {variance_errors['vertical_error']} meters^2\n")
        f.write("\nMean GDOP:\n")
        f.write(f"  {gdop_mean}\n")
        f.write("\nVariance of GDOP:\n")
        f.write(f"  {gdop_variance}\n")
        f.write(f"  {vdop_mean}\n")
        # f.write("\nVariance of GDOP:\n")
        # f.write(f"  {vdop_variance}\n")


def plot_error(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="time",
        y="norm_error",
        label="Norm Error",
    )

    sns.lineplot(
        data=df,
        x="time",
        y="horizontal_error_lat",
        label="Horizontal Error Latitude",
    )

    sns.lineplot(
        data=df,
        x="time",
        y="horizontal_error_lon",
        label="Horizontal Error Longitude",
    )

    sns.lineplot(data=df, x="time", y="vertical_error", label="Vertical Error")

    plt.title("Position Estimation Errors Over Time")
    plt.xlabel("Time")
    plt.ylabel("Error (meters)")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_file + "error.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_error_histograms(df, result_file, plot_flag=True):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(df["norm_error"], kde=True)
    plt.title("Norm Error Distribution")
    plt.xlabel("Norm Error (meters)")

    plt.subplot(2, 2, 2)
    sns.histplot(df["horizontal_error_lat"], kde=True)
    plt.title("Horizontal Error Latitude Distribution")
    plt.xlabel("Horizontal Error Latitude (meters)")

    plt.subplot(2, 2, 3)
    sns.histplot(df["horizontal_error_lon"], kde=True)
    plt.title("Horizontal Error Longitude Distribution")
    plt.xlabel("Horizontal Error Longitude (meters)")

    plt.subplot(2, 2, 4)
    sns.histplot(df["vertical_error"], kde=True)
    plt.title("Vertical Error Distribution")
    plt.xlabel("Vertical Error (meters)")

    plt.tight_layout()
    plt.savefig(result_file + "error_histograms.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_gdop(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="GDOP", label="GDOP")

    plt.title("GDOP Over Time")
    plt.xlabel("Time")
    plt.ylabel("GDOP")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_file + "gdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()



def plot_gdop_histogram(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    sns.histplot(df["GDOP"], kde=True)
    plt.title("GDOP Distribution")
    plt.xlabel("GDOP")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(result_file + "gdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


#VDOP

def plot_vdop(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="VDOP", label="VDOP")

    plt.title("Vertical Dilution of Precision (VDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("VDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_file + "vdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_vdop_histogram(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["VDOP"], kde=True, bins=20, color="blue", label="VDOP")
    plt.title("VDOP Histogram")
    plt.xlabel("VDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_file + "vdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


#HDOP

def plot_hdop(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="HDOP", label="HDOP")

    plt.title("Horizontal Dilution of Precision (HDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("HDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_file + "hdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_hdop_histogram(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["HDOP"], kde=True, bins=20, color="green", label="HDOP")
    plt.title("HDOP Histogram")
    plt.xlabel("HDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_file + "hdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


#PDOP


def plot_pdop(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="PDOP", label="PDOP")

    plt.title("Position Dilution of Precision (PDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("PDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_file + "pdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_pdop_histogram(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["PDOP"], kde=True, bins=20, color="red", label="PDOP")
    plt.title("PDOP Histogram")
    plt.xlabel("PDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_file + "pdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


#TDOP 

def plot_tdop(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="TDOP", label="TDOP")

    plt.title("Time Dilution of Precision (TDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("TDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_file + "tdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()

def plot_tdop_histogram(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["TDOP"], kde=True, bins=20, color="purple", label="TDOP")
    plt.title("TDOP Histogram")
    plt.xlabel("TDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_file + "tdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()




def plot_coordinates(df, result_file, plot_flag=True):
    plt.figure(figsize=(10, 6))

    plt.plot(df["lon"], df["lat"], label="Estimated Coordinates", color="blue")
    plt.plot(df["truth_lon"], df["truth_lat"], label="True Coordinates", color="red")

    plt.title("True vs Estimated Coordinates")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_file + "coordinates.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_grouped_dop_histogram(df, result_file, plot_flag=True):
    """
    Plots a grouped histogram for GDOP, HDOP, VDOP, PDOP, and TDOP.
    
    Parameters:
    - df: pandas DataFrame containing GDOP, HDOP, VDOP, PDOP, and TDOP columns.
    - result_file: Path to save the resulting histogram.
    - plot_flag: If True, displays the plot.
    """
    # Define bin edges based on the combined range of all DOP values
    dop_columns = ["GDOP", "HDOP", "VDOP", "PDOP", "TDOP"]
    all_data = np.concatenate([df[col].dropna().values for col in dop_columns])
    bins = np.linspace(all_data.min(), all_data.max(), 21)

    # Compute histograms for each DOP type
    histograms = {col: np.histogram(df[col].dropna(), bins=bins)[0] for col in dop_columns}

    # Bar positions and width
    bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
    width = (bins[1] - bins[0]) * 0.15  # Width for grouped bars

    # Colors for each DOP type
    colors = {
        "GDOP": "red",
        "HDOP": "green",
        "VDOP": "blue",
        "PDOP": "orange",
        "TDOP": "purple"
    }

    # Create the plot
    plt.figure(figsize=(12, 8))
    for i, (dop_type, hist) in enumerate(histograms.items()):
        plt.bar(bin_centers + i * width - (len(histograms) / 2) * width, 
                hist, 
                width=width, 
                label=dop_type, 
                color=colors[dop_type], 
                alpha=0.7)

    # Add labels, legend, and title
    plt.title("Grouped Histogram of DOP Values")
    plt.xlabel("DOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(result_file + "grouped_dop_histogram.png", dpi=300, bbox_inches="tight")

    # Show the plot if flag is True
    if plot_flag:
        plt.show()


# def plot_satellite_distribution_with_dop(satellite_data, result_file, plot_flag=True):
#     """
#     Plots a compound graph showing satellite distribution and DOP values.

#     Parameters:
#     - satellite_data: pandas DataFrame with 'Azimuth' and 'Elevation' columns for satellites.
#     - dop_values: pandas DataFrame with DOP values (GDOP, HDOP, VDOP, PDOP, TDOP).
#     - result_file: Path to save the resulting plot.
#     - plot_flag: If True, displays the plot.
#     """
#     # Create the figure with two subplots
#     fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={0: {'projection': 'polar'}})

#     ### Satellite Distribution (Polar Plot)
#     ax_polar = axs[0]
#     azimuth = np.radians(satellite_data["Azimuth"])  # Convert azimuth to radians
#     elevation = satellite_data["Elevation"]  # Elevation remains as-is
#     ax_polar.scatter(azimuth, elevation, c='blue', alpha=0.7, s=50, label="Satellites")
#     ax_polar.set_theta_zero_location('N')  # North as 0°
#     ax_polar.set_theta_direction(-1)  # Clockwise azimuth
#     ax_polar.set_title("Satellite Distribution")
#     ax_polar.legend(loc="upper right", fontsize="small")
#     ax_polar.set_rlabel_position(135)  # Radial labels

#     ### DOP Values (Grouped Histogram)
#     ax_bar = axs[1]
#     dop_types = dop_values.columns.tolist()
#     dop_vals = dop_values.mean().tolist()  # Take mean if multiple DOP records exist
#     ax_bar.bar(dop_types, dop_vals, color=["red", "green", "blue", "orange", "purple"], alpha=0.7)
#     ax_bar.set_title("DOP Values")
#     ax_bar.set_ylabel("DOP Value")
#     ax_bar.set_xlabel("DOP Type")

#     # Adjust layout and save the plot
#     plt.tight_layout()
#     plt.savefig(result_file + "satellite_distribution_with_dop.png", dpi=300, bbox_inches="tight")

#     # Show the plot if flag is True
#     if plot_flag:
#         plt.show()


def plot_map(df, result_file):
    map = Map()

    estimated_locations = [
        Vector3(row["lon"], row["lat"], 0) for _, row in df.iterrows()
    ]
    true_locations = [
        Vector3(row["truth_lon"], row["truth_lat"], 0) for _, row in df.iterrows()
    ]

    path_styles = [
        PathStyle(
            weight=2,
            color="0x0000FF",
            transparency=1,
            fillcolor="",
            fillTransparency=0.5,
        ),
        PathStyle(
            weight=2,
            color="0xFF0000",
            transparency=1,
            fillcolor="",
            fillTransparency=0.5,
        ),
    ]
    estimated_uri = map.construct_paths_uri([(estimated_locations, path_styles[0])])
    true_uri = map.construct_paths_uri([(true_locations, path_styles[1])])

    estimated_map_image = map.get_map_image(estimated_uri)
    true_map_image = map.get_map_image(true_uri)

    with open(result_file + "estimated_map.png", "wb") as f:
        f.write(estimated_map_image)

    with open(result_file + "true_map.png", "wb") as f:
        f.write(true_map_image)


def plot_ecef_coordinates(df, result_file, plot_flag=True):
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    sns.lineplot(data=df, x="time", y="ecef_x", label="ECEF X")
    plt.title("ECEF X Coordinate Over Time")
    plt.xlabel("Time")
    plt.ylabel("ECEF X (meters)")

    plt.subplot(3, 1, 2)
    sns.lineplot(data=df, x="time", y="ecef_y", label="ECEF Y")
    plt.title("ECEF Y Coordinate Over Time")
    plt.xlabel("Time")
    plt.ylabel("ECEF Y (meters)")

    plt.subplot(3, 1, 3)
    sns.lineplot(data=df, x="time", y="ecef_z", label="ECEF Z")
    plt.title("ECEF Z Coordinate Over Time")
    plt.xlabel("Time")
    plt.ylabel("ECEF Z (meters)")

    plt.tight_layout()

    plt.savefig(result_file + "ecef_coordinates.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def draw_satellite_tracks(satellite_position, result_file, plot_flag=True, max_gap=90):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="polar")
    colors = plt.colormaps["tab20"]

    for idx, prn in enumerate(satellite_position["prn"].unique()):
        sat_data = satellite_position[satellite_position["prn"] == prn]

        sat_data.loc[:, "azimuth"] = np.mod(sat_data["azimuth"], 2 * np.pi)
        sat_data.loc[:, "elevation"] = np.mod(sat_data["elevation"], 2 * np.pi)
        azimuth_diff = np.diff(sat_data["azimuth"])
        elevation_diff = np.diff(sat_data["elevation"])
        large_changes = (np.abs(azimuth_diff) > np.radians(max_gap)) | (
            np.abs(elevation_diff) > np.radians(max_gap)
        )
        segments = np.split(sat_data, np.where(large_changes)[0] + 1)

        for segment in segments:
            ax.plot(
                segment["azimuth"],
                90 - np.degrees(segment["elevation"]),
                color=colors(idx),
                label=f"PRN {prn}" if segment.index[0] == sat_data.index[0] else "",
            )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_rticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.grid(True)

    plt.title("Satellite Tracks on Skyplot", pad=40)
    plt.tight_layout()

    plt.savefig(result_file + "satellite_tracks.png", dpi=300, bbox_inches="tight")

    if plot_flag:
        plt.show()


if __name__ == "__main__":
    ##for i in range(0,8):
    file_index = 6
    plot_show_flag = True

    nav_files = ["./nav/brdc3490.24n", "./nav/brdc3630.24n"]
    if file_index == -1 or file_index == 7:
        nav_file = nav_files[1]
    else:
        nav_file = nav_files[0]

    obs_files = [
        "./data/1_Medium_Interference_Near_SAA_1525/gnss_log_2024_12_14_15_17_38.24o",
        "./data/2_Open_field_Lawn_1541/gnss_log_2024_12_14_15_28_53.24o",
        "./data/3_Forest_1542/gnss_log_2024_12_14_15_46_31.24o",
        "./data/4_LibraryBasement_1606/gnss_log_2024_12_14_16_02_14.24o",
        "./data/5_DormArea_HighInterference_1620/gnss_log_2024_12_14_16_14_30.24o",
        "./data/6_FootBall Field_1639/gnss_log_2024_12_14_16_32_45.24o",
        "./data/7_WayBackToSAA_1659/gnss_log_2024_12_14_16_49_34.24o",
        "./data/8_/gnss_log_2024_12_28_14_16_01.24o",
    ]
    obs_file = obs_files[file_index]

    track_files = [
        "./data/1_Medium_Interference_Near_SAA_1525/doc.kml",
        "./data/2_Open_field_Lawn_1541/doc.kml",
        "./data/3_Forest_1542/doc.kml",
        "./data/4_LibraryBasement_1606/doc.kml",
        "./data/5_DormArea_HighInterference_1620/doc.kml",
        "./data/6_FootBall Field_1639/doc.kml",
        "./data/7_WayBackToSAA_1659/doc.kml",
        "./data/8_/doc.kml",
    ]
    track_file = track_files[file_index]

    result_file = "./results/" + str(file_index) + "/"

    estimation = PositionEstimation(
        obs_file, nav_file, track_file, step=5, threshold=8, result_file=result_file
    )
    observations = estimation.load_observation_data()
    head, navigations = estimation.load_navigation_data(observations[0][0])

    evaluation_results = {
        "time": [],
        "norm_error": [],
        "horizontal_error_lat": [],
        "horizontal_error_lon": [],
        "vertical_error": [],
        "ecef_x": [],
        "ecef_y": [],
        "ecef_z": [],
        "lat": [],
        "lon": [],
        "alt": [],
        "truth_lat": [],
        "truth_lon": [],
        "truth_alt": [],
        "GDOP": [],
        "VDOP": [],  
        "HDOP": [],  
        "PDOP": [],  
        "TDOP": [],
    }

    satellite_position = None

    for idx, (time, observation) in tqdm(
        enumerate(observations), total=len(observations), desc="Processing Observations"
    ):
        satellite_info, truth_lla = estimation.extract_satellite_info(
            time, observation, head, navigations
        )
        try:
            estimation_result = estimation.estimate_position(
                time, truth_lla, satellite_info, is_init=(idx == 0)
            )
        except ValueError as e:
            print(f"Error: {e}")
            continue
        except RuntimeError as e:
            print(f"Error: {e}")
            continue

        estimation.draw_skymap(time, satellite_info)
        estimation.log(time, truth_lla, satellite_info, estimation_result)

        evaluation_results["time"].append(datetime(*time))
        evaluation_results["norm_error"].append(estimation_result[1]["norm_error"])
        evaluation_results["horizontal_error_lat"].append(
            estimation_result[1]["horizontal_error"][0]
        )
        evaluation_results["horizontal_error_lon"].append(
            estimation_result[1]["horizontal_error"][1]
        )
        evaluation_results["vertical_error"].append(
            estimation_result[1]["vertical_error"]
        )

        evaluation_results["GDOP"].append(estimation_result[1]["GDOP"])

        evaluation_results["VDOP"].append(estimation_result[1]["VDOP"])
        evaluation_results["HDOP"].append(estimation_result[1]["HDOP"])
        evaluation_results["PDOP"].append(estimation_result[1]["PDOP"])
        evaluation_results["TDOP"].append(estimation_result[1]["TDOP"])

        ecef_position = estimation_result[0]
        evaluation_results["ecef_x"].append(ecef_position[0])
        evaluation_results["ecef_y"].append(ecef_position[1])
        evaluation_results["ecef_z"].append(ecef_position[2])

        lla_position = ecef_to_lla(Vector3.from_list(ecef_position))
        evaluation_results["lat"].append(lla_position[0])
        evaluation_results["lon"].append(lla_position[1])
        evaluation_results["alt"].append(lla_position[2])

        evaluation_results["truth_lat"].append(truth_lla[0])
        evaluation_results["truth_lon"].append(truth_lla[1])
        evaluation_results["truth_alt"].append(truth_lla[2])

        for sat_info in satellite_info:
            new_row = pd.DataFrame(
                {
                    "time": [datetime(*time)],
                    "prn": [sat_info["prn"]],
                    "azimuth": [sat_info["azimuth"]],
                    "elevation": [sat_info["elevation"]],
                }
            )
            if satellite_position is None:
                satellite_position = new_row
            else:
                satellite_position = pd.concat(
                    [satellite_position, new_row], ignore_index=True
                )

    df = pd.DataFrame(evaluation_results)
    sns.set_theme(style="whitegrid")
    evalution(df)
    plot_map(df, result_file=result_file)
    plot_error(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_grouped_dop_histogram(df, result_file, plot_flag=True)
    plot_error_histograms(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_coordinates(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_ecef_coordinates(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_gdop(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_vdop(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_hdop(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_pdop(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_tdop(df, result_file=result_file, plot_flag=plot_show_flag)
    
    #histograms
    
    plot_gdop_histogram(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_vdop_histogram(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_hdop_histogram(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_pdop_histogram(df, result_file=result_file, plot_flag=plot_show_flag)
    plot_tdop_histogram(df, result_file=result_file, plot_flag=plot_show_flag)
    draw_satellite_tracks(
         satellite_position, result_file=result_file, plot_flag=plot_show_flag)



def calculate_gdop(satellite_position):
    gdop_values = []
    times = satellite_position["time"].unique()
    
    for t in times:
        sat_data = satellite_position[satellite_position["time"] == t]
        
        azimuth = sat_data["azimuth"].values
        elevation = sat_data["elevation"].values
        elevation = np.maximum(elevation, 0.01)  
        
        G = np.column_stack((
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
            np.ones(len(azimuth))
        ))
        
        try:
            Q = np.linalg.inv(G.T @ G)  
            gdop = np.sqrt(np.trace(Q)) 
        except np.linalg.LinAlgError:
            gdop = np.nan  
        
        gdop_values.append((t, gdop))
    
    gdop_df = pd.DataFrame(gdop_values, columns=["time", "gdop"])
    return gdop_df
var = calculate_gdop(satellite_position)
print(var)

def plot_satellite_distribution_vs_gdop(satellite_position, gdop_df):

    merged_data = satellite_position.merge(gdop_df, on="time")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        merged_data["azimuth"],
        merged_data["elevation"],
        c=merged_data["gdop"],
        cmap="viridis",
        alpha=0.7
    )
    plt.colorbar(scatter, label="GDOP")
    plt.title("Satellite Distribution vs GDOP")
    plt.xlabel("Azimuth (radians)")
    plt.ylabel("Elevation (radians)")
    plt.grid(True)
    plt.show()

gdop_df = calculate_gdop(satellite_position)
plot_satellite_distribution_vs_gdop(satellite_position, gdop_df)



def plot_gdop_with_satellite_distribution_3d(satellite_position, gdop_df):
   
    satellite_position["time"] = pd.to_datetime(satellite_position["time"])
    
    
    time_vals = []
    azimuth_vals = []
    elevation_vals = []
    gdop_vals = []

    for t in gdop_df["time"]:
        sat_data = satellite_position[satellite_position["time"] == t]
        gdop = gdop_df[gdop_df["time"] == t]["gdop"].values[0]
        
        time_vals.extend([t] * len(sat_data))
        azimuth_vals.extend(sat_data["azimuth"].values)
        elevation_vals.extend(sat_data["elevation"].values)
        gdop_vals.extend([gdop] * len(sat_data))
    
    time_vals = [(pd.Timestamp(t).timestamp() - pd.Timestamp(time_vals[0]).timestamp()) for t in time_vals]

    azimuth_vals = np.degrees(azimuth_vals) if max(azimuth_vals) <= 2 * np.pi else azimuth_vals
    elevation_vals = np.degrees(elevation_vals) if max(elevation_vals) <= np.pi else elevation_vals

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(time_vals, azimuth_vals, elevation_vals, c=gdop_vals, cmap='viridis', s=50)

    ax.set_xlabel('Time (seconds since start)', fontsize=12)
    ax.set_ylabel('Azimuth (degrees)', fontsize=12)
    ax.set_zlabel('Elevation (degrees)', fontsize=12)
    ax.set_title('GDOP with Satellite Distribution (Time, Azimuth, Elevation)', fontsize=14)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('GDOP', fontsize=12)

    ax.set_xlim(min(time_vals), max(time_vals)) 
    ax.set_ylim(0, 360)  
    ax.set_zlim(0, 90)  
    plt.tight_layout()
    plt.show()

plot_gdop_with_satellite_distribution_3d(satellite_position, gdop_df)


def calculate_dops(satellite_position):
  
    dop_values = []
    times = satellite_position["time"].unique()
    
    for t in times:
        sat_data = satellite_position[satellite_position["time"] == t]
        
        azimuth = sat_data["azimuth"].values
        elevation = sat_data["elevation"].values
        elevation = np.maximum(elevation, 0.01)  
        
        G = np.column_stack((
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
            np.ones(len(azimuth))
        ))
        
        try:
            Q = np.linalg.inv(G.T @ G) 
            gdop = np.sqrt(np.trace(Q))               
            hdop = np.sqrt(Q[0, 0] + Q[1, 1])         
            vdop = np.sqrt(Q[2, 2])                   
            tdop = np.sqrt(Q[3, 3])                   
            pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])  
        except np.linalg.LinAlgError:
            gdop, hdop, vdop, tdop, pdop = np.nan, np.nan, np.nan, np.nan, np.nan
        
        dop_values.append((t, gdop, hdop, vdop, tdop, pdop))
    
    dop_df = pd.DataFrame(dop_values, columns=["time", "gdop", "hdop", "vdop", "tdop", "pdop"])
    return dop_df

satellite_position = pd.DataFrame({
    "time": pd.date_range(start="2025-01-05 12:00:00", periods=10, freq="S").repeat(5),
    "azimuth": np.random.uniform(0, 2 * np.pi, 50),
    "elevation": np.random.uniform(0, np.pi / 2, 50)
})

dop_df = calculate_dops(satellite_position)
print(dop_df)


def plot_dops_with_satellite_distribution_3d(satellite_position, dop_df):
   
    satellite_position["time"] = pd.to_datetime(satellite_position["time"])
    dop_df["time"] = pd.to_datetime(dop_df["time"])

    time_vals = []
    azimuth_vals = []
    elevation_vals = []
    gdop_vals, hdop_vals, vdop_vals, tdop_vals, pdop_vals = [], [], [], [], []

    for t in dop_df["time"]:
        sat_data = satellite_position[satellite_position["time"] == t]
        gdop = dop_df[dop_df["time"] == t]["gdop"].values[0]
        hdop = dop_df[dop_df["time"] == t]["hdop"].values[0]
        vdop = dop_df[dop_df["time"] == t]["vdop"].values[0]
        tdop = dop_df[dop_df["time"] == t]["tdop"].values[0]
        pdop = dop_df[dop_df["time"] == t]["pdop"].values[0]

        time_vals.extend([t] * len(sat_data))
        azimuth_vals.extend(sat_data["azimuth"].values)
        elevation_vals.extend(sat_data["elevation"].values)
        gdop_vals.extend([gdop] * len(sat_data))
        hdop_vals.extend([hdop] * len(sat_data))
        vdop_vals.extend([vdop] * len(sat_data))
        tdop_vals.extend([tdop] * len(sat_data))
        pdop_vals.extend([pdop] * len(sat_data))

    time_start = pd.Timestamp(dop_df["time"].iloc[0]).timestamp()
    time_vals = [(pd.Timestamp(t).timestamp() - pd.Timestamp(time_vals[0]).timestamp())*100 for t in time_vals]

    azimuth_vals = np.degrees(azimuth_vals) if max(azimuth_vals) <= 2 * np.pi else azimuth_vals
    elevation_vals = np.degrees(elevation_vals) if max(elevation_vals) <= np.pi else elevation_vals

    fig = plt.figure(figsize=(20, 16))
    dop_types = [("GDOP", gdop_vals), ("HDOP", hdop_vals), ("VDOP", vdop_vals), ("TDOP", tdop_vals), ("PDOP", pdop_vals)]

    for i, (dop_name, dop_vals) in enumerate(dop_types, start=1):
        ax = fig.add_subplot(3, 2, i, projection='3d')

        scatter = ax.scatter(time_vals, azimuth_vals, elevation_vals, c=dop_vals, cmap='viridis', s=50)

        ax.set_xlabel('Time (seconds since start)', fontsize=10)
        ax.set_ylabel('Azimuth (degrees)', fontsize=10)
        ax.set_zlabel('Elevation (degrees)', fontsize=10)
        ax.set_title(f'{dop_name} with Satellite Distribution', fontsize=12)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label(f'{dop_name}', fontsize=10)

        ax.set_xlim(min(time_vals), max(time_vals))  
        ax.set_ylim(0, 360)  
        ax.set_zlim(0, 90)  

    plt.tight_layout()
    plt.show()

dop_df = calculate_dops(satellite_position)
plot_dops_with_satellite_distribution_3d(satellite_position, dop_df)