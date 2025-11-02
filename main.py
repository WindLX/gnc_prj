import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.core.fromnumeric")
from datetime import datetime
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from pathlib import Path  # Added for Path handling

from model import Vector3
from utils import ecef_to_lla
from data_loader import DataLoader
from estimation import PositionEstimation
from evaluation import *

if __name__ == "__main__":
    file_index = 0
    plot_show_flag = False
    RAIM_flag = True  # Remember to set "enable_fault_detection = True" in estimation.py

    data_loader = DataLoader("./data")
    trajectory = data_loader[file_index]

    if RAIM_flag:
        result_dir_str = "./results_RAIM/" + str(file_index) + "/"
    else:
        result_dir_str = "./results/" + str(file_index) + "/"

    result_dir = Path(result_dir_str)  # Convert to Path for mkdir safety
    result_dir.mkdir(exist_ok=True, parents=True)  # Create dir early

    estimation = PositionEstimation(
        trajectory, step=5, threshold=8, result_dir=result_dir_str  # Pass str if Estimation expects it
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
    evaluation(df, result_dir=result_dir)  # Fixed: Now Path object

    plot_error(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_error_histograms(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_coordinates(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_ecef_coordinates(df, result_dir=result_dir, plot_flag=plot_show_flag)

    # dop
    plot_gdop(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_vdop(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_hdop(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_pdop(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_tdop(df, result_dir=result_dir, plot_flag=plot_show_flag)

    # histograms
    plot_gdop_histogram(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_vdop_histogram(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_hdop_histogram(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_pdop_histogram(df, result_dir=result_dir, plot_flag=plot_show_flag)
    plot_tdop_histogram(df, result_dir=result_dir, plot_flag=plot_show_flag)
    draw_satellite_tracks(
        satellite_position, result_dir=result_dir, plot_flag=plot_show_flag
    )

    # map
    plot_map(df, result_dir=result_dir, plot_mode=MapPlotMode.BOTH)