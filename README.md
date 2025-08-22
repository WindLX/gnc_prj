A sophisticated Python-based project for processing and analyzing raw Global Navigation Satellite System (GNSS) data. This tool parses RINEX observation files to calculate satellite positions, visualize skyplots, and investigate critical phenomena in satellite navigation, such as Dilution of Precision (DOP) and its impact on positioning accuracy.

Project Overview
The core objective of this project is to transform raw GNSS data (collected via apps like GNSS Logger) into actionable insights. It focuses on the relationship between satellite geometry—specifically, Azimuth and Elevation—and the precision of a calculated position. A key challenge was visualizing this multi-dimensional relationship (Time, Azimuth, Elevation, DOP value), which was solved by creating an innovative 3D visualization with a temporal heatmap.

Key Features
RINEX File Parser: Processes standard RINEX observation files to extract satellite data, including pseudorange measurements, signal strength (CN0), and satellite IDs.

Satellite Position Calculation: Computes the precise position of each satellite in the Earth-Centered, Earth-Fixed (ECEF) coordinate frame at the time of observation.

Skyplot Visualization: Generates standard skyplots (Azimuth vs. Elevation) to visualize satellite constellations over time.

Dilution of Precision (DOP) Analysis: Calculates and analyzes various DOP metrics (GDOP, PDOP, HDOP, VDOP) which quantify how satellite geometry affects positional accuracy.

Advanced 3D Visualization: Creates a dynamic 3D plot (Azimuth, Elevation, Time) with a color-mapped heatmap for DOP values, effectively illustrating the "Delusion of Precision" phenomenon where good geometry falsely implies high accuracy.

Position Solving: implements a basic Least Squares algorithm to compute the receiver's position based on satellite pseudoranges.

Technology Stack
Language: Python 3

Core Libraries:

numpy - For numerical computations and matrix operations (essential for Least Squares).

matplotlib - For creating 2D skyplots and advanced 3D visualizations (mpl_toolkits.mplot3d).

pandas - For efficient data manipulation and handling of epoch-based observation data.

Data Source: Raw data collected and exported in RINEX format from Android apps like GNSS Logger.

Results and Implications
This project successfully demonstrates that the distribution of satellites in the sky is a primary factor in positioning accuracy, sometimes more so than the number of satellites. The innovative 3D visualization clearly shows how periods of apparently good satellite geometry (low DOP) can be misleading if the satellites are clustered, leading to a higher potential for error—the "Delusion of Precision."

These findings are critical for applications in:

Aerospace Engineering: Mission planning and GNC (Guidance, Navigation, and Control) systems.

Autonomous Systems: Path planning for drones and self-driving cars, where reliable GPS is crucial.

Geomatics: Surveying and precision agriculture.

Repository Structure
text
gnc_prj/
├── data/                   # Directory for raw RINEX observation files
├── src/
│   ├── rinex_parser.py    # Script to parse RINEX files and extract data
│   ├── satellite.py       # Contains functions for satellite position calculation
│   ├── geometry.py        # Functions for DOP calculation and geometry analysis
│   ├── visualization.py   # Functions for generating skyplots and 3D visualizations
│   └── main.py            # Main script to run the analysis pipeline
├── outputs/               # Generated plots and visualizations
├── requirements.txt       # Project dependencies
└── README.md              # This file
Installation and Usage
Clone the repository:

bash
git clone https://github.com/WindLX/gnc_prj.git
cd gnc_prj
Install dependencies:
It is recommended to use a virtual environment.

bash
pip install -r requirements.txt
Add your data:
Place your RINEX observation (.XXo) files in the data/ directory.

Run the analysis:
Execute the main script to process the data and generate visualizations.

bash
python src/main.py
The results, including skyplots and the 3D DOP visualization, will be saved in the outputs/ directory.

Future Enhancements
Integration of orbital data from SP3 or BRDC files for more precise satellite position calculations.

Implementation of more advanced positioning algorithms, such as Kalman Filtering.

A graphical user interface (GUI) for easier interaction and analysis.

Real-time data processing capabilities.
