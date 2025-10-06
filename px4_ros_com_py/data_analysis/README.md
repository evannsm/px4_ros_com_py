# Trajectory Data Analysis Toolkit

Publication-quality trajectory analysis toolkit for PX4 flight data. Automatically processes CSV log files, detects trajectory planes, calculates RMSE metrics, and generates LaTeX-formatted figures and tables for research papers.

## Features

### 🔄 Automatic Data Processing
- **CSV Loading**: Automatically loads and cleans column names (removes `/plotjuggler/logging/` prefix)
- **Batch Processing**: Load all CSV files from a directory with one function call
- **Smart Metadata Extraction**: Automatically extracts platform, controller, and trajectory information from logged data columns (not from filenames!)
- **Trajectory Modifiers**: Detects and tracks 2x speed and spinning modifiers

### 📊 Trajectory Analysis
- **Automatic Plane Detection**: Detects whether trajectories are in XY, XZ, or YZ planes based on variance
- **Lookahead Time Alignment**: Automatically aligns reference and actual values to account for the control system's lookahead time
- **RMSE Calculation**:
  - Overall RMSE (position + yaw)
  - Position-only RMSE (x, y, z)
  - Per-axis RMSE
- **Computation Time Analysis**: Extracts and analyzes controller computation times
- **Trajectory Time**: Uses `traj_time` (trajectory time) instead of absolute timestamps for consistent time-based plotting

### 📈 Publication-Quality Plotting
- **2D Trajectory Plots**: Automatically plots in the correct plane (XY, XZ, or YZ)
- **Time Series**: Multi-panel plots with position and attitude over time
- **Multi-Controller Comparison**: Grid layouts comparing multiple controllers across trajectories
- **LaTeX Formatting**: Uses Computer Modern fonts and LaTeX rendering
- **PDF Export**: High-resolution PDFs ready for Overleaf/LaTeX documents

### 📋 Results Tables
- **Automated Table Generation**: Creates summary tables with RMSE and computation time
- **LaTeX Export**: Generates LaTeX table code ready to paste into papers
- **CSV Export**: Exports results as CSV for further analysis

## Installation

Required packages:
```bash
pip install pandas numpy matplotlib
```

For LaTeX rendering in plots, you need a LaTeX distribution:
- **Linux**: `sudo apt-get install texlive texlive-latex-extra cm-super`
- **macOS**: Install MacTeX
- **Windows**: Install MiKTeX

## Quick Start

### 1. Organize Your Data

Place CSV files exported from PlotJuggler in `log_files/` directory.

**Required Columns** (automatically logged by the ROS node):
- **Metadata** (enums from Logging.msg):
  - `platform` (0=Sim, 1=Hardware)
  - `controller` (0=NR Standard, 1=NR Enhanced, 2=MPC)
  - `trajectory` (0=Hover, 1=Circle H, 2=Circle V, 3=Fig8 H, etc.)
  - `traj_double` (boolean: 2x speed modifier)
  - `traj_spin` (boolean: spinning modifier)
- **State & Reference**:
  - Position: `x`, `y`, `z`
  - References: `x_ref`, `y_ref`, `z_ref`, `yaw_ref`
  - Time: `traj_time` (trajectory time, preferred), `time`, or `timestamp`
  - Lookahead: `lookahead_time` (for alignment correction)
- **Optional**:
  - Computation time: `comp_time` or `ctrl_comp_time`
  - Velocities: `vx`, `vy`, `vz`, etc.

The column prefix `/plotjuggler/logging/` is automatically removed during loading.

### 2. Run the Notebook

Open `DataAnalysis.ipynb` in Jupyter and run all cells. The notebook will:
1. Load all CSV files
2. Detect trajectory planes
3. Calculate RMSE metrics
4. Generate publication-quality plots
5. Create LaTeX tables
6. Save everything to `output/` directory

### 3. Use in Your Paper

Include the generated PDFs in your LaTeX document:
```latex
\begin{figure}
  \centering
  \includegraphics[width=\linewidth]{figures/multi_controller_comparison.pdf}
  \caption{Trajectory tracking comparison across controllers.}
  \label{fig:comparison}
\end{figure}
```

Copy the LaTeX table code from `output/results_table.tex` directly into your paper.

## Usage Examples

### Load and Analyze a Single File

```python
from utilities import *

# Load CSV
df = load_csv('log_files/flight_data.csv')

# Extract metadata from logged data (not filename!)
metadata = extract_metadata_from_data(df)
print(f"Platform: {metadata['platform']}")
print(f"Controller: {metadata['controller']}")
print(f"Trajectory: {metadata['trajectory']}")
print(f"2x Speed: {metadata['traj_double']}")
print(f"Spinning: {metadata['traj_spin']}")

# Detect trajectory plane
plane = detect_trajectory_plane(df)  # Returns 'xy', 'xz', or 'yz'

# Calculate RMSE
rmse = calculate_position_rmse(df)
rmse_per_axis = calculate_rmse_per_axis(df)

# Get computation time
comp_time = calculate_mean_comp_time(df)
```

### Load All Files and Generate Results Table

```python
# Load all CSVs
data_dict = load_all_csvs('log_files/')

# Print metadata for all datasets
print_dataset_metadata(data_dict)

# Generate results table (automatically extracts metadata from data columns)
results_df = generate_results_table(data_dict, use_data_metadata=True)

# Results table includes: Platform, Controller, Trajectory, Modifiers, RMSE, Comp_Time
print(results_df)

# Export to CSV and LaTeX
results_df.to_csv('output/results.csv', index=False)
latex_table = format_latex_table(results_df)
```

### Create Publication-Quality Plots

```python
import matplotlib.pyplot as plt

# Setup publication style (LaTeX fonts, high DPI)
setup_publication_style()

# Single trajectory plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_trajectory_2d(ax, df, plane='xy')  # Or let it auto-detect
plt.savefig('output/trajectory.pdf')

# Time series
fig = plot_time_series(df, vars_to_plot=['x', 'y', 'z', 'yaw'])
plt.savefig('output/timeseries.pdf')

# Multi-controller comparison
controller_groups = {
    'NR': ['nr_circle.csv', 'nr_line.csv'],
    'MPC': ['mpc_circle.csv', 'mpc_line.csv']
}
fig = plot_multi_controller_comparison(
    data_dict,
    controller_groups,
    save_path='output/comparison.pdf'
)
```

## API Reference

### Data Loading

#### `load_csv(file_path: str) -> pd.DataFrame`
Load a CSV file and clean column names (removes `/plotjuggler/logging/` prefix).

#### `load_all_csvs(directory: str) -> Dict[str, pd.DataFrame]`
Load all CSV files from a directory.

#### `extract_metadata_from_data(df: pd.DataFrame) -> Dict[str, str]`
Extract platform, controller, trajectory, and modifiers from logged data columns.
Returns dict with keys: `platform`, `controller`, `trajectory`, `traj_double`, `traj_spin`.
**This is the preferred method** - it uses actual logged enums, not filename parsing.

#### `extract_metadata_from_filename(filename: str) -> Dict[str, str]`
[DEPRECATED] Extract controller and trajectory information from filename.
Use `extract_metadata_from_data()` instead for accurate metadata.

#### `print_dataset_metadata(data_dict: Dict[str, pd.DataFrame])`
Print metadata summary for all datasets in a dictionary.

### Trajectory Analysis

#### `detect_trajectory_plane(df: pd.DataFrame) -> str`
Automatically detect trajectory plane ('xy', 'xz', or 'yz').

#### `align_reference_to_actual(df: pd.DataFrame, sampling_rate: float = 10.0) -> pd.DataFrame`
Align reference values to actual values by shifting reference backward in time to account for lookahead_time. Returns DataFrame with aligned data.

#### `get_flat_output_and_desired(df: pd.DataFrame, flip_z: bool = True, align_lookahead: bool = True, sampling_rate: float = 10.0) -> Tuple[np.ndarray, np.ndarray]`
Extract actual and reference trajectories as numpy arrays. By default, applies lookahead alignment before extraction.

### RMSE Calculation

#### `calculate_position_rmse(df: pd.DataFrame) -> float`
Calculate RMSE for position (x, y, z) only.

#### `calculate_overall_rmse(df: pd.DataFrame) -> float`
Calculate RMSE including yaw.

#### `calculate_rmse_per_axis(df: pd.DataFrame) -> Dict[str, float]`
Calculate RMSE for each axis separately.

#### `calculate_mean_comp_time(df: pd.DataFrame) -> Optional[float]`
Calculate mean computation time in milliseconds.

### Results Tables

#### `generate_results_table(data_dict, use_data_metadata=True, controller_map=None, trajectory_map=None) -> pd.DataFrame`
Generate comprehensive results table with Platform, Controller, Trajectory, Modifiers, RMSE, and Computation Time.

- `use_data_metadata=True` (recommended): Extract metadata from data columns
- `use_data_metadata=False`: Fall back to filename parsing (deprecated)
- `controller_map`, `trajectory_map`: Only used when `use_data_metadata=False`

#### `format_latex_table(df: pd.DataFrame) -> str`
Format DataFrame as LaTeX table.

### Plotting

#### `setup_publication_style()`
Configure matplotlib for publication-quality plots.

#### `plot_trajectory_2d(ax, df, plane=None, flip_z=True, align_lookahead=True, sampling_rate=10.0, ...)`
Plot 2D trajectory with automatic plane detection. By default, applies lookahead alignment and uses trajectory time (`traj_time`).

#### `plot_time_series(df, vars_to_plot=['x', 'y', 'z', 'yaw'], flip_z=True, align_lookahead=True, sampling_rate=10.0, ...)`
Create multi-panel time series plots. By default, applies lookahead alignment and uses trajectory time (`traj_time`) for the x-axis.

#### `plot_multi_controller_comparison(data_dict, controller_groups, ...)`
Create grid comparison of multiple controllers and trajectories.

## How Metadata Works

The analysis toolkit now uses **actual logged data** instead of filename parsing:

### Automatic Enumeration Mapping

When you run experiments, the ROS node automatically logs:
```
platform    → 0 (Sim) or 1 (Hardware)
controller  → 0 (NR Standard), 1 (NR Enhanced), 2 (MPC)
trajectory  → 0 (Hover), 1 (Circle H), 2 (Circle V), 3 (Fig8 H), etc.
traj_double → true/false (2x speed modifier)
traj_spin   → true/false (spinning modifier)
```

The analysis utilities automatically map these numeric values to human-readable names:
```python
PLATFORM_NAMES = {0: 'Simulation', 1: 'Hardware'}
CONTROLLER_NAMES = {0: 'NR Standard', 1: 'NR Enhanced', 2: 'MPC'}
TRAJECTORY_NAMES = {0: 'Hover', 1: 'Circle H', ...}
```

**Benefits:**
- ✅ No manual filename parsing needed
- ✅ Always accurate (uses actual test configuration)
- ✅ Handles trajectory modifiers automatically
- ✅ Works regardless of how you name your files

## Trajectory Plane Detection

The system automatically detects which plane your trajectory is in by calculating variance in the reference trajectory:

- **XY plane**: Horizontal trajectories (z relatively constant)
- **XZ plane**: Vertical circles in X direction
- **YZ plane**: Vertical circles in Y direction

You can override auto-detection by specifying `plane='xy'`, `plane='xz'`, or `plane='yz'` in plot functions.

## Lookahead Time Alignment

### Why Alignment is Needed

The control system computes reference values `lookahead_time` seconds in the future to enable predictive control. This means that at any given trajectory time `t`, the reference values logged correspond to time `t + lookahead_time`. Without alignment, RMSE calculations and trajectory plots would compare:
- Actual position at time `t`
- Reference position at time `t + lookahead_time`

This misalignment inflates RMSE values and makes plots misleading.

### Automatic Alignment

The toolkit automatically corrects for this lookahead by:
1. Reading the `lookahead_time` value from the logged data
2. Shifting reference values backward by `lookahead_time` seconds
3. Trimming data to only include properly aligned samples

**Example:**
- Logged `lookahead_time`: 1.2 seconds
- Data sampling rate: 10 Hz (0.1 second intervals)
- Shift amount: 12 samples backward

The alignment is applied by default in:
- `get_flat_output_and_desired()` - used for RMSE calculations
- `plot_trajectory_2d()` - used for trajectory plots
- `plot_time_series()` - used for time series plots

### Controlling Alignment

You can control alignment behavior with the `align_lookahead` parameter:

```python
# With alignment (default - recommended)
rmse = calculate_position_rmse(df)  # Uses alignment

# Disable alignment if needed
actual, ref = get_flat_output_and_desired(df, align_lookahead=False)

# Plot without alignment
plot_trajectory_2d(ax, df, align_lookahead=False)

# Custom sampling rate (if not 10 Hz)
plot_time_series(df, align_lookahead=True, sampling_rate=20.0)
```

### Time Column Usage

The toolkit now uses **`traj_time`** (trajectory time) instead of `timestamp` (absolute system time) for all time-based plotting. This provides:
- Consistent time axis starting from 0
- Independence from system clock variations
- Better alignment with trajectory phase

If `traj_time` is not available, it falls back to `time`, then to sample indices.

### Required Columns for Alignment

For automatic alignment to work, your CSV must include:
- `lookahead_time`: The lookahead time used by the controller (logged automatically)
- `traj_time`: Trajectory time for time-based plots (logged automatically)
- Reference columns ending in `_ref`: e.g., `x_ref`, `y_ref`, `z_ref`, `yaw_ref`

All of these are automatically logged by the ROS node during flight.

## Customization

### Custom Plot Styles

```python
# After setup_publication_style(), override specific settings
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 3
```

### Custom Enum Name Mappings

If you want to customize how enum values are displayed, edit the mappings in `utilities.py`:

```python
# In utilities.py
CONTROLLER_NAMES = {
    0: 'Newton-Raphson',  # Instead of 'NR Standard'
    1: 'NR Enhanced',
    2: 'Nonlinear MPC',   # Instead of 'MPC'
}

TRAJECTORY_NAMES = {
    0: 'Stationary Hover',  # Instead of 'Hover'
    1: 'Horizontal Circle',  # Instead of 'Circle H'
    # ... etc
}
```

The analysis will automatically use your custom names for all tables and plots.

### Add Custom Metrics

```python
# Calculate additional metrics
for filename, df in data_dict.items():
    # Max tracking error
    actual, ref = get_flat_output_and_desired(df)
    max_error = np.max(np.linalg.norm(actual[:, :3] - ref[:, :3], axis=1))

    # Settling time, overshoot, etc.
    # ... your custom analysis
```

## Output Directory Structure

```
output/
├── results_table.csv              # Results in CSV format
├── results_table.tex              # LaTeX table code
├── single_trajectory.pdf          # Example single trajectory
├── time_series.pdf                # Time series plot
├── multi_controller_comparison.pdf # Grid comparison
└── all_trajectories_grid.pdf      # All trajectories in grid
```

## Tips for Research Papers

1. **Consistent Naming**: Use consistent file naming for automatic grouping
2. **Add Computation Time**: Include `comp_time` column in CSVs for timing analysis
3. **High DPI**: All PDFs are saved at 300 DPI for publication quality
4. **Vector Graphics**: PDFs are vector graphics - they scale perfectly
5. **LaTeX Fonts**: Plots use Computer Modern fonts to match LaTeX documents
6. **Color Blind Friendly**: Consider using colorblind-friendly palettes:
   ```python
   actual_color = '#D55E00'  # Vermillion
   ref_color = '#0072B2'     # Blue
   ```

## Troubleshooting

### LaTeX Rendering Issues

If you get errors about LaTeX not being found:
```python
# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False
```

### Column Not Found Errors

Check your CSV column names:
```python
df = load_csv('your_file.csv')
print(df.columns.tolist())
```

The toolkit expects: `x`, `y`, `z`, `x_ref`, `y_ref`, `z_ref`, `yaw`, `yaw_ref`

### Memory Issues with Large Files

Process files individually:
```python
for csv_file in Path('log_files/').glob('*.csv'):
    df = load_csv(str(csv_file))
    # Process and plot
    # ...
    del df  # Free memory
```

## Contributing

To add new features:
1. Add functions to `utilities.py`
2. Add example usage to notebook
3. Document in this README

## License

MIT License - feel free to use in your research. If you use this toolkit in a publication, a citation would be appreciated!

## Support

For issues or questions, check:
1. CSV file structure and column names
2. LaTeX installation (for rendering)
3. Example notebook cells for usage patterns

Happy analyzing! 🚁📊
