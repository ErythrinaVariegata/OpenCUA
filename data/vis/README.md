# AgentNet Visualizer

A Streamlit-based web application for visualizing AgentNet task trajectories and metadata.

## Features

- **Interactive Trajectory Viewer**: Browse through task execution steps with images and action details
- **Advanced Filtering**: Filter tasks by step length, quality, system, and domain
- **Fuzzy Search**: Search tasks by ID or instruction text
- **Image Annotation**: Automatically draws coordinate markers on screenshots
- **Navigation Tools**: Navigate between tasks with prev/next buttons, random selection, or direct ID jump

## Requirements

- Python 3.x
- Streamlit
- pandas
- PIL (Pillow)

Install dependencies:
```bash
pip install streamlit pandas pillow
```

## Usage

Run the application with the following command:

```bash
streamlit run app.py -- --traj_file <PATH_TO_TRAJ_JSONL> --meta_file <PATH_TO_META_JSONL> --image_dir <PATH_TO_IMAGE_DIR>
```

### Arguments

- `--traj_file`: Path to the trajectory JSONL file containing task execution steps
- `--meta_file`: Path to the metadata JSONL file containing task metadata (quality, system, domains, etc.)
- `--image_dir`: Path to the directory containing screenshot images referenced in trajectories

### Example

```bash
streamlit run app.py -- --traj_file data/trajectories.jsonl --meta_file data/metadata.jsonl --image_dir data/images/
```

## Interface Guide

### Sidebar Filters

- **Fuzzy Search**: Type task ID or instruction keywords to search
- **Step Length**: Adjust slider to filter by number of execution steps
- **Quality**: Multi-select filter for task quality labels (e.g., "good", "bad")
- **System**: Filter by agent system name
- **Domains**: Filter by task domain categories

### Main View

- **Navigation Bar**: Use Prev/Next buttons, Random selection, or Jump to ID input
- **Task Details**: View task ID, instruction, quality badges, and metadata
- **Trajectory Steps**: Each step displays:
  - Screenshot image with coordinate markers (if available)
  - Action type and code
  - Thought and observation (expandable)

## Data Format

The application expects:

- **Trajectory JSONL**: Each line is a JSON object with `task_id`, `instruction`, and `traj` (list of steps)
- **Metadata JSONL**: Each line is a JSON object with `task_id`, `quality`, `system`, `domains`, etc.
- **Images**: Screenshot files referenced by filename in trajectory steps

## Notes

- The app automatically merges trajectory and metadata files by `task_id`
- Coordinate markers are drawn when action code contains `x=` and `y=` parameters
- All filters work together - tasks must match all selected criteria to be displayed

