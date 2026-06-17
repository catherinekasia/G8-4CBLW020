# G8-4CBLW020
Group 8 work for CBL
Use Python 3.11

## Setup

```bash
conda create -n 4CBL python=3.11 -y
conda activate 4CBL
pip install -r requirements.txt
pip install "anywidget[dev]"
```

> Always use the `4CBL` environment. Do NOT run `pip install torch` b/c it will pull PyTorch 2.6 which has a breaking security change

## Structure

```
├── allocation/
│   ├── 00_build_allocation_inputs.ipynb      # Builds allocation_model.db from source files
│   ├── 01_load_data.ipynb                    # Prepares forecast demand, geography, rurality, and crime weights
│   ├── 02_workforce_capacity.ipynb           # Prepares Local policing workforce capacity
│   ├── 03_allocation.ipynb                   # Runs the msoa allocation model
│   └── 04_allocation_pressure_review.ipynb   # Adds the low allocation pressure flag
│
├── crime_forecasting_tft/
│   ├── config.py                       # Batch size, seed, prediction length, etc.
│   ├── data_loader.py                  # Reads raw data from SQLite DB into a DataFrame
│   ├── dataset.py                      # TimeSeriesDataSet and DataLoaders
│   ├── preprocessing.py                # Time index, dtype coercion, null handling
│   ├── model.py                        # TFT architecture
│   ├── train.py                        # Runs the full training pipeline
│   ├── evaluate.py                     # Evaluation scripts
│   ├── collect_test_predictions.py     # Runs inference and writes predictions to CSV
│   └── tbd
│
├── dashboard/
│   └── dashboard.ipynb                 # Runs dashboard inside the notebook
│
├── data/                               # Data from OneDrive Folder
│
├── data_pipeline/
│   ├── 01_load_data.ipynb              # Reads LSOA/crime CSVs and saves to DB
│   ├── 02_clean_street_crimes.ipynb    # Drops rows, remaps crime types, checks for missing LSOAs
│   ├── 03_clean_other_tables.ipynb     # month_info table, lsoa_demographics, lsoa_info
│   ├── 04_derived_attributes.ipynb     # Rolling averages, diff, and spatial lag features
│   ├── 05_prelim_graphs.ipynb          # Raw crime counts, outlier detection
│   └── pstation_collection.ipynb       # Police station data collection
│
├── retired/                           # Old code not ready to be deleted
├── .gitignore                         # Files/folders to be excluded from Git
├── README.md                          # Current document
├── requirements.txt                   # Libraries used
└── test.py 
```

## Data

The data folder is available at [https://tuenl-my.sharepoint.com/:f:/g/personal/a_bekesi_student_tue_nl/IgAwp8xMiJRlR5tPSwWYgYh2AXXqx2UuGWakbt29Lwx6FZo?e=8DEoEp]. Place it unchanged into `G8-4CBLW020/`.

## Running the Pipeline

**Data pipeline** (England and Wales are separate DBs — run each pair of notebooks for both):

1. `Create_weather_dataset.ipynb`
2. `01_load_data.ipynb` + `01_load_data_Wales.ipynb`
3. `02_clean_street_crimes.ipynb` + `02_clean_street_crimes_Wales.ipynb`
4. `03_clean_other_tables.ipynb` + `03_clean_other_tables_Wales.ipynb`
5. `04_derived_attributes.ipynb` + `04_derived_attributes_Wales.ipynb`
6. `pstation_collection.ipynb` + `pstation_collection_Wales.ipynb`

**TFT model:**

1. `model.py` — find optimal initial learning rate
2. `train.py` — train the model
3. `collect_test_predictions.py` — run inference on both England and Wales, writes CSVs to `data/`
4. `evaluate.py` / `plot_horizon_mae.py` — evaluation and plots (reads from CSVs, no re-inference needed)

**Allocation:**

1. `00_build_allocation_inputs.ipynb` — builds `allocation_model.db` from the source files in `data/`
2. `01_load_data.ipynb` — prepares demand, MSOA context, and crime weights
3. `02_workforce_capacity.ipynb` — prepares Local policing FTE capacity
4. `03_allocation.ipynb` — runs the allocation and writes `msoa_allocation.csv` and `fte_resources.csv`
5. `04_allocation_pressure_review.ipynb` — adds the low allocation pressure flag to `msoa_allocation.csv`

**Dashboardd**
1. `dashboard.ipynb` - loads the dashboard

## Tutorials Used
Outside of documentation, the following tutorials have also been used assist in writing the code within the repo:
1. [https://darribas.org/gds_course/content/bE/lab_E.html#data]
2. [https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html]
3. [https://towardsdatascience.com/ipyleaflet-interactive-mapping-in-jupyter-notebook-994f19611e79/]
