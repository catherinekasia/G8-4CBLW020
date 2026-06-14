# G8-4CBLW020
**Group 8 work for DBL**  
Use Python 3.11


## Structure

```
├── allocation
│   ├── 01_load_data.ipynb
│   ├── 02_workforce_capacity.ipynb
│   └── 03_allocation_v1.ipynb
│
├── crime_forecasting_tft/
│   ├── collect_test_predictions.py                 # 
│   ├── config.py                                   # Configuration (batch size, etc)
│   ├── data_loader.py                              # Getting raw data out of db and into df
│   ├── dataset.py                                  # TimeSeriesDataSet
│   ├── evaluate.py                                 # Evaluation scripts
│   ├── model.py                                    # Architecture
│   ├── preprocessing.py                            # Formatting for pytorch
│   └── train.py                                    # Training
│
├── data/
│   ├── raw/
│   │   ├── 2025_all_iod_scores_ranks_deciles.csv
│   │   └── uk_police_downloads/
│   └── police_data.db
│
├── data_pipeline/
│   ├── 01_load_data.ipynb                          # Reads LSOA/crime CSVs and saves to DB
│   ├── 02_clean_street_crimes.ipynb                # Drops rows, applies new crime types, checks for missing LSOAs, creates lsoa_month
│   ├── 03_clean_other_tables.ipynb                 # Creates month_info table, fixes lsoa_demographics, fixes lsoa_info
│   ├── 04_derived_attributes.ipynb                 # RA, diff, and spatial lag feature creation
│   ├── 05_prelim_graphs.ipynb                      # raw crime counts, outlier count detection
│   └── pstation_collection.ipynb                   # police station collection
│
├── retired/                           # Old code not ready to be deleted
├── .gitignore                         # Files/folders to be excluded from Git
├── README.md                          # Current document
├── requirements.txt                   # Libraries used
└── test.py                            
```
> **Notes:** `police_data.db` must remain in the `data` folder; data not in github because of size

## Environment Setup Instructions
Create a virtual environment to install packages + dependencies 

`conda create -n 4CBL python=3.11 -y`

`conda activate 4CBL`

`pip install -r requirements.txt`

`pip install "anywidget[dev]"`

==> always use this environment while working on project
> note: do NOT use pytorch 2.6 b/c of the new security feature (if you do straight pip install torch it'll be 2.6, only use requirements.txt)

## Data Setup + Usage Instructions

The data relevant to all parts of the project is available within ____. Place the data folder unchanged into G8-4CBLW020 File. The following subsection contains the order in which to run the files from start to finish of the recommendation pipeline detailed within the technical report.

data_pipeline/:
England and Wales are two separate databases already provided within the link above. Each notebook contains high-level summaries on what is done within each file. To run the full data pipeline from scratch the following must be ran:
1. Create_weather_dataset.ipynb
2. 01_load_data.ipynb
3. 01_load_data_Wales.ipynb
4. 02_clean_street_crimes.ipynb
5. 02_clean_street_crimes_Wales.ipynb
6. 03_clean_other_tables.ipynb
7. 03_clean_other_tables_Wales.ipynb
8. 04_derived_attributes.ipynb
9. 04_derived_attributes_Wales.ipynb
10. pstation_collection.ipynb
11. pstation_collection_Wales.ipynb

crime_forecasting_tft/:
1. model.py -> obtain optimal initial learning rate
2. train.py -> begin training
3. collect_test_predictions.py -> collect final inference on both England and Wales

allocation/:
1. 01_load_data.ipynb
2. 02_workforce_capacity.ipynb
3. 03_allocation_v1.ipynb


## TFT Info

`config.py` contains all configuration steps (batch size, seed, prediction length, etc)

`data_loader.py` contains loading all data into dataframe (getting raw data out of sqlite db)

`preprocessing.py` compute time index, ensure all cols in correct data format, null handling (format for pytorch)

`dataset.py` TimeSeriesDataSet + DataLoaders

`model.py` architecture 

`train.py` imports all functions from files and runs in order to train model

`evaluate.py` evaluation scripts




