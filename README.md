# G8-4CBLW020
**Group 8 work for DBL**  
Use Python 3.11


## Structure

```
├── data/
│   ├── raw/
│   │   ├── 2025_all_iod_scores_ranks_deciles.csv
│   │   └── uk_police_downloads/
│   └── police_data.db
│
├── crime_forecasting_tft/
│   ├── config.py                                   # Config steps (batch size etc)
│   ├── data_loader.py                              # Getting raw data out of db and into df
│   ├── dataset.py                                  # TimeSeriesDataSet
│   ├── evaluate.py                                 # Evaluation scripts
│   ├── hypertune.py                               #in progress
│   ├── model.py                                    # Architecture
│   ├── preprocessing.py                            # Formatting for pytorch
│   └── train.py                                    # Training
│
├── data_pipeline/
│   ├── 01_load_data.ipynb                          # Reads LSOA/crime CSVs and saves to DB
│   └── 02_clean_database.ipynb                     # Contains all pre-processing + cleaning + aggregating etc
│
├── requirements.txt                   # Libraries used
├── .gitignore                         # Files/folders to be excluded from Git
└── README.md                          # Current document
```
> **Notes:** File reorganization in process; `police_data.db` must remain in the `data` folder; data not in github because of size

1. 01_load_data.ipynb
2. 02_clean_database.ipynb
3. ....

## Environment Setup Instructions
Create a virtual environment to install packages + dependencies 

`conda create -n 4CBL python=3.11 -y`

`conda activate 4CBL`

`pip install -r requirements.txt`

==> always use this environment while working on project
> note: do NOT use pytorch 2.6 b/c of the new security feature (if you do straight pip install torch it'll be 2.6, only use requirements.txt)

## TFT Info

`config.py` contains all configuration steps (batch size, seed, prediction length, etc)

`data_loader.py` contains loading all data into dataframe (getting raw data out of sqlite db)

`preprocessing.py` compute time index, ensure all cols in correct data format, null handling (format for pytorch)

`dataset.py` TimeSeriesDataSet + DataLoaders

`model.py` architecture 

`train.py` imports all functions from files and runs in order to train model

`evaluate.py` evaluation scripts




