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
│
├── data_pipeline/
│   ├── 01_load_data.ipynb                          # Reads LSOA/crime CSVs and saves to DB
│   └── 02_clean_database.ipynb                     # Contains all pre-processing + cleaning + aggregating etc
│
├── requirements.txt                   # tbd
├── .gitignore                         # Files/folders to be excluded from Git
└── README.md                          # Current document
```
> **Notes:** `police_data.db` must remain in the `data` folder; data not in github because of size

1. 01_load_data.ipynb
2. 02_clean_database.ipynb
3. ....

## Environment Setup Instructions
Create a virtual environment to install packages + dependencies 

`conda create -n 4CBL python=3.11 -y`

`conda activate 4CBL`

`pip install -r requirements.txt`

==> always use this environment while working on project

## TFT Info

`config.py` contains all configuration steps (batch size, seed, prediction length, etc)

`data_loader.py` contains loading all data into dataframe

`preprocessing.py` compute time index, ensure all cols in correct data format, null handling



