# G8-4CBLW020
**Group 8 work for DBL**  
Use Python (number idk yet)


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
│   └── 02_clean_database.ipynb                     # Cleans the data
│
├── requirements.txt                   # tbd
├── .gitignore                         # Files/folders to be excluded from Git
└── README.md                          # Current document
```
> **Notes:** `police_data.db` must remain in the `data` folder; data not in github because of size

1. 01_load_data.ipynb
2. 02_clean_database.ipynb
3. ....
