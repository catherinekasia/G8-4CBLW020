# Notebook order

Run these notebooks in order:

1. `01_load_data.ipynb`
   - prepares forecast demand, LSOA/MSOA geography, and rurality
   - reads area data from `data/police_data.db`
   - reads allocation input tables from `data/allocation_model.db`
   - calculates crime weights from harm rank and response complexity rank
   - saves `msoa_bucket_demand_v1`, `msoa_context_v1`, `crime_bucket_mapping_v1`, and `crime_weight_model_v1`

Current crime weight formula:

`weight = (0.75 * harm_rank + 0.25 * response_complexity_rank) / 5`

2. `02_workforce_capacity.ipynb`
   - prepares the workforce capacity table
   - uses the whole wider function Local policing
   - saves `force_capacity_v1` and `capacity_settings_v1`

3. `03_allocation_v1.ipynb`
   - runs the v1 allocation model
   - change `RHO`, `RURALITY_WEIGHT`, or bucket shares here
   - adds an uncertainty flag when raw q25-q75 demand is wide and raw forecast demand changes sharply
   - writes final and intermediate allocation model tables to `data/allocation_model.db`
   - exports shareable CSV results to `outputs/`

4. `04_verification_checks.ipynb`
   - does not change the allocation model
   - checks whether LSOAs, MSOAs, forces, or final allocation rows were lost during joins and aggregation
   - saves verification summaries in `data/allocation_model.db`
   - exports shareable verification CSVs to `outputs/`

The notebooks only need two database files as inputs:

- `data/police_data.db`
- `data/allocation_model.db`

The CSV files in `outputs/` are generated results for sharing and checking. They are not needed as inputs when rerunning the model.

Current uncertainty thresholds:

- raw q25-q75 width divided by raw q50 demand is at least `1.25`
- raw month-to-month q50 demand change is at least `10%`
- raw absolute q50 demand change is at least `2`
