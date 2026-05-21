## claude assisted code, reviewed and edited and looks like it works; couldn't figure it out fully without

"""
Migrate 2011 LSOA codes in street_crimes to 2021 LSOA codes.

The police data from before 2021 used the old 2011 LSOA boundary codes.
1,034 of those codes no longer exist in the 2021 boundary set stored in lsoa_info.

Strategy: for each old 2011 LSOA, compute the intersection area with every
2021 LSOA it overlaps, then map it to the 2021 LSOA with the largest overlap.
This handles both simple 1-to-1 renames and cases where a 2011 LSOA was split
across multiple 2021 LSOAs (we pick the dominant one by area).

All area calculations are done in EPSG:27700 (British National Grid, metres)
so the values are meaningful.

Usage:
    python migrate_lsoa_2011_to_2021.py            # live run — commits changes
    python migrate_lsoa_2011_to_2021.py --dry-run  # full rehearsal, rolls back
"""

import sys
import sqlite3
import geopandas as gpd
import pandas as pd

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    print("=== DRY RUN — no changes will be committed ===\n")

DB_PATH      = "../data/police_data.db"
SHP_2011     = "../data/LSOA_2011/LSOA_2011_EW_BFC_V3.shp"
GEOJSON_2021 = "../data/lsoa_spatial.geojson"

print("Connecting to database ...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    SELECT DISTINCT sc.lsoa_code
    FROM street_crimes sc
    LEFT JOIN lsoa_info li ON sc.lsoa_code = li.lsoa_code
    WHERE li.lsoa_code IS NULL
      AND sc.lsoa_code IS NOT NULL
""")
old_codes = [r[0] for r in cur.fetchall()]
print(f"Old LSOA codes to remap: {len(old_codes):,}")

cur.execute("""
    SELECT COUNT(*)
    FROM street_crimes sc
    LEFT JOIN lsoa_info li ON sc.lsoa_code = li.lsoa_code
    WHERE li.lsoa_code IS NULL
      AND sc.lsoa_code IS NOT NULL
""")
n_rows = cur.fetchone()[0]
print(f"Rows that will be updated: {n_rows:,}")

# ── 2. Load geometries ───────────────────────────────────────────────────────
print("\nLoading 2011 shapefile ...")
gdf11 = gpd.read_file(SHP_2011)  # EPSG:27700
gdf11 = gdf11[gdf11["LSOA11CD"].isin(old_codes)][["LSOA11CD", "geometry"]].copy()
print(f"  Old LSOAs loaded: {len(gdf11):,}")

print("Loading 2021 GeoJSON ...")
gdf21 = gpd.read_file(GEOJSON_2021)                 # EPSG:4326
gdf21 = gdf21[["LSOA21CD", "geometry"]].copy()
gdf21 = gdf21.to_crs(gdf11.crs)                     # reproject to EPSG:27700
print(f"  2021 LSOAs loaded: {len(gdf21):,}")

# ── 3. Spatial overlay: largest-area winner ──────────────────────────────────
print("\nComputing spatial overlay (this takes ~1-2 minutes) ...")
overlay = gpd.overlay(gdf11, gdf21, how="intersection", keep_geom_type=False)
overlay["area"] = overlay.geometry.area

# Attach original area of each old LSOA so we can compute coverage %
old_areas = gdf11.copy()
old_areas["old_area"] = old_areas.geometry.area
overlay = overlay.merge(old_areas[["LSOA11CD", "old_area"]], on="LSOA11CD", how="left")
overlay["pct_covered"] = (overlay["area"] / overlay["old_area"] * 100).round(1)

# For each old code, keep the 2021 LSOA with the greatest intersection area
best = (
    overlay.sort_values("area", ascending=False)
    .drop_duplicates(subset="LSOA11CD", keep="first")
    [["LSOA11CD", "LSOA21CD", "pct_covered"]]
    .rename(columns={"LSOA11CD": "old_code", "LSOA21CD": "new_code"})
)
lookup = best[["old_code", "new_code"]].copy()

# ── 3a. Coverage diagnostics ─────────────────────────────────────────────────
print(f"\nLookup table built:")
print(f"  Old codes mapped: {len(lookup):,} / {len(old_codes):,}")
print(f"  Distinct 2021 codes assigned: {lookup['new_code'].nunique():,}")

# Distribution of how much of each old LSOA the winning 2021 LSOA covers
bins = [0, 50, 70, 90, 100]
labels = ["<50%", "50-70%", "70-90%", "90-100%"]
best["coverage_band"] = pd.cut(best["pct_covered"], bins=bins, labels=labels, right=True)
coverage_dist = best["coverage_band"].value_counts().sort_index()
print("\n  Coverage of old LSOA area by winning 2021 LSOA:")
for band, count in coverage_dist.items():
    print(f"    {band}: {count:>4} codes")

# Flag low-confidence matches (winner covers < 50% of the old LSOA)
low_conf = best[best["pct_covered"] < 50].sort_values("pct_covered")
if not low_conf.empty:
    print(f"\n  LOW-CONFIDENCE matches (winner < 50% area coverage) — inspect these:")
    print(low_conf[["old_code", "new_code", "pct_covered"]].to_string(index=False))
else:
    print("\n  No low-confidence matches (all winners cover ≥ 50% of old LSOA). ✓")

# Show a sample of the full lookup
print("\nSample mapping (old → new, % of old LSOA covered by winner):")
print(best[["old_code", "new_code", "pct_covered"]].head(20).to_string(index=False))

# Check unmapped
unmapped = set(old_codes) - set(lookup["old_code"])
if unmapped:
    print(f"\n  WARNING: {len(unmapped)} old codes had no spatial match:")
    for c in sorted(unmapped):
        print(f"    {c}")
else:
    print("\n  All old codes have a spatial match. ✓")

# Check that every new code is a valid 2021 LSOA
cur.execute("SELECT lsoa_code FROM lsoa_info")
valid_2021 = {r[0] for r in cur.fetchall()}
invalid_new = set(lookup["new_code"]) - valid_2021
if invalid_new:
    print(f"\n  WARNING: {len(invalid_new)} mapped 2021 codes not in lsoa_info:")
    for c in sorted(invalid_new)[:20]:
        print(f"    {c}")
else:
    print("  All mapped 2021 codes exist in lsoa_info. ✓")

# ── 4. Update the database ───────────────────────────────────────────────────
mode_label = "DRY RUN" if DRY_RUN else "LIVE"
print(f"\n[{mode_label}] Applying changes to street_crimes ...")

# Write mapping to a temp table for a single efficient UPDATE
cur.execute("DROP TABLE IF EXISTS _lsoa_remap")
cur.execute("""
    CREATE TEMP TABLE _lsoa_remap (
        old_code TEXT PRIMARY KEY,
        new_code TEXT
    )
""")
cur.executemany(
    "INSERT INTO _lsoa_remap VALUES (?, ?)",
    lookup[["old_code", "new_code"]].values.tolist(),
)

# Index on lsoa_code speeds up the WHERE clause on 82M rows
print("  Creating index on street_crimes(lsoa_code) ...")
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_street_crimes_lsoa
    ON street_crimes(lsoa_code)
""")

print("  Running UPDATE ...")
cur.execute("""
    UPDATE street_crimes
    SET lsoa_code = (
        SELECT new_code FROM _lsoa_remap
        WHERE _lsoa_remap.old_code = street_crimes.lsoa_code
    )
    WHERE lsoa_code IN (SELECT old_code FROM _lsoa_remap)
""")
n_updated = cur.rowcount
print(f"  Rows updated: {n_updated:,}")

# ── 5. Verify (inside the same transaction) ───────────────────────────────────
cur.execute("""
    SELECT COUNT(DISTINCT sc.lsoa_code)
    FROM street_crimes sc
    LEFT JOIN lsoa_info li ON sc.lsoa_code = li.lsoa_code
    WHERE li.lsoa_code IS NULL
      AND sc.lsoa_code IS NOT NULL
""")
remaining_after = cur.fetchone()[0]
print(f"  Unrecognised codes remaining after update: {remaining_after}")

# Spot-check: show before/after for 10 sample old codes
sample_old = old_codes[:10]
placeholders = ",".join("?" * len(sample_old))
cur.execute(f"""
    SELECT lsoa_code, COUNT(*) as n
    FROM street_crimes
    WHERE lsoa_code IN ({placeholders})
    GROUP BY lsoa_code
""", sample_old)
still_old = cur.fetchall()

cur.execute(f"""
    SELECT r.new_code, COUNT(*) as n
    FROM street_crimes sc
    JOIN _lsoa_remap r ON sc.lsoa_code = r.new_code
    WHERE r.old_code IN ({placeholders})
    GROUP BY r.new_code
    LIMIT 10
""", sample_old)
now_new = cur.fetchall()

print("\n  Spot-check — old codes still present (should be 0 rows):")
if still_old:
    for code, n in still_old:
        print(f"    {code}: {n:,} rows  ← PROBLEM")
else:
    print("    (none — old codes successfully replaced) ✓")

print("  Spot-check — new codes now present for those 10 groups:")
for code, n in now_new:
    print(f"    {code}: {n:,} rows")

# ── 6. Commit or roll back ────────────────────────────────────────────────────
if DRY_RUN:
    conn.rollback()
    print(f"\n=== DRY RUN complete — all changes rolled back. ===")
    print("    Re-run without --dry-run to apply for real.")
else:
    conn.commit()
    if remaining_after == 0:
        print("\nMigration complete. All LSOA codes now match lsoa_info. ✓")
    else:
        print(f"\nWARNING: {remaining_after} unrecognised codes remain.")

conn.close()
