# Lactate Clearance and 30-Day Mortality in Cirrhosis with Sepsis (MIMIC-IV v3.1)

Code for the study **"Association between 48-Hour Lactate Clearance and Thirty-Day Mortality Hazard in Patients with Cirrhosis and Sepsis"** (World Journal of Critical Care Medicine, manuscript no. 126511).

The study uses MIMIC-IV v3.1 to evaluate whether lactate clearance at 12, 24 and 48 hours after sepsis onset is associated with 30-day mortality in adult ICU patients with cirrhosis and Sepsis-3, using a 48-hour landmark design.

## Files

| File | What it does |
| --- | --- |
| `01_cohort_extraction_mimiciv.sql` | Builds the cohort in Google BigQuery: Sepsis-3 + cirrhosis (ICD-9/10), first eligible ICU stay per patient, lactate at 0/12/24/48 h, covariates, 30-day outcomes and 48-hour landmark variables. |
| `02_lactate_clearance_main_analysis.py` | Main analysis: cohort flow, complete-case landmark Cox models (continuous and categorical clearance), diagnostics, sensitivity analyses, multiple imputation (MICE) and MNAR delta adjustment, discrimination, NRI/IDI, calibration and decision curves. |
| `03_lactate_clearance_sensitivity_rcs_reduced.py` | Additional sensitivity analyses: reduced 7-parameter Cox model (12/24/48 h) and restricted cubic spline for 48-hour clearance. Imports functions from `02_...` and rebuilds the identical cohorts. |
| `requirements.txt` | Python package versions used for the published results. |

## Data access

MIMIC-IV is a credentialed-access database. Patient-level data **cannot be shared** in this repository.
To reproduce the analysis you need your own PhysioNet credentialed access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) (including completion of the required CITI training) and access to the MIMIC-IV BigQuery datasets (`physionet-data.mimiciv_3_1_*`).

## How to run

1. **Extract the cohort.** Run `01_cohort_extraction_mimiciv.sql` in Google BigQuery. Export the result as CSV and save it as `data/cohort_extract.csv`.
2. **Set up Python** (3.13 was used):
   ```
   pip install -r requirements.txt
   ```
3. **Run the main analysis:**
   ```
   python 02_lactate_clearance_main_analysis.py
   ```
   Results are written to `outputs/main/`.
4. **Run the sensitivity analyses:**
   ```
   python 03_lactate_clearance_sensitivity_rcs_reduced.py
   ```
   Results are written to `outputs/sensitivity/`. A plain-text summary of key results is saved as `Round3_Summary.txt`.

Both scripts stop with an error if the cohort sizes differ from the published cohort (730 with baseline hyperlactatemia, 665 in the 48-hour landmark cohort; 413 / 334 / 281 complete cases at 12 / 24 / 48 hours). This guards against a changed extraction or data version.

A different CSV location can be given with the `DATA_PATH` environment variable.

## Reproducibility

The Cox models, spline knots and cohort construction are deterministic. Multiple imputation, cross-validation and bootstrap steps use a fixed random seed (42). With the same data and package versions, the scripts reproduce the published numbers exactly.

## Contact

Corresponding author: Aalam Sohal, MD — Division of Gastroenterology and Hepatology, Creighton University School of Medicine, Phoenix, AZ, USA.
