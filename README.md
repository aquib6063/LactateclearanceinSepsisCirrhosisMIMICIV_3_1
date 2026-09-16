# LactateclearanceinSepsisCirrhosisMIMICIV_3_1

# Association between 48-Hour Lactate Clearance and Thirty-Day Mortality in Patients with Cirrhosis and Sepsis

This repository contains the analysis code used for the manuscript:

**Association between 48-Hour Lactate Clearance and Thirty-Day Mortality in Patients with Cirrhosis and Sepsis**

The study uses data from **MIMIC-IV version 3.1** to evaluate the association between lactate clearance at 12, 24, and 48 hours and 30-day mortality among critically ill adults with cirrhosis and sepsis.

## Repository Contents

### 1. `finalized_sql_landmark_corrected.sql`

SQL code used to construct the primary study cohort from MIMIC-IV v3.1, including:

- Identification of adult ICU admissions meeting Sepsis-3 criteria
- Identification of concurrent cirrhosis
- Selection of the first eligible ICU admission per patient
- Baseline lactate assessment
- Definition of baseline hyperlactatemia (lactate ≥2.0 mmol/L)
- 48-hour landmark cohort construction
- Extraction of baseline clinical and laboratory covariates
- Ascertainment of 30-day mortality and survival time

### 2. `phase2_corrected_analysis.py`

Python code used for the statistical analyses, including:

- Complete-case Cox proportional hazards models for 12-, 24-, and 48-hour lactate clearance
- Categorical and continuous lactate-clearance analyses
- Multiple imputation using chained equations (MICE)
- Missing-not-at-random sensitivity analyses
- Treatment-confounding sensitivity analyses
- Cirrhosis-etiology subgroup analyses
- Proportional-hazards and nonlinearity assessments
- Discrimination analyses using AUROC
- Incremental predictive-performance analyses
- Reclassification analyses using NRI and IDI
- Calibration analyses
- Decision-curve analysis
- Bootstrap optimism correction

## Primary Analysis

The primary association analysis uses a **48-hour landmark design**. Patients who died before the 48-hour/day-2 boundary were excluded from the landmark cohort, and subsequent mortality was analyzed from the landmark.

The primary exposure is **48-hour lactate clearance**, calculated from baseline and 48-hour lactate measurements.

## Reproducibility

The repository provides the SQL and Python analysis code used to generate the study cohort and statistical analyses. The analyses require access to the **MIMIC-IV v3.1 database** and appropriate PhysioNet credentialing and data-use permissions.

No patient-level MIMIC-IV data are included in this repository.

Researchers reproducing the analysis should use their own authorized MIMIC-IV v3.1 data and follow the MIMIC-IV data-use requirements.

## Data Availability

MIMIC-IV is a restricted-access critical care database available through PhysioNet following completion of the required credentialing and data-use procedures.

The authors do not redistribute MIMIC-IV patient-level data.

## Analysis Workflow

The analysis follows the workflow:

MIMIC-IV v3.1  
↓  
SQL cohort construction  
↓  
Hyperlactatemia cohort  
↓  
48-hour landmark cohort  
↓  
Complete-case and multiple-imputation analyses  
↓  
Cox regression and sensitivity analyses  
↓  
Predictive-performance and clinical-utility analyses

## Software

The statistical analyses were performed using Python and standard scientific/statistical computing libraries. SQL was used for cohort construction from MIMIC-IV.

## Manuscript

This repository accompanies the manuscript submitted to the:

**World Journal of Critical Care Medicine**

Manuscript No.: **126511**

## Citation

If you use or adapt this analysis code, please cite the associated manuscript and the original MIMIC-IV database publication.
