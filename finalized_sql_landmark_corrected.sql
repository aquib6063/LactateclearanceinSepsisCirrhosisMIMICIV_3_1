-- Phase 1a: Identify Cirrhosis Patients using ICD-9 and ICD-10 codes
WITH cirrhosis_dx AS (
    SELECT DISTINCT subject_id, hadm_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE
        (icd_version = 9 AND icd_code IN ('5712', '5715', '5716'))
        OR
        (icd_version = 10 AND icd_code IN ('K7030', 'K7031', 'K743', 'K744', 'K745', 'K7460', 'K7469'))
),

-- Phase 1b: Flag specific Hepatic/GI Comorbidities and Decompensations
comorbidities AS (
    SELECT
        subject_id,
        hadm_id,
        MAX(CASE WHEN (icd_version = 9 AND icd_code = '1550') OR (icd_version = 10 AND icd_code = 'C220') THEN 1 ELSE 0 END) AS hcc_flag,
        MAX(CASE WHEN (icd_version = 9 AND icd_code IN ('4560', '45620')) OR (icd_version = 10 AND icd_code IN ('I8501', 'I8511')) THEN 1 ELSE 0 END) AS variceal_bleed_flag,
        MAX(CASE WHEN (icd_version = 9 AND icd_code = '56723') OR (icd_version = 10 AND icd_code = 'K652') THEN 1 ELSE 0 END) AS sbp_flag,
        MAX(CASE WHEN (icd_version = 9 AND icd_code = '5722') OR (icd_version = 10 AND icd_code LIKE 'K72%') THEN 1 ELSE 0 END) AS he_flag,
        MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '7895%') OR (icd_version = 10 AND icd_code LIKE 'R18%') THEN 1 ELSE 0 END) AS ascites_flag,
        MAX(CASE WHEN (icd_version = 9 AND icd_code = '5724') OR (icd_version = 10 AND icd_code = 'K767') THEN 1 ELSE 0 END) AS hrs_flag
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    GROUP BY subject_id, hadm_id
),

-- Cirrhosis etiology classification by admission
cirrhosis_etiology AS (
    SELECT
        subject_id,
        hadm_id,
        CASE
            WHEN MAX(CASE WHEN (icd_version = 9 AND icd_code = '5712') OR (icd_version = 10 AND icd_code = 'K7030') THEN 1 ELSE 0 END) = 1 THEN 'alcohol_associated'
            WHEN MAX(CASE WHEN (icd_version = 10 AND icd_code IN ('B18.1', 'B18.2', 'B182', 'B181')) OR (icd_version = 9 AND icd_code IN ('0705', '0707')) THEN 1 ELSE 0 END) = 1 THEN 'viral'
            WHEN MAX(CASE WHEN (icd_version = 10 AND icd_code IN ('K743', 'K744', 'K745')) OR (icd_version = 9 AND icd_code = '5716') THEN 1 ELSE 0 END) = 1 THEN 'cholestatic_biliary'
            ELSE 'other_cryptogenic'
        END AS cirrhosis_etiology
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    GROUP BY subject_id, hadm_id
),

-- Phase 2: Define Sepsis + Cirrhosis Cohort (one index stay per patient: earliest eligible ICU stay)
sepsis_cirrhosis_cohort_all AS (
    SELECT
        s.subject_id,
        s.stay_id,
        s.suspected_infection_time,
        i.hadm_id,
        i.intime,
        ROW_NUMBER() OVER (PARTITION BY s.subject_id ORDER BY s.suspected_infection_time, i.intime, i.stay_id) AS stay_rank
    FROM `physionet-data.mimiciv_3_1_derived.sepsis3` s
    JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
        ON s.stay_id = i.stay_id
    JOIN cirrhosis_dx c
        ON i.hadm_id = c.hadm_id
),
sepsis_cirrhosis_cohort AS (
    SELECT subject_id, stay_id, suspected_infection_time, hadm_id, intime
    FROM sepsis_cirrhosis_cohort_all
    WHERE stay_rank = 1  -- one index episode per patient (first eligible ICU stay)
),

-- Phase 3: Gather Demographics, Mortality Flags, and 30/365 Survival Days
patient_info AS (
    SELECT
        c.subject_id,
        c.stay_id,
        c.hadm_id,
        c.suspected_infection_time,
        p.gender,
        p.anchor_age + (EXTRACT(YEAR FROM c.suspected_infection_time) - p.anchor_year) AS age,
        a.race,
        p.dod,

        CASE
            WHEN p.dod IS NOT NULL
             AND DATETIME_DIFF(CAST(p.dod AS DATETIME), CAST(c.suspected_infection_time AS DATETIME), DAY) <= 30
            THEN 1
            ELSE 0
        END AS mortality_30d,
        CASE
            WHEN p.dod IS NOT NULL
             AND DATETIME_DIFF(CAST(p.dod AS DATETIME), CAST(c.suspected_infection_time AS DATETIME), DAY) <= 30
            THEN DATETIME_DIFF(CAST(p.dod AS DATETIME), CAST(c.suspected_infection_time AS DATETIME), DAY)
            ELSE 30
        END AS survival_days_30,

        CASE
            WHEN p.dod IS NOT NULL
             AND DATETIME_DIFF(CAST(p.dod AS DATETIME), CAST(c.suspected_infection_time AS DATETIME), DAY) <= 365
            THEN 1
            ELSE 0
        END AS mortality_365d,
        CASE
            WHEN p.dod IS NOT NULL
             AND DATETIME_DIFF(CAST(p.dod AS DATETIME), CAST(c.suspected_infection_time AS DATETIME), DAY) <= 365
            THEN DATETIME_DIFF(CAST(p.dod AS DATETIME), CAST(c.suspected_infection_time AS DATETIME), DAY)
            ELSE 365
        END AS survival_days_365

    FROM sepsis_cirrhosis_cohort c
    JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
        ON c.subject_id = p.subject_id
    JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
        ON c.hadm_id = a.hadm_id
),

-- Phase 3b: Landmark timestamp and landmark-relative survival/censoring
landmark_survival AS (
    SELECT
        c.stay_id,
        DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR) AS landmark_48h,
        CASE
            WHEN p.dod IS NOT NULL
             AND CAST(p.dod AS DATETIME) < DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR)
            THEN NULL
            WHEN p.dod IS NOT NULL
             AND CAST(p.dod AS DATETIME) >= DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR)
             AND DATETIME_DIFF(CAST(p.dod AS DATETIME), DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR), MINUTE) / 1440.0 <= 28.0
            THEN 1 ELSE 0
        END AS event_30d_after_landmark,
        CASE
            WHEN p.dod IS NOT NULL
             AND CAST(p.dod AS DATETIME) < DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR)
            THEN NULL
            WHEN p.dod IS NOT NULL
             AND CAST(p.dod AS DATETIME) >= DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR)
             AND DATETIME_DIFF(CAST(p.dod AS DATETIME), DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR), MINUTE) / 1440.0 <= 28.0
            THEN DATETIME_DIFF(CAST(p.dod AS DATETIME), DATETIME_ADD(CAST(c.suspected_infection_time AS DATETIME), INTERVAL 48 HOUR), MINUTE) / 1440.0
            ELSE 28.0
        END AS time_from_landmark_to_censor_or_death
    FROM sepsis_cirrhosis_cohort c
    JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
        ON c.subject_id = p.subject_id
),

-- Phase 4: Extract All Lactate Values
all_lactate AS (
    SELECT
        c.stay_id,
        c.suspected_infection_time,
        le.charttime,
        le.valuenum AS lactate_value,
        DATETIME_DIFF(CAST(le.charttime AS DATETIME), CAST(c.suspected_infection_time AS DATETIME), MINUTE) / 60.0 AS hours_from_sepsis
    FROM sepsis_cirrhosis_cohort c
    JOIN `physionet-data.mimiciv_3_1_hosp.labevents` le
        ON c.subject_id = le.subject_id
       AND c.hadm_id = le.hadm_id
    WHERE le.itemid IN (50813, 52442)
      AND le.valuenum IS NOT NULL
),

-- Phase 5: Isolate Target Time Windows
lactate_t0 AS (
    SELECT stay_id, lactate_value AS lactate_0, charttime AS lactate_0_time,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY ABS(hours_from_sepsis)) AS rn
    FROM all_lactate WHERE hours_from_sepsis BETWEEN -6 AND 6
),
lactate_t12 AS (
    SELECT stay_id, lactate_value AS lactate_12, charttime AS lactate_12_time,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY ABS(hours_from_sepsis - 12)) AS rn
    FROM all_lactate WHERE hours_from_sepsis BETWEEN 6 AND 18
),
lactate_t24 AS (
    SELECT stay_id, lactate_value AS lactate_24, charttime AS lactate_24_time,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY ABS(hours_from_sepsis - 24)) AS rn
    FROM all_lactate WHERE hours_from_sepsis BETWEEN 18 AND 30
),
lactate_t48 AS (
    SELECT stay_id, lactate_value AS lactate_48, charttime AS lactate_48_time,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY ABS(hours_from_sepsis - 48)) AS rn
    FROM all_lactate WHERE hours_from_sepsis BETWEEN 36 AND 60
),

-- QC: number of lactate measurements available in each target window
lactate_window_counts AS (
    SELECT
        stay_id,
        COUNTIF(hours_from_sepsis BETWEEN 6 AND 18) AS n_lactate_12h_window,
        COUNTIF(hours_from_sepsis BETWEEN 18 AND 30) AS n_lactate_24h_window,
        COUNTIF(hours_from_sepsis BETWEEN 36 AND 60) AS n_lactate_48h_window
    FROM all_lactate
    GROUP BY stay_id
),

-- Phase 5b: Vasopressor use strictly within first 24h post-sepsis-onset
vasopressor_flag AS (
    SELECT DISTINCT va.stay_id, 1 AS vasopressor_within_24h
    FROM `physionet-data.mimiciv_3_1_derived.vasoactive_agent` va
    JOIN sepsis_cirrhosis_cohort s ON va.stay_id = s.stay_id
    WHERE va.starttime >= CAST(s.suspected_infection_time AS DATETIME)
      AND va.starttime <= DATETIME_ADD(CAST(s.suspected_infection_time AS DATETIME), INTERVAL 24 HOUR)
),

-- Phase 5c: Invasive mechanical ventilation strictly within first 24h post-sepsis-onset
mech_vent_flag AS (
    SELECT DISTINCT v.stay_id, 1 AS mech_vent_within_24h
    FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
    JOIN sepsis_cirrhosis_cohort s ON v.stay_id = s.stay_id
    WHERE v.ventilation_status = 'InvasiveVent'
      AND v.starttime >= CAST(s.suspected_infection_time AS DATETIME)
      AND v.starttime <= DATETIME_ADD(CAST(s.suspected_infection_time AS DATETIME), INTERVAL 24 HOUR)
),

-- Phase 5d: Hours from suspected infection to first antibiotic (continuous, unchanged timing logic)
antibiotic_timing AS (
    SELECT
        s.stay_id,
        MIN(DATETIME_DIFF(CAST(a.starttime AS DATETIME), CAST(s.suspected_infection_time AS DATETIME), MINUTE)) / 60.0 AS hours_to_antibiotic
    FROM sepsis_cirrhosis_cohort s
    JOIN `physionet-data.mimiciv_3_1_derived.antibiotic` a ON s.hadm_id = a.hadm_id
    GROUP BY s.stay_id
),

-- Phase 5e: CRRT strictly within first 24h post-sepsis-onset
crrt_24h_flag AS (
    SELECT DISTINCT cr.stay_id, 1 AS crrt_within_24h
    FROM `physionet-data.mimiciv_3_1_derived.crrt` cr
    JOIN sepsis_cirrhosis_cohort s ON cr.stay_id = s.stay_id
    WHERE cr.charttime >= CAST(s.suspected_infection_time AS DATETIME)
      AND cr.charttime <= DATETIME_ADD(CAST(s.suspected_infection_time AS DATETIME), INTERVAL 24 HOUR)
)
-- fluid_24h intentionally omitted: itemid list unverified against d_items.
-- Treatment-intensity proxy for now is vasopressor/ventilation/CRRT only.

-- Phase 6: Final Aggregation
SELECT
    p.subject_id,
    p.stay_id,
    p.hadm_id,
    p.age,
    p.gender AS sex,
    p.race,
    p.suspected_infection_time,

    p.mortality_30d,
    p.survival_days_30,
    p.mortality_365d,
    p.survival_days_365,

    lm.landmark_48h,
    lm.event_30d_after_landmark,
    lm.time_from_landmark_to_censor_or_death,

    etio.cirrhosis_etiology,

    COALESCE(como.hcc_flag, 0) AS hcc_flag,
    COALESCE(como.variceal_bleed_flag, 0) AS variceal_bleed_flag,
    COALESCE(como.sbp_flag, 0) AS sbp_flag,
    COALESCE(como.he_flag, 0) AS he_flag,
    COALESCE(como.ascites_flag, 0) AS ascites_flag,
    COALESCE(como.hrs_flag, 0) AS hrs_flag,
    charlson.charlson_comorbidity_index AS cci_score,

    l0.lactate_0,
    l0.lactate_0_time,
    l12.lactate_12,
    l12.lactate_12_time,
    l24.lactate_24,
    l24.lactate_24_time,
    l48.lactate_48,
    l48.lactate_48_time,
    CASE WHEN l0.lactate_0 IS NOT NULL AND l0.lactate_0 >= 2.0 THEN 1 ELSE 0 END AS baseline_lactate_ge2,
    CASE WHEN l48.lactate_48 IS NOT NULL THEN 1 ELSE 0 END AS lactate48_observed,
    COALESCE(lwc.n_lactate_12h_window, 0) AS n_lactate_12h_window,
    COALESCE(lwc.n_lactate_24h_window, 0) AS n_lactate_24h_window,
    COALESCE(lwc.n_lactate_48h_window, 0) AS n_lactate_48h_window,
    CASE WHEN l0.lactate_0 > 0 THEN ((l0.lactate_0 - l12.lactate_12) / l0.lactate_0) * 100 END AS clearance_12h,
    CASE WHEN l0.lactate_0 > 0 THEN ((l0.lactate_0 - l24.lactate_24) / l0.lactate_0) * 100 END AS clearance_24h,
    CASE WHEN l0.lactate_0 > 0 THEN ((l0.lactate_0 - l48.lactate_48) / l0.lactate_0) * 100 END AS clearance_48h,

    sofa.sofa AS day1_sofa,
    saps.sapsii,

    labs.albumin_min AS day1_albumin,
    labs.bilirubin_total_max AS day1_total_bilirubin,
    labs.creatinine_max AS day1_creatinine,
    labs.inr_max AS day1_inr,
    labs.sodium_min AS day1_sodium,
    gcs.gcs_min AS day1_min_gcs,

    COALESCE(crrt24.crrt_within_24h, 0) AS crrt_within_24h,
    COALESCE(vaso.vasopressor_within_24h, 0) AS vasopressor_within_24h,
    COALESCE(vent.mech_vent_within_24h, 0) AS mech_vent_within_24h,
    abx.hours_to_antibiotic

FROM patient_info p
LEFT JOIN landmark_survival lm ON p.stay_id = lm.stay_id
LEFT JOIN cirrhosis_etiology etio ON p.hadm_id = etio.hadm_id
LEFT JOIN comorbidities como ON p.hadm_id = como.hadm_id
LEFT JOIN lactate_t0 l0 ON p.stay_id = l0.stay_id AND l0.rn = 1
LEFT JOIN lactate_t12 l12 ON p.stay_id = l12.stay_id AND l12.rn = 1
LEFT JOIN lactate_t24 l24 ON p.stay_id = l24.stay_id AND l24.rn = 1
LEFT JOIN lactate_t48 l48 ON p.stay_id = l48.stay_id AND l48.rn = 1
LEFT JOIN lactate_window_counts lwc ON p.stay_id = lwc.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_sofa` sofa ON p.stay_id = sofa.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.sapsii` saps ON p.stay_id = saps.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.charlson` charlson ON p.hadm_id = charlson.hadm_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_lab` labs ON p.stay_id = labs.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_gcs` gcs ON p.stay_id = gcs.stay_id
LEFT JOIN crrt_24h_flag crrt24 ON p.stay_id = crrt24.stay_id
LEFT JOIN vasopressor_flag vaso ON p.stay_id = vaso.stay_id
LEFT JOIN mech_vent_flag vent ON p.stay_id = vent.stay_id
LEFT JOIN antibiotic_timing abx ON p.stay_id = abx.stay_id;
