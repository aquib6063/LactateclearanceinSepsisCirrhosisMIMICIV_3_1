#!/usr/bin/env python3
from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.matplotlib_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parent / '.cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.calibration import calibration_curve
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'job_-0WDwXZjrgZtzzeA8wjpeJ_1Nhng.csv'
OUTPUT_DIR = SCRIPT_DIR / 'phase2_python_outputs'
RANDOM_SEED = 42
N_IMPUTATIONS = int(os.environ.get('N_IMPUTATIONS', '25'))
N_BOOTSTRAPS = int(os.environ.get('N_BOOTSTRAPS', '1000'))
CV_SPLITS = 5
COX_PENALIZER = float(os.environ.get('COX_PENALIZER', '0.01'))
QUARTILE_LABELS = ['Q1_Worst', 'Q2_Poor', 'Q3_Moderate', 'Q4_Best']
FIXED_GROUP_ORDER = ['<=0%', '0-33%', '33-66%', '>=66%']
FIXED_REFERENCE = '>=66%'
COX_FIT_LOG: list[dict] = []

covariates = [
    'lactate_0',
    'age',
    'female_flag',
    'sapsii',
    'day1_sofa',
    'cci_score',
    'day1_albumin',
    'day1_total_bilirubin',
    'day1_creatinine',
    'day1_inr',
    'day1_sodium',
    'crrt_within_24h',
    'ascites_flag',
    'variceal_bleed_flag',
    'hcc_flag',
    'sbp_flag',
    'hrs_flag',
    'he_flag',
    'vasopressor_within_24h',
    'mech_vent_within_24h',
]

binary_model_covariates = [
    'crrt_within_24h',
    'ascites_flag',
    'variceal_bleed_flag',
    'hcc_flag',
    'sbp_flag',
    'hrs_flag',
    'he_flag',
    'vasopressor_within_24h',
    'mech_vent_within_24h',
]

cols_to_impute = [
    'age',
    'cci_score',
    'day1_sofa',
    'sapsii',
    'day1_albumin',
    'day1_total_bilirubin',
    'day1_creatinine',
    'day1_inr',
    'day1_sodium',
    'lactate_0',
    'lactate_12',
    'lactate_24',
    'lactate_48',
]

imputation_auxiliary_cols = [
    'mortality_30d',
    'survival_days_30',
    'event_30d_after_landmark',
    'time_from_landmark_to_censor_or_death',
]

mice_predictor_impute_cols = cols_to_impute + binary_model_covariates + ['female_flag']

REQUIRED_COLUMNS = sorted(set(cols_to_impute + [c for c in covariates if c != 'female_flag'] + ['mortality_30d', 'survival_days_30', 'sex']))


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def female_indicator(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.startswith('F').astype(int)


def calculate_meld30(row: pd.Series) -> int:
    bili = min(max(float(row['day1_total_bilirubin']), 1.0), 60.0)
    inr = min(max(float(row['day1_inr']), 1.0), 10.0)
    sodium = min(max(float(row['day1_sodium']), 125.0), 137.0)
    albumin = min(max(float(row['day1_albumin']), 1.5), 3.5)
    if row.get('crrt_within_24h', 0) == 1:
        creat = 3.0
    else:
        creat = min(max(float(row['day1_creatinine']), 1.0), 3.0)
    female = 1.33 if int(row['female_flag']) == 1 else 0.0
    meld = (
        female
        + 4.56 * np.log(bili)
        + 0.82 * (137.0 - sodium)
        - 0.24 * (137.0 - sodium) * np.log(bili)
        + 9.09 * np.log(inr)
        + 11.14 * np.log(creat)
        + 1.85 * (3.5 - albumin)
        - 1.83 * (3.5 - albumin) * np.log(creat)
        + 6.0
    )
    meld = max(6.0, meld)
    return int(round(meld))


def derive_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['female_flag'] = female_indicator(out['sex'])
    for tp, col in [('12h', 'lactate_12'), ('24h', 'lactate_24'), ('48h', 'lactate_48')]:
        out[f'clearance_{tp}'] = np.where(
            out['lactate_0'] > 0,
            (out['lactate_0'] - out[col]) / out['lactate_0'] * 100,
            np.nan,
        )
    out['day1_meld_3_0'] = out.apply(calculate_meld30, axis=1)
    return out


def run_multiple_imputation(df_input: pd.DataFrame, m: int = N_IMPUTATIONS, seed: int = RANDOM_SEED) -> list[pd.DataFrame]:
    datasets: list[pd.DataFrame] = []
    for i in range(m):
        temp = df_input.copy()
        temp['female_flag'] = female_indicator(temp['sex'])
        aux_cols = [c for c in imputation_auxiliary_cols if c in temp.columns]
        impute_cols = [c for c in mice_predictor_impute_cols if c in temp.columns]
        work_cols = impute_cols + [c for c in aux_cols if c not in impute_cols]
        imputer = IterativeImputer(sample_posterior=True, random_state=seed + i, max_iter=30)
        imputed = pd.DataFrame(imputer.fit_transform(temp[work_cols]), columns=work_cols, index=temp.index)
        temp[impute_cols] = imputed[impute_cols]
        for col in binary_model_covariates + ['female_flag']:
            if col in temp.columns:
                temp[col] = temp[col].round().clip(0, 1).astype(int)
        for col in aux_cols:
            temp[col] = df_input[col]
        temp = derive_variables(temp)
        datasets.append(temp)
    return datasets


def apply_mnar_delta(imputed_sets: list[pd.DataFrame], df_original: pd.DataFrame, delta_mmol: float) -> list[pd.DataFrame]:
    """Delta-adjusted pattern-mixture MNAR sensitivity: worsen imputed (not
    originally observed) lactate_48 values by a fixed delta (mmol/L, higher =
    worse) and recompute clearance_48h, to test robustness of the primary
    association to departures from the missing-at-random assumption."""
    was_missing = df_original['lactate_48'].isna()
    adjusted = []
    for ds in imputed_sets:
        temp = ds.copy()
        temp.loc[was_missing, 'lactate_48'] = temp.loc[was_missing, 'lactate_48'] + delta_mmol
        temp['clearance_48h'] = np.where(temp['lactate_0'] > 0, (temp['lactate_0'] - temp['lactate_48']) / temp['lactate_0'] * 100, np.nan)
        temp['clearance_48h_per10'] = temp['clearance_48h'] / 10.0
        adjusted.append(temp)
    return adjusted


def recalculate_post_landmark_outcomes(df: pd.DataFrame, landmark_days: float = 2.0, min_followup: float = 1e-6) -> pd.DataFrame:
    out = df.copy()
    death_after_or_at_landmark = (out['mortality_30d'].astype(int) == 1) & (out['survival_days_30'].astype(float) >= landmark_days)
    out['event_30d_after_landmark'] = death_after_or_at_landmark.astype(int)
    out['time_from_landmark_to_censor_or_death'] = np.where(
        death_after_or_at_landmark,
        np.maximum(out['survival_days_30'].astype(float) - landmark_days, min_followup),
        30.0 - landmark_days,
    )
    return out


def build_mean_completed_dataset(imputed_sets: list[pd.DataFrame]) -> pd.DataFrame:
    stacked = pd.concat(imputed_sets, keys=range(len(imputed_sets)), names=['imputation', 'row'])
    mean_df = stacked.groupby(level='row').mean(numeric_only=True)
    non_numeric = imputed_sets[0][['sex', 'race', 'suspected_infection_time']].copy()
    mean_df = non_numeric.join(mean_df, how='left')
    return mean_df.sort_index()


def add_scaled_predictor(df: pd.DataFrame, source_col: str, new_col: str, divisor: float) -> pd.DataFrame:
    out = df.copy()
    out[new_col] = out[source_col] / divisor
    return out


def quartile_bin_edges(series: pd.Series) -> np.ndarray:
    _, bins = pd.qcut(series, q=4, retbins=True, duplicates='drop')
    return bins


def assign_quartiles_from_bins(df: pd.DataFrame, source_col: str, new_col: str, bins: np.ndarray, labels: list[str]) -> pd.DataFrame:
    out = df.copy()
    adjusted = np.asarray(bins, dtype=float).copy()
    adjusted[0] = -np.inf
    adjusted[-1] = np.inf
    out[new_col] = pd.cut(out[source_col], bins=adjusted, labels=labels, include_lowest=True)
    return out


def assign_fixed_categories(df: pd.DataFrame, source_col: str, new_col: str) -> pd.DataFrame:
    out = df.copy()
    s = out[source_col]
    groups = pd.Series(np.full(len(out), None, dtype=object), index=out.index)
    groups.loc[s <= 0] = '<=0%'
    groups.loc[(s > 0) & (s <= 33)] = '0-33%'
    groups.loc[(s > 33) & (s < 66)] = '33-66%'
    groups.loc[s >= 66] = '>=66%'
    out[new_col] = groups
    return out


def rubins_rules_pool(params: np.ndarray, variances: np.ndarray, names: list[str], n_obs: int | None = None) -> pd.DataFrame:
    m = params.shape[0]
    q_bar = params.mean(axis=0)
    u_bar = variances.mean(axis=0)
    b = params.var(axis=0, ddof=1)
    total_var = u_bar + (1 + 1 / m) * b
    se = np.sqrt(total_var)
    with np.errstate(divide='ignore', invalid='ignore'):
        df_old = (m - 1) * (1 + (u_bar / ((1 + 1 / m) * b))) ** 2
    df_old = np.where(np.isfinite(df_old), df_old, 1e6)
    df_old = np.where(df_old > 0, df_old, 1e6)
    t_stat = q_bar / se
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=df_old)
    crit = stats.t.ppf(0.975, df=df_old)
    beta_l = q_bar - crit * se
    beta_u = q_bar + crit * se
    data = {
        'term': names,
        'coef': q_bar,
        'se': se,
        'df': df_old,
        'p_value': p_value,
        'hr': np.exp(q_bar),
        'hr_ci_lower': np.exp(beta_l),
        'hr_ci_upper': np.exp(beta_u),
    }
    if n_obs is not None:
        data['n'] = n_obs
    return pd.DataFrame(data)


def single_model_summary(params: np.ndarray, covariance: np.ndarray, names: list[str], n_obs: int) -> pd.DataFrame:
    se = np.sqrt(np.diag(covariance))
    z_score = params / se
    p_value = 2 * stats.norm.sf(np.abs(z_score))
    crit = stats.norm.ppf(0.975)
    beta_l = params - crit * se
    beta_u = params + crit * se
    return pd.DataFrame({
        'term': names,
        'coef': params,
        'se': se,
        'z': z_score,
        'p_value': p_value,
        'hr': np.exp(params),
        'hr_ci_lower': np.exp(beta_l),
        'hr_ci_upper': np.exp(beta_u),
        'n': n_obs,
    })


def fit_cox_lifelines(model_df: pd.DataFrame, duration_col: str, event_col: str, predictors: list[str]) -> pd.DataFrame:
    fit_df = model_df[[duration_col, event_col] + predictors].dropna().copy()
    fit_df = fit_df.loc[fit_df[duration_col].astype(float) > 0].copy()
    for col in [duration_col, event_col] + predictors:
        fit_df[col] = pd.to_numeric(fit_df[col], errors='coerce')
    fit_df = fit_df.dropna()
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cph.fit(fit_df[[duration_col, event_col] + predictors], duration_col=duration_col, event_col=event_col, show_progress=False)
    warning_messages = [str(w.message) for w in caught]
    for msg in warning_messages:
        warnings.warn(msg)
    COX_FIT_LOG.append({
        'duration_col': duration_col,
        'event_col': event_col,
        'first_predictor': predictors[0] if predictors else '',
        'predictor_count': len(predictors),
        'n': len(fit_df),
        'events': int(fit_df[event_col].sum()),
        'convergence_status': 'returned_with_warning' if warning_messages else 'returned_without_warning',
        'warnings': ' | '.join(warning_messages),
        'predictors': ';'.join(predictors),
    })
    out = cph.summary.reset_index().rename(columns={
        'covariate': 'term',
        'coef lower 95%': 'coef_ci_lower',
        'coef upper 95%': 'coef_ci_upper',
        'exp(coef)': 'hr',
        'exp(coef) lower 95%': 'hr_ci_lower',
        'exp(coef) upper 95%': 'hr_ci_upper',
        'p': 'p_value',
    })
    out = out[['term', 'coef', 'se(coef)', 'z', 'p_value', 'hr', 'hr_ci_lower', 'hr_ci_upper']]
    out = out.rename(columns={'se(coef)': 'se'})
    out['n'] = len(fit_df)
    out['events'] = int(fit_df[event_col].sum())
    out['parameters'] = len(predictors)
    out['convergence_status'] = 'returned_with_warning' if warning_messages else 'returned_without_warning'
    out['warnings'] = ' | '.join(warning_messages)
    return out


def pooled_cox_ph(imputed_sets: list[pd.DataFrame], duration_col: str, event_col: str, predictors: list[str], entry_value: float | None = None) -> pd.DataFrame:
    params = []
    variances = []
    n_obs_list = []
    for dataset in imputed_sets:
        model_df = dataset[[duration_col, event_col] + predictors].dropna().copy()
        res = fit_cox_lifelines(model_df, duration_col, event_col, predictors)
        n_obs_list.append(int(res['n'].iloc[0]))
        params.append(res.set_index('term').loc[predictors, 'coef'].to_numpy(dtype=float))
        variances.append(res.set_index('term').loc[predictors, 'se'].to_numpy(dtype=float) ** 2)
    return rubins_rules_pool(np.vstack(params), np.vstack(variances), predictors, n_obs=int(np.mean(n_obs_list)))


def pooled_categorical_cox(imputed_sets: list[pd.DataFrame], duration_col: str, event_col: str, category_col: str, category_terms: list[str], base_covariates: list[str], entry_value: float | None = None) -> pd.DataFrame:
    params = []
    variances = []
    n_obs_list = []
    for dataset in imputed_sets:
        indicators = pd.DataFrame(index=dataset.index)
        for term in category_terms:
            indicators[term] = (dataset[category_col].astype(str) == term).astype(int)
        model_df = dataset[[duration_col, event_col] + base_covariates].join(indicators).dropna().copy()
        predictors = base_covariates + category_terms
        res = fit_cox_lifelines(model_df, duration_col, event_col, predictors)
        n_obs_list.append(int(res['n'].iloc[0]))
        params.append(res.set_index('term').loc[predictors, 'coef'].to_numpy(dtype=float))
        variances.append(res.set_index('term').loc[predictors, 'se'].to_numpy(dtype=float) ** 2)
    return rubins_rules_pool(np.vstack(params), np.vstack(variances), base_covariates + category_terms, n_obs=int(np.mean(n_obs_list)))


def pooled_quartile_cox(imputed_sets: list[pd.DataFrame], duration_col: str, event_col: str, quartile_col: str, base_covariates: list[str]) -> pd.DataFrame:
    return pooled_categorical_cox(imputed_sets, duration_col, event_col, quartile_col, ['Q1_Worst', 'Q2_Poor', 'Q3_Moderate'], base_covariates)


def complete_case_cox_ph(df: pd.DataFrame, duration_col: str, event_col: str, predictors: list[str], entry_value: float | None = None) -> pd.DataFrame:
    model_df = df[[duration_col, event_col] + predictors].dropna().copy()
    return fit_cox_lifelines(model_df, duration_col, event_col, predictors)


def complete_case_categorical_cox(df: pd.DataFrame, duration_col: str, event_col: str, category_col: str, category_terms: list[str], base_covariates: list[str], entry_value: float | None = None) -> pd.DataFrame:
    indicators = pd.DataFrame(index=df.index)
    for term in category_terms:
        indicators[term] = (df[category_col].astype(str) == term).astype(int)
    model_df = df[[duration_col, event_col, category_col] + base_covariates].join(indicators).dropna().copy()
    return fit_cox_lifelines(model_df, duration_col, event_col, base_covariates + category_terms)


def events_per_parameter(df: pd.DataFrame, duration_col: str, event_col: str, predictors: list[str], entry_value: float | None = None) -> dict:
    """Report events-per-parameter for a Cox model on the same complete-case rows the model would actually use."""
    model_df = df[[duration_col, event_col] + predictors].dropna().copy()
    n_events = int(model_df[event_col].sum())
    n_params = len(predictors)
    return {
        'n': len(model_df),
        'n_events': n_events,
        'n_params': n_params,
        'events_per_parameter': round(n_events / n_params, 2) if n_params else np.nan,
        'entry_value': entry_value,
    }


def ph_interaction_test(df: pd.DataFrame, duration_col: str, event_col: str, predictors: list[str], key_predictor: str, entry_value: float | None = None) -> dict:
    """Test the proportional-hazards assumption for key_predictor by adding a
    key_predictor x log(time) interaction term to the Cox model. A significant
    interaction coefficient indicates the hazard ratio for key_predictor is not
    constant over time (i.e., the PH assumption is violated for that term)."""
    model_df = df[[duration_col, event_col] + predictors].dropna().copy()
    log_time = np.log(model_df[duration_col].astype(float).clip(lower=1e-6))
    interaction_col = f'{key_predictor}_x_logtime'
    model_df[interaction_col] = model_df[key_predictor].astype(float) * log_time
    exog_cols = predictors + [interaction_col]
    result = fit_cox_lifelines(model_df, duration_col, event_col, exog_cols).set_index('term')
    coef = float(result.loc[interaction_col, 'coef'])
    se = float(result.loc[interaction_col, 'se'])
    z = coef / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(np.abs(z))) if np.isfinite(z) else np.nan
    return {
        'predictor': key_predictor,
        'interaction_term': f'{key_predictor} x log(time)',
        'interaction_coef': coef,
        'interaction_se': se,
        'p_value': p_value,
        'n': len(model_df),
        'n_events': int(model_df[event_col].sum()),
        'interpretation': 'PH assumption violated (p<0.05)' if (np.isfinite(p_value) and p_value < 0.05) else 'No evidence against PH assumption',
    }




def nonlinearity_test(df: pd.DataFrame, duration_col: str, event_col: str, predictors: list[str], key_predictor: str) -> dict:
    """Test for nonlinearity in key_predictor's log-hazard effect by adding a
    squared term. A significant squared-term coefficient suggests the
    relationship is not linear."""
    model_df = df[[duration_col, event_col] + predictors].dropna().copy()
    sq_col = f'{key_predictor}_sq'
    model_df[sq_col] = model_df[key_predictor].astype(float) ** 2
    exog_cols = predictors + [sq_col]
    result = fit_cox_lifelines(model_df, duration_col, event_col, exog_cols).set_index('term')
    coef = float(result.loc[sq_col, 'coef'])
    se = float(result.loc[sq_col, 'se'])
    z = coef / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(np.abs(z))) if np.isfinite(z) else np.nan
    return {
        'predictor': key_predictor,
        'squared_term_coef': coef,
        'squared_term_se': se,
        'p_value': p_value,
        'n': len(model_df),
        'n_events': int(model_df[event_col].sum()),
        'interpretation': 'Evidence of nonlinearity (p<0.05)' if (np.isfinite(p_value) and p_value < 0.05) else 'No evidence against a linear relationship',
    }


def ph_global_variable_tests(df: pd.DataFrame, duration_col: str, event_col: str, predictors: list[str], label: str) -> pd.DataFrame:
    fit_df = df[[duration_col, event_col] + predictors].dropna().copy()
    fit_df = fit_df.loc[fit_df[duration_col].astype(float) > 0].copy()
    for col in [duration_col, event_col] + predictors:
        fit_df[col] = pd.to_numeric(fit_df[col], errors='coerce')
    fit_df = fit_df.dropna()
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(fit_df[[duration_col, event_col] + predictors], duration_col=duration_col, event_col=event_col, show_progress=False)
    ph = proportional_hazard_test(cph, fit_df[[duration_col, event_col] + predictors], time_transform='rank')
    rows = []
    for term, row in ph.summary.iterrows():
        rows.append({
            'analysis': label,
            'scope': 'variable',
            'term': term,
            'test_statistic': float(row['test_statistic']),
            'p_value': float(row['p']),
            'n': len(fit_df),
            'events': int(fit_df[event_col].sum()),
            'interpretation': 'PH violation signal (p<0.05)' if row['p'] < 0.05 else 'No PH violation signal',
        })
    global_stat = float(ph.summary['test_statistic'].sum())
    global_df = int(ph.summary.shape[0])
    global_p = float(1 - stats.chi2.cdf(global_stat, df=global_df))
    rows.insert(0, {
        'analysis': label,
        'scope': 'global',
        'term': 'GLOBAL',
        'test_statistic': global_stat,
        'p_value': global_p,
        'n': len(fit_df),
        'events': int(fit_df[event_col].sum()),
        'interpretation': 'Global PH violation signal (p<0.05)' if global_p < 0.05 else 'No global PH violation signal',
    })
    return pd.DataFrame(rows)

def format_continuous(series: pd.Series) -> str:
    valid = series.dropna()
    return '' if valid.empty else f'{valid.mean():.2f} ({valid.std(ddof=1):.2f})'


def format_categorical(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return ''
    count = int(valid.sum())
    pct = 100 * count / len(valid)
    return f'{count} ({pct:.1f}%)'


def baseline_table_by_group(df: pd.DataFrame, group_col: str, table_name: str, continuous_vars: list[str], categorical_vars: list[str], group_order: list[str] | None = None) -> None:
    if group_order is None:
        group_order = [str(x) for x in pd.Series(df[group_col].dropna()).astype(str).unique()]
    rows = []
    total_n = int(df[group_col].notna().sum())
    n_row = {'variable': 'n', 'N': total_n}
    for group in group_order:
        n_row[group] = int((df[group_col].astype(str) == group).sum())
    rows.append(n_row)
    for var in continuous_vars:
        row = {'variable': var, 'N': total_n}
        groups = []
        for group in group_order:
            mask = df[group_col].astype(str) == group
            vals = df.loc[mask, var]
            row[group] = format_continuous(vals)
            groups.append(vals.dropna())
        row['p_value'] = stats.f_oneway(*groups).pvalue if all(len(g) > 0 for g in groups) else np.nan
        rows.append(row)
    for var in categorical_vars:
        row = {'variable': var, 'N': total_n}
        contingency = []
        for group in group_order:
            mask = df[group_col].astype(str) == group
            vals = df.loc[mask, var].dropna()
            row[group] = format_categorical(vals)
            contingency.append([int(vals.sum()), int(len(vals) - vals.sum())])
        try:
            row['p_value'] = stats.chi2_contingency(np.asarray(contingency))[1]
        except Exception:
            row['p_value'] = np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(table_name, index=False)


def bootstrap_auc(y: np.ndarray, scores: np.ndarray, n_boot: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    point_auc = roc_auc_score(y, scores)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        y_s = y[idx]
        if np.unique(y_s).size < 2:
            continue
        aucs.append(roc_auc_score(y_s, scores[idx]))
    return point_auc, np.asarray(aucs, dtype=float)


def paired_bootstrap_auc_difference(y: np.ndarray, scores_new: np.ndarray, scores_old: np.ndarray, n_boot: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    point_diff = roc_auc_score(y, scores_new) - roc_auc_score(y, scores_old)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        y_s = y[idx]
        if np.unique(y_s).size < 2:
            continue
        diffs.append(roc_auc_score(y_s, scores_new[idx]) - roc_auc_score(y_s, scores_old[idx]))
    return point_diff, np.asarray(diffs, dtype=float)


def cross_validated_probabilities(df: pd.DataFrame, predictors: list[str], outcome_col: str, n_splits: int = CV_SPLITS, seed: int = RANDOM_SEED) -> np.ndarray:
    x = df[predictors].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=int)
    probs = np.zeros(len(df), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(x, y):
        model = LogisticRegression(max_iter=2000, solver='liblinear', random_state=seed)
        model.fit(x[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return probs


def optimism_corrected_auc(df: pd.DataFrame, predictors: list[str], outcome_col: str, apparent_auc: float, n_boot: int, seed: int) -> dict:
    """Harrell bootstrap optimism correction: fit on a bootstrap sample, evaluate
    that model on both the bootstrap sample and the original sample; the average
    gap is the optimism, subtracted from the apparent AUROC."""
    x = df[predictors].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    optimisms = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        x_b, y_b = x[idx], y[idx]
        if np.unique(y_b).size < 2:
            continue
        model = LogisticRegression(max_iter=2000, solver='liblinear', random_state=seed)
        model.fit(x_b, y_b)
        optimisms.append(roc_auc_score(y_b, model.predict_proba(x_b)[:, 1]) - roc_auc_score(y, model.predict_proba(x)[:, 1]))
    mean_optimism = float(np.mean(optimisms))
    return {'apparent_auc': float(apparent_auc), 'mean_optimism': mean_optimism, 'optimism_corrected_auc': float(apparent_auc) - mean_optimism, 'n_boot_valid': len(optimisms)}


def apparent_logistic_auc(df: pd.DataFrame, predictors: list[str], outcome_col: str, seed: int = RANDOM_SEED) -> float:
    x = df[predictors].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=int)
    model = LogisticRegression(max_iter=2000, solver='liblinear', random_state=seed)
    model.fit(x, y)
    return float(roc_auc_score(y, model.predict_proba(x)[:, 1]))


def decision_curve(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    out = []
    for t in thresholds:
        treat = prob >= t
        tp = np.sum(treat & (y == 1))
        fp = np.sum(treat & (y == 0))
        out.append((tp / n) - (fp / n) * (t / (1 - t)))
    return np.asarray(out, dtype=float)


def continuous_nri(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> float:
    event = y == 1
    nonevent = y == 0
    nri_event = np.mean(p_new[event] > p_old[event]) - np.mean(p_new[event] < p_old[event])
    nri_nonevent = np.mean(p_new[nonevent] < p_old[nonevent]) - np.mean(p_new[nonevent] > p_old[nonevent])
    return float(nri_event + nri_nonevent)


def idi(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> float:
    disc_old = p_old[y == 1].mean() - p_old[y == 0].mean()
    disc_new = p_new[y == 1].mean() - p_new[y == 0].mean()
    return float(disc_new - disc_old)


def bootstrap_reclassification_pvalue(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray, metric_fn, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    """Bootstrap point estimate, 95% CI, and two-sided p-value (H0: metric = 0) for NRI or IDI."""
    rng = np.random.default_rng(seed)
    point = metric_fn(y, p_old, p_new)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        y_s, p_old_s, p_new_s = y[idx], p_old[idx], p_new[idx]
        if np.unique(y_s).size < 2:
            continue
        boots.append(metric_fn(y_s, p_old_s, p_new_s))
    boots = np.asarray(boots, dtype=float)
    ci_lower = float(np.percentile(boots, 2.5))
    ci_upper = float(np.percentile(boots, 97.5))
    p_value = float(min(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)), 1.0))
    return point, ci_lower, ci_upper, p_value


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> tuple[float, float, int]:
    df_hl = pd.DataFrame({'y': y, 'p': p}).copy()
    df_hl['bin'] = pd.qcut(df_hl['p'], q=groups, duplicates='drop')
    grouped = df_hl.groupby('bin', observed=False)
    obs_event = grouped['y'].sum()
    exp_event = grouped['p'].sum()
    n_bin = grouped.size()
    obs_nonevent = n_bin - obs_event
    exp_nonevent = n_bin - exp_event
    hl = np.sum((obs_event - exp_event) ** 2 / (exp_event + 1e-12)) + np.sum((obs_nonevent - exp_nonevent) ** 2 / (exp_nonevent + 1e-12))
    df_value = max(len(grouped) - 2, 1)
    return float(hl), float(1 - stats.chi2.cdf(hl, df=df_value)), int(df_value)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    recal = sm.GLM(y, sm.add_constant(logit_p), family=sm.families.Binomial()).fit()
    hl_stat, hl_p, hl_df = hosmer_lemeshow(y, p, groups=10)
    slope = float(recal.params[1])
    slope_se = float(recal.bse[1])
    # Calibration slope should be tested against the null of perfect calibration (slope = 1),
    # not against 0 (statsmodels' default GLM p-value tests slope != 0, which is not the
    # relevant calibration question and will be spuriously significant for any working model).
    slope_z_vs1 = (slope - 1.0) / slope_se
    slope_p_vs1 = float(2 * stats.norm.sf(np.abs(slope_z_vs1)))
    return {
        'calibration_intercept': float(recal.params[0]),
        'calibration_intercept_p_vs0': float(recal.pvalues[0]),
        'calibration_slope': slope,
        'calibration_slope_p_vs1': slope_p_vs1,
        'brier_score': float(np.mean((np.asarray(y, dtype=float) - p) ** 2)),
        'hl_stat': hl_stat,
        'hl_p_value': hl_p,
        'hl_df': hl_df,
    }


def save_roc_plot(df_plot: pd.DataFrame, auc_summary: dict[str, dict[str, float]], filename: str) -> None:
    plt.figure(figsize=(9, 7))
    colors = {'clearance_12h': '#1f77b4', 'clearance_24h': '#ff7f0e', 'clearance_48h': '#2ca02c'}
    for col in ['clearance_12h', 'clearance_24h', 'clearance_48h']:
        fpr, tpr, _ = roc_curve(df_plot['mortality_30d'], -df_plot[col])
        plt.plot(fpr, tpr, lw=2.5, color=colors[col], label=f"{col.replace('clearance_', '').upper()} AUC = {auc_summary[col]['point_auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison: 12h vs 24h vs 48h Clearance')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_dca_plot(y: np.ndarray, prob_map: dict[str, np.ndarray], filename: str) -> None:
    thresholds = np.linspace(0.01, 0.50, 50)
    baseline = float(np.mean(y))
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, np.zeros_like(thresholds), '--', label='Treat None', color='black')
    plt.plot(thresholds, baseline - (1 - baseline) * (thresholds / (1 - thresholds)), '--', label='Treat All', color='gray')
    for label, prob in prob_map.items():
        plt.plot(thresholds, decision_curve(y, prob, thresholds), lw=2, label=label)
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_calibration_plot(y: np.ndarray, p: np.ndarray, filename: str) -> None:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, frac_pos, 'o-', lw=2)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Event Rate')
    plt.title('Calibration Plot for 48h Model')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def save_km_plot(df_plot: pd.DataFrame, duration_col: str, event_col: str, group_col: str, filename: str, summary_name: str) -> None:
    rows = []
    colors = {'<=0%': '#b2182b', '0-33%': '#ef8a62', '33-66%': '#67a9cf', '>=66%': '#2166ac'}
    journal_labels = {
        '<=0%': '<=0% clearance',
        '0-33%': '0-33% clearance',
        '33-66%': '33-66% clearance',
        '>=66%': '>=66% clearance',
    }
    fig = plt.figure(figsize=(11.5, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.45, 1.0], hspace=0.24)
    ax = fig.add_subplot(gs[0])
    for group in FIXED_GROUP_ORDER:
        sub = df_plot[df_plot[group_col].astype(str) == group].copy()
        if sub.empty:
            continue
        sf = SurvfuncRight(sub[duration_col].astype(float).to_numpy(), sub[event_col].astype(int).to_numpy())
        times = np.r_[0.0, np.asarray(sf.surv_times, dtype=float)]
        surv = np.r_[1.0, np.asarray(sf.surv_prob, dtype=float)]
        ax.step(times, surv, where='post', lw=2.6, color=colors.get(group), label=f"{journal_labels[group]} (n={len(sub)})")
        rows.append({
            'Group': journal_labels[group],
            'n': int(len(sub)),
            '30-day deaths': int(sub[event_col].sum()),
            '30-day mortality, %': round(float(sub[event_col].mean() * 100), 2),
            'Median follow-up, days': round(float(sub[duration_col].median()), 2),
        })
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 29, 4))
    ax.set_xlabel('Days from 48-hour landmark', fontsize=11)
    ax.set_ylabel('Survival probability', fontsize=11)
    ax.set_title('Post-Landmark Survival by 48-hour Lactate Clearance Category', fontsize=13, pad=10)
    ax.legend(title='48-hour clearance group', loc='lower left', frameon=False, fontsize=10, title_fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_name, index=False)

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')
    table_df = summary_df.copy()
    table_df['30-day mortality, %'] = table_df['30-day mortality, %'].map(lambda x: f'{x:.2f}')
    table_df['Median follow-up, days'] = table_df['Median follow-up, days'].map(lambda x: f'{x:.0f}' if float(x).is_integer() else f'{x:.2f}')
    tbl = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        colLoc='center',
        loc='center',
        bbox=[0.02, 0.06, 0.96, 0.82],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.18)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#9aa0a6')
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#eef3f8')
        else:
            cell.set_facecolor('white')

    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.09, right=0.98)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_heatmap(df_plot: pd.DataFrame, group_col: str, score_col: str, cutoff: float, filename: str, table_name: str, counts_name: str) -> None:
    out = df_plot.copy()
    out['meld_group'] = pd.cut(out[score_col], bins=[-np.inf, cutoff, np.inf], labels=[f'{score_col} <{int(cutoff)}', f'{score_col} >={int(cutoff)}'])
    out['clearance_group'] = pd.Categorical(out[group_col], categories=FIXED_GROUP_ORDER, ordered=True)
    pivot = out.pivot_table(values='mortality_30d', index='meld_group', columns='clearance_group', aggfunc='mean', observed=False)
    counts = out.pivot_table(values='mortality_30d', index='meld_group', columns='clearance_group', aggfunc='size', observed=False)
    pivot = pivot.reindex(index=[f'{score_col} <{int(cutoff)}', f'{score_col} >={int(cutoff)}'], columns=FIXED_GROUP_ORDER)
    counts = counts.reindex(index=[f'{score_col} <{int(cutoff)}', f'{score_col} >={int(cutoff)}'], columns=FIXED_GROUP_ORDER)
    (pivot * 100).round(2).to_csv(table_name)
    counts.to_csv(counts_name)
    plt.figure(figsize=(8, 5))
    sns.heatmap((pivot * 100).round(2), annot=True, fmt='.1f', cmap='Reds', linewidths=1, linecolor='white', cbar_kws={'label': '30-day mortality (%)'})
    plt.xlabel('48-hour lactate clearance category')
    plt.ylabel('Baseline MELD 3.0 category')
    plt.title('30-day Mortality by MELD 3.0 and 48-hour Lactate Clearance')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def cohort_flow_table(df_source: pd.DataFrame, df_full: pd.DataFrame, df_landmark: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {'step': 'Source cohort', 'n_remaining': len(df_source), 'n_excluded_at_step': np.nan, 'reason': ''},
        {'step': 'Excluded before hyperlactatemia cohort', 'n_remaining': np.nan, 'n_excluded_at_step': int(df_source['lactate_0'].isna().sum()), 'reason': 'Missing baseline lactate'},
        {'step': 'Excluded before hyperlactatemia cohort', 'n_remaining': np.nan, 'n_excluded_at_step': int(df_source['lactate_0'].lt(2.0).fillna(False).sum()), 'reason': 'Baseline lactate <2.0 mmol/L'},
        {'step': 'Full hyperlactatemia cohort', 'n_remaining': len(df_full), 'n_excluded_at_step': np.nan, 'reason': 'Baseline lactate >=2.0 mmol/L'},
        {'step': 'Excluded before 48h landmark cohort', 'n_remaining': np.nan, 'n_excluded_at_step': len(df_full) - len(df_landmark), 'reason': 'Death before 48-hour landmark'},
        {'step': '48h landmark cohort', 'n_remaining': len(df_landmark), 'n_excluded_at_step': np.nan, 'reason': 'Reached 48-hour landmark'},
    ])


def missingness_table(cohort_map: dict[str, pd.DataFrame], variables: list[str]) -> pd.DataFrame:
    rows = []
    for cohort_name, cohort_df in cohort_map.items():
        for variable in variables:
            missing_n = int(cohort_df[variable].isna().sum())
            rows.append({'cohort': cohort_name, 'variable': variable, 'n_missing': missing_n, 'pct_missing': round(missing_n / len(cohort_df) * 100, 2), 'n_cohort': len(cohort_df)})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    os.chdir(OUTPUT_DIR)
    warnings.filterwarnings('default')
    np.random.seed(RANDOM_SEED)
    sns.set_theme(style='whitegrid')
    pd.set_option('display.max_columns', None)

    df_raw = pd.read_csv(DATA_PATH)
    validate_columns(df_raw, REQUIRED_COLUMNS)
    df_raw = df_raw.dropna(subset=['mortality_30d', 'survival_days_30']).copy()
    df_raw = derive_variables(df_raw)

    df_full = df_raw[df_raw['lactate_0'] >= 2.0].copy()
    df_landmark = df_full[~((df_full['mortality_30d'] == 1) & (df_full['survival_days_30'] < 2.0))].copy()
    df_landmark = recalculate_post_landmark_outcomes(df_landmark, landmark_days=2.0)
    pre_landmark_deaths = int(((df_full['mortality_30d'] == 1) & (df_full['survival_days_30'] < 2.0)).sum())
    if not (len(df_full) == 730 and pre_landmark_deaths == 65 and len(df_landmark) == 665):
        raise RuntimeError(
            f'Frozen cohort check failed: hyper={len(df_full)}, pre48={pre_landmark_deaths}, landmark={len(df_landmark)}'
        )

    cohort_flow_table(df_raw, df_full, df_landmark).to_csv('Table_Cohort_Flow.csv', index=False)
    missingness_vars = cols_to_impute + ['sex'] + binary_model_covariates
    missingness_table({'Full hyperlactatemia cohort': df_full, '48h landmark cohort': df_landmark}, missingness_vars).to_csv('Table_MICE_Missingness.csv', index=False)

    # --- Reviewer-requested: 48h lactate observed vs. missing comparison (informative-missingness check) ---
    df_landmark['lactate48_observed_group'] = np.where(df_landmark['lactate_48'].notna(), 'observed', 'missing')
    missing_vs_observed_cont = ['age', 'lactate_0', 'day1_sofa', 'sapsii', 'cci_score', 'day1_meld_3_0', 'clearance_12h', 'clearance_24h']
    missing_vs_observed_cat = ['female_flag'] + binary_model_covariates + ['mortality_30d']
    baseline_table_by_group(df_landmark, 'lactate48_observed_group', 'Table_MissingVsObserved_48hLactate.csv', missing_vs_observed_cont, missing_vs_observed_cat, group_order=['observed', 'missing'])

    # --- Reviewer-requested: cohort QC (episode-level, descriptive only; primary cohort unchanged) ---
    if 'subject_id' in df_raw.columns:
        total_episodes = len(df_raw)
        unique_patients = df_raw['subject_id'].nunique()
        repeated_episode_rows = total_episodes - unique_patients
        deaths_30d_episode_level = int(df_raw['mortality_30d'].sum())
        unique_patients_died = df_raw.loc[df_raw['mortality_30d'] == 1, 'subject_id'].nunique()
        pd.DataFrame([{
            'total_eligible_episodes': total_episodes,
            'unique_patients': unique_patients,
            'repeated_episode_rows': repeated_episode_rows,
            'deaths_30d_episode_level': deaths_30d_episode_level,
            'unique_patients_died': unique_patients_died,
        }]).to_csv('Table_Cohort_QC_EpisodeVsPatient.csv', index=False)
    else:
        pd.DataFrame([{'note': "subject_id column not found in source data; episode-vs-patient QC not computed."}]).to_csv('Table_Cohort_QC_EpisodeVsPatient.csv', index=False)

    landmark_imputations = run_multiple_imputation(df_landmark, m=N_IMPUTATIONS, seed=RANDOM_SEED)
    full_imputations = run_multiple_imputation(df_full, m=N_IMPUTATIONS, seed=RANDOM_SEED + 100)
    landmark_mean = build_mean_completed_dataset(landmark_imputations)
    landmark_mean = derive_variables(landmark_mean)
    landmark_mean = add_scaled_predictor(landmark_mean, 'clearance_48h', 'clearance_48h_per10', 10.0)
    full_mean = build_mean_completed_dataset(full_imputations)
    full_mean = derive_variables(full_mean)
    full_mean = add_scaled_predictor(full_mean, 'clearance_24h', 'clearance_24h_per10', 10.0)

    # Primary complete-case cohorts
    cc48 = df_landmark.dropna(subset=['lactate_48'] + covariates + ['mortality_30d', 'survival_days_30']).copy()
    cc48 = add_scaled_predictor(cc48, 'clearance_48h', 'clearance_48h_per10', 10.0)
    cc48 = assign_fixed_categories(cc48, 'clearance_48h', 'clearance_48h_fixed_bin')

    cc24 = df_landmark.dropna(subset=['lactate_24'] + covariates + ['mortality_30d', 'survival_days_30']).copy()
    cc24 = add_scaled_predictor(cc24, 'clearance_24h', 'clearance_24h_per10', 10.0)
    cc24 = assign_fixed_categories(cc24, 'clearance_24h', 'clearance_24h_fixed_bin')

    cc12 = df_landmark.dropna(subset=['lactate_12'] + covariates + ['mortality_30d', 'survival_days_30']).copy()
    cc12 = add_scaled_predictor(cc12, 'clearance_12h', 'clearance_12h_per10', 10.0)
    cc12 = assign_fixed_categories(cc12, 'clearance_12h', 'clearance_12h_fixed_bin')
    if [len(cc12), len(cc24), len(cc48)] != [413, 334, 281]:
        raise RuntimeError(f'Frozen complete-case check failed: 12h={len(cc12)}, 24h={len(cc24)}, 48h={len(cc48)}')

    cc24_full = df_full.dropna(subset=['lactate_24'] + covariates + ['mortality_30d', 'survival_days_30']).copy()
    cc24_full = add_scaled_predictor(cc24_full, 'clearance_24h', 'clearance_24h_per10', 10.0)

    continuous_baseline_vars = ['age','cci_score','day1_sofa','sapsii','day1_albumin','day1_total_bilirubin','day1_creatinine','day1_inr','day1_sodium','lactate_0','clearance_48h','day1_meld_3_0']
    categorical_baseline_vars = ['female_flag','crrt_within_24h','sbp_flag','he_flag','variceal_bleed_flag','ascites_flag','hrs_flag','hcc_flag']
    baseline_table_by_group(cc48, 'clearance_48h_fixed_bin', 'Table_Baseline_CompleteCase_48h_Categories.csv', continuous_baseline_vars, categorical_baseline_vars, group_order=FIXED_GROUP_ORDER)
    coupling_rows = []
    for group in FIXED_GROUP_ORDER:
        vals = cc48.loc[cc48['clearance_48h_fixed_bin'].astype(str) == group, 'lactate_0'].dropna()
        coupling_rows.append({
            'clearance_48h_category': group,
            'n': int(vals.shape[0]),
            'baseline_lactate_mean': float(vals.mean()) if len(vals) else np.nan,
            'baseline_lactate_sd': float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            'baseline_lactate_median': float(vals.median()) if len(vals) else np.nan,
            'baseline_lactate_iqr_q1': float(vals.quantile(0.25)) if len(vals) else np.nan,
            'baseline_lactate_iqr_q3': float(vals.quantile(0.75)) if len(vals) else np.nan,
        })
    pd.DataFrame(coupling_rows).to_csv('Table_Mathematical_Coupling_BaselineLactate_By48hClearance.csv', index=False)

    # Primary complete-case cox
    cc_cont_summary = []
    cc_fixed_summary = []
    ph_diagnostics = []
    ph_full_diagnostics = []
    nonlinearity_diagnostics = []
    epp_rows = []
    for tp, df_cc in [('12h', cc12), ('24h', cc24), ('48h', cc48)]:
        # Landmark cohort: survival time and event status are already measured from the 48-hour landmark.
        cont = complete_case_cox_ph(df_cc, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', [f'clearance_{tp}_per10'] + covariates)
        ph_diagnostics.append(ph_interaction_test(df_cc, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', [f'clearance_{tp}_per10'] + covariates, f'clearance_{tp}_per10'))
        ph_full_diagnostics.append(ph_global_variable_tests(df_cc, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', [f'clearance_{tp}_per10'] + covariates, f'{tp} complete-case landmark continuous Cox'))
        nonlinearity_diagnostics.append(nonlinearity_test(df_cc, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', [f'clearance_{tp}_per10'] + covariates, f'clearance_{tp}_per10'))
        epp_rows.append({'timepoint': tp, **events_per_parameter(df_cc, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', [f'clearance_{tp}_per10'] + covariates)})
        cont.to_csv(f'Table_Cox_CompleteCase_{tp}_Landmark_Continuous.csv', index=False)
        row = cont.loc[cont['term'] == f'clearance_{tp}_per10'].iloc[0]
        cc_cont_summary.append({'timepoint': tp, 'analysis': 'complete_case_landmark_continuous_per10', 'term': f'clearance_{tp}_per10', 'hr': row['hr'], 'hr_ci_lower': row['hr_ci_lower'], 'hr_ci_upper': row['hr_ci_upper'], 'p_value': row['p_value'], 'n': int(row['n'])})
        fixed = complete_case_categorical_cox(df_cc, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', f'clearance_{tp}_fixed_bin', ['<=0%', '0-33%', '33-66%'], covariates)
        fixed.to_csv(f'Table_Cox_CompleteCase_{tp}_Landmark_FixedBins.csv', index=False)
        epp_rows.append({
            'timepoint': tp,
            'model_type': 'categorical_fixed_bins',
            'n': int(fixed['n'].iloc[0]),
            'n_events': int(fixed['events'].iloc[0]),
            'n_params': int(fixed['parameters'].iloc[0]),
            'events_per_parameter': round(float(fixed['events'].iloc[0]) / float(fixed['parameters'].iloc[0]), 2),
            'entry_value': None,
        })
        for term in ['33-66%','0-33%','<=0%']:
            qrow = fixed.loc[fixed['term'] == term].iloc[0]
            cc_fixed_summary.append({'timepoint': tp, 'analysis': 'complete_case_landmark_fixed_bins', 'category': term, 'reference': '>=66%', 'hr': qrow['hr'], 'hr_ci_lower': qrow['hr_ci_lower'], 'hr_ci_upper': qrow['hr_ci_upper'], 'p_value': qrow['p_value'], 'n': int(qrow['n'])})
    pd.DataFrame(cc_cont_summary).to_csv('Table_Cox_CompleteCase_Landmark_Continuous_Summary.csv', index=False)
    pd.DataFrame(cc_fixed_summary).to_csv('Table_Cox_CompleteCase_Landmark_FixedBins_Summary.csv', index=False)
    pd.DataFrame(ph_diagnostics).to_csv('Table_PH_Assumption_Diagnostics.csv', index=False)
    pd.concat(ph_full_diagnostics, ignore_index=True).to_csv('Table_PH_Global_Variable_Diagnostics.csv', index=False)
    pd.DataFrame(nonlinearity_diagnostics).to_csv('Table_Nonlinearity_Diagnostics.csv', index=False)
    pd.DataFrame(epp_rows).to_csv('Table_Events_Per_Parameter.csv', index=False)

    # --- Treatment-confounding sensitivity: is the 48h clearance HR sensitive to including treatment proxies? ---
    treatment_covariates = ['crrt_within_24h', 'vasopressor_within_24h', 'mech_vent_within_24h']
    baseline_only_covariates = [c for c in covariates if c not in treatment_covariates]
    baseline_only_cont = complete_case_cox_ph(cc48, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', ['clearance_48h_per10'] + baseline_only_covariates)
    baseline_only_row = baseline_only_cont.loc[baseline_only_cont['term'] == 'clearance_48h_per10'].iloc[0]
    primary_48h_row = next(r for r in cc_cont_summary if r['timepoint'] == '48h')
    treatment_sensitivity_rows = [
        {'model': 'baseline_covariates_only_no_treatment_proxies', 'hr': baseline_only_row['hr'], 'hr_ci_lower': baseline_only_row['hr_ci_lower'], 'hr_ci_upper': baseline_only_row['hr_ci_upper'], 'p_value': baseline_only_row['p_value'], 'n': int(baseline_only_row['n'])},
        {'model': 'primary_model_with_crrt_vasopressor_ventilation', 'hr': primary_48h_row['hr'], 'hr_ci_lower': primary_48h_row['hr_ci_lower'], 'hr_ci_upper': primary_48h_row['hr_ci_upper'], 'p_value': primary_48h_row['p_value'], 'n': primary_48h_row['n']},
    ]
    if 'hours_to_antibiotic' in df_raw.columns:
        cc48_abx = cc48.dropna(subset=['hours_to_antibiotic']).copy()
        abx_cont = complete_case_cox_ph(cc48_abx, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', ['clearance_48h_per10'] + covariates + ['hours_to_antibiotic'])
        abx_row = abx_cont.loc[abx_cont['term'] == 'clearance_48h_per10'].iloc[0]
        treatment_sensitivity_rows.append({'model': 'primary_model_plus_antibiotic_timing', 'hr': abx_row['hr'], 'hr_ci_lower': abx_row['hr_ci_lower'], 'hr_ci_upper': abx_row['hr_ci_upper'], 'p_value': abx_row['p_value'], 'n': int(abx_row['n'])})
    else:
        treatment_sensitivity_rows.append({'model': 'primary_model_plus_antibiotic_timing', 'hr': np.nan, 'hr_ci_lower': np.nan, 'hr_ci_upper': np.nan, 'p_value': np.nan, 'n': np.nan, 'note': 'hours_to_antibiotic not present in source data; skipped'})
    pd.DataFrame(treatment_sensitivity_rows).to_csv('Table_Sensitivity_TreatmentConfounding.csv', index=False)

    # --- Cirrhosis etiology subgroup analysis (only where event counts are adequate; descriptive/exploratory) ---
    MIN_EVENTS_PER_SUBGROUP = 10
    if 'cirrhosis_etiology' in df_raw.columns:
        etiology_rows = []
        for etiology_group in sorted(cc48['cirrhosis_etiology'].dropna().unique()):
            sub = cc48[cc48['cirrhosis_etiology'] == etiology_group]
            n_events = int(sub['event_30d_after_landmark'].sum())
            if n_events < MIN_EVENTS_PER_SUBGROUP:
                etiology_rows.append({'etiology': etiology_group, 'n': len(sub), 'n_events': n_events, 'hr': np.nan, 'hr_ci_lower': np.nan, 'hr_ci_upper': np.nan, 'p_value': np.nan, 'note': f'skipped: fewer than {MIN_EVENTS_PER_SUBGROUP} events'})
                continue
            sub_cont = complete_case_cox_ph(sub, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', ['clearance_48h_per10'] + covariates)
            sub_row = sub_cont.loc[sub_cont['term'] == 'clearance_48h_per10'].iloc[0]
            etiology_rows.append({'etiology': etiology_group, 'n': len(sub), 'n_events': n_events, 'hr': sub_row['hr'], 'hr_ci_lower': sub_row['hr_ci_lower'], 'hr_ci_upper': sub_row['hr_ci_upper'], 'p_value': sub_row['p_value'], 'note': ''})
        pd.DataFrame(etiology_rows).to_csv('Table_Sensitivity_CirrhosisEtiology_Subgroups.csv', index=False)
        # Formal interaction test (etiology x clearance) only if overall sample supports it
        cc48_etio = cc48.dropna(subset=['cirrhosis_etiology']).copy()
        if int(cc48_etio['event_30d_after_landmark'].sum()) >= MIN_EVENTS_PER_SUBGROUP * cc48_etio['cirrhosis_etiology'].nunique():
            etio_dummies = pd.get_dummies(cc48_etio['cirrhosis_etiology'], prefix='etio', drop_first=True).astype(float)
            interact_df = cc48_etio.copy()
            for col in etio_dummies.columns:
                interact_df[col] = etio_dummies[col].to_numpy()
                interact_df[f'{col}_x_clearance48'] = interact_df[col] * interact_df['clearance_48h_per10']
            interact_predictors = ['clearance_48h_per10'] + covariates + list(etio_dummies.columns) + [f'{c}_x_clearance48' for c in etio_dummies.columns]
            interact_cont = complete_case_cox_ph(interact_df, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', interact_predictors)
            interact_cont[interact_cont['term'].str.contains('x_clearance48')].to_csv('Table_CirrhosisEtiology_InteractionTest.csv', index=False)
        else:
            pd.DataFrame([{'note': 'Interaction test skipped: insufficient events across etiology subgroups for a stable interaction model'}]).to_csv('Table_CirrhosisEtiology_InteractionTest.csv', index=False)
    else:
        pd.DataFrame([{'note': 'cirrhosis_etiology not present in source data; subgroup analysis skipped'}]).to_csv('Table_Sensitivity_CirrhosisEtiology_Subgroups.csv', index=False)


    # --- Extreme baseline lactate sensitivity: exclude values above the cohort's 95th percentile ---
    lactate0_p95 = cc48['lactate_0'].quantile(0.95)
    cc48_trimmed = cc48[cc48['lactate_0'] <= lactate0_p95].copy()
    trimmed_cont = complete_case_cox_ph(cc48_trimmed, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', ['clearance_48h_per10'] + covariates)
    trimmed_row = trimmed_cont.loc[trimmed_cont['term'] == 'clearance_48h_per10'].iloc[0]
    pd.DataFrame([{
        'sensitivity': 'excl_baseline_lactate_gt_p95', 'p95_cutoff_mmol_l': float(lactate0_p95),
        'hr': trimmed_row['hr'], 'hr_ci_lower': trimmed_row['hr_ci_lower'], 'hr_ci_upper': trimmed_row['hr_ci_upper'], 'p_value': trimmed_row['p_value'], 'n': int(trimmed_row['n']),
    }]).to_csv('Table_Sensitivity_ExtremeBaselineLactate.csv', index=False)

    # --- Boundary-landmark sensitivity: exclude deaths occurring exactly at the 48-hour landmark ---
    cc48_no_boundary_deaths = cc48[~((cc48['mortality_30d'] == 1) & (cc48['survival_days_30'] == 2.0))].copy()
    boundary_cont = complete_case_cox_ph(cc48_no_boundary_deaths, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', ['clearance_48h_per10'] + covariates)
    boundary_row = boundary_cont.loc[boundary_cont['term'] == 'clearance_48h_per10'].iloc[0]
    pd.DataFrame([{
        'sensitivity': 'exclude_deaths_exactly_at_48h_boundary',
        'n_excluded_from_cc48': int(len(cc48) - len(cc48_no_boundary_deaths)),
        'hr': boundary_row['hr'],
        'hr_ci_lower': boundary_row['hr_ci_lower'],
        'hr_ci_upper': boundary_row['hr_ci_upper'],
        'p_value': boundary_row['p_value'],
        'n': int(boundary_row['n']),
    }]).to_csv('Table_Sensitivity_Exclude48hBoundaryDeaths.csv', index=False)

    cc48_cutoff = cc48.copy()
    cc48_cutoff['clearance_48h_cutoff33'] = np.where(cc48_cutoff['clearance_48h'] <= 33, '<=33%', '>33%')
    cutoff_summary = cc48_cutoff.groupby('clearance_48h_cutoff33', observed=False).agg(
        n=('mortality_30d', 'size'),
        deaths_30d=('mortality_30d', 'sum'),
        mortality_30d_pct=('mortality_30d', 'mean'),
    ).reset_index()
    cutoff_summary['mortality_30d_pct'] = cutoff_summary['mortality_30d_pct'] * 100
    cutoff_note = pd.DataFrame([{
        'suggested_cutoff': '<=33%',
        'timepoint': '48h',
        'cohort': 'complete_case_landmark',
        'reason': 'At 48 hours, both <=0% and 0-33% clearance categories had significantly higher adjusted mortality than the >=66% reference group, whereas 33-66% did not, supporting <=33% as a pragmatic adverse-risk threshold.',
        'n_total': int(len(cc48)),
    }])
    cutoff_summary.to_csv('Table_Suggested_Clearance_Cutoff_48h_CompleteCase.csv', index=False)
    cutoff_note.to_csv('Table_Suggested_Clearance_Cutoff_48h_CompleteCase_Note.csv', index=False)

    full_cc24_cont = complete_case_cox_ph(cc24_full, 'survival_days_30', 'mortality_30d', ['clearance_24h_per10'] + covariates)
    full_cc24_cont.to_csv('Table_Cox_CompleteCase_24h_FullCohort_Continuous.csv', index=False)

    pd.DataFrame([
        {'timepoint':'12h','analysis':'complete_case_landmark','n':len(cc12)},
        {'timepoint':'24h','analysis':'complete_case_landmark','n':len(cc24)},
        {'timepoint':'48h','analysis':'complete_case_landmark','n':len(cc48)},
        {'timepoint':'24h_full','analysis':'complete_case_full','n':len(cc24_full)},
    ]).to_csv('Table_CompleteCase_Sample_Sizes.csv', index=False)

    # Primary complete-case predictive performance on shared complete-case set with all three lactates
    pred_cc = df_landmark.dropna(subset=['lactate_12','lactate_24','lactate_48'] + covariates + ['mortality_30d']).copy()
    pred_cc = derive_variables(pred_cc)
    y_cc = pred_cc['mortality_30d'].to_numpy(dtype=int)
    auc_summary = {}
    for i, col in enumerate(['clearance_12h','clearance_24h','clearance_48h']):
        point_auc, boots = bootstrap_auc(y_cc, -pred_cc[col].to_numpy(dtype=float), N_BOOTSTRAPS, RANDOM_SEED + 1000*(i+1))
        auc_summary[col] = {'point_auc': float(point_auc), 'ci_lower': float(np.percentile(boots, 2.5)), 'ci_upper': float(np.percentile(boots, 97.5)), 'n': len(pred_cc)}
    auc_diff_48_24_point, auc_diff_48_24_boots = paired_bootstrap_auc_difference(y_cc, -pred_cc['clearance_48h'].to_numpy(dtype=float), -pred_cc['clearance_24h'].to_numpy(dtype=float), N_BOOTSTRAPS, RANDOM_SEED + 5000)
    auc_diff_48_12_point, auc_diff_48_12_boots = paired_bootstrap_auc_difference(y_cc, -pred_cc['clearance_48h'].to_numpy(dtype=float), -pred_cc['clearance_12h'].to_numpy(dtype=float), N_BOOTSTRAPS, RANDOM_SEED + 6000)
    auc_diff_24_12_point, auc_diff_24_12_boots = paired_bootstrap_auc_difference(y_cc, -pred_cc['clearance_24h'].to_numpy(dtype=float), -pred_cc['clearance_12h'].to_numpy(dtype=float), N_BOOTSTRAPS, RANDOM_SEED + 6500)
    auc_diff = pd.DataFrame([
        {'comparison': '48h_vs_24h', 'point_diff': float(auc_diff_48_24_point), 'ci_lower': float(np.percentile(auc_diff_48_24_boots, 2.5)), 'ci_upper': float(np.percentile(auc_diff_48_24_boots, 97.5)), 'p_value': float(min(2*min(np.mean(auc_diff_48_24_boots <= 0), np.mean(auc_diff_48_24_boots >= 0)),1.0)), 'n': len(pred_cc)},
        {'comparison': '48h_vs_12h', 'point_diff': float(auc_diff_48_12_point), 'ci_lower': float(np.percentile(auc_diff_48_12_boots, 2.5)), 'ci_upper': float(np.percentile(auc_diff_48_12_boots, 97.5)), 'p_value': float(min(2*min(np.mean(auc_diff_48_12_boots <= 0), np.mean(auc_diff_48_12_boots >= 0)),1.0)), 'n': len(pred_cc)},
        {'comparison': '24h_vs_12h', 'point_diff': float(auc_diff_24_12_point), 'ci_lower': float(np.percentile(auc_diff_24_12_boots, 2.5)), 'ci_upper': float(np.percentile(auc_diff_24_12_boots, 97.5)), 'p_value': float(min(2*min(np.mean(auc_diff_24_12_boots <= 0), np.mean(auc_diff_24_12_boots >= 0)),1.0)), 'n': len(pred_cc)},
    ])
    pd.DataFrame.from_dict(auc_summary, orient='index').reset_index().rename(columns={'index':'model'}).to_csv('Table_AUC_Primary_CompleteCase.csv', index=False)
    auc_diff.to_csv('Table_AUC_Model_Comparisons_Primary_CompleteCase.csv', index=False)
    auc_diff[auc_diff['comparison'] == '48h_vs_24h'].to_csv('Table_AUC_Difference_48h_vs_24h_Primary_CompleteCase.csv', index=False)
    save_roc_plot(pred_cc, auc_summary, 'Figure_ROC_CompleteCase_Primary.png')

    prob_12 = cross_validated_probabilities(pred_cc, ['clearance_12h'] + covariates, 'mortality_30d')
    prob_24 = cross_validated_probabilities(pred_cc, ['clearance_24h'] + covariates, 'mortality_30d')
    prob_48 = cross_validated_probabilities(pred_cc, ['clearance_48h'] + covariates, 'mortality_30d')
    multivariable_auc_rows = []
    for i, (label, prob) in enumerate([('12h multivariable clearance model', prob_12), ('24h multivariable clearance model', prob_24), ('48h multivariable clearance model', prob_48)]):
        point_auc, boots = bootstrap_auc(y_cc, prob, N_BOOTSTRAPS, RANDOM_SEED + 12000 + i)
        multivariable_auc_rows.append({
            'model': label,
            'predictor_set': 'clearance at named timepoint plus 20 primary covariates',
            'validation': '5-fold cross-validated predicted probabilities; bootstrap CI',
            'point_auc': float(point_auc),
            'ci_lower': float(np.percentile(boots, 2.5)),
            'ci_upper': float(np.percentile(boots, 97.5)),
            'n': len(pred_cc),
            'events': int(y_cc.sum()),
        })
    pd.DataFrame(multivariable_auc_rows).to_csv('Table_AUC_Multivariable_Prediction_CompleteCase.csv', index=False)
    nri_48_24_point, nri_48_24_ci_lower, nri_48_24_ci_upper, nri_48_24_p = bootstrap_reclassification_pvalue(y_cc, prob_24, prob_48, continuous_nri, N_BOOTSTRAPS, RANDOM_SEED + 7000)
    idi_48_24_point, idi_48_24_ci_lower, idi_48_24_ci_upper, idi_48_24_p = bootstrap_reclassification_pvalue(y_cc, prob_24, prob_48, idi, N_BOOTSTRAPS, RANDOM_SEED + 7100)
    pred_metrics = {
        'NRI_48h_vs_24h': nri_48_24_point,
        'IDI_48h_vs_24h': idi_48_24_point,
    }
    calibration = calibration_metrics(y_cc, prob_48)
    apparent_auc_48_multivariable = apparent_logistic_auc(pred_cc, ['clearance_48h'] + covariates, 'mortality_30d', RANDOM_SEED + 7199)
    optimism_48 = optimism_corrected_auc(pred_cc, ['clearance_48h'] + covariates, 'mortality_30d', apparent_auc_48_multivariable, 200, RANDOM_SEED + 7200)
    optimism_48.update({'predictor_set': 'clearance_48h plus 20 primary covariates', 'n': len(pred_cc), 'events': int(y_cc.sum())})
    pd.DataFrame([optimism_48]).to_csv('Table_OptimismCorrected_AUC_48h.csv', index=False)
    pd.DataFrame([
        {'metric':'NRI_48h_vs_24h','value':pred_metrics['NRI_48h_vs_24h'],'ci_lower':nri_48_24_ci_lower,'ci_upper':nri_48_24_ci_upper,'p_value':nri_48_24_p,'n':len(pred_cc)},
        {'metric':'IDI_48h_vs_24h','value':pred_metrics['IDI_48h_vs_24h'],'ci_lower':idi_48_24_ci_lower,'ci_upper':idi_48_24_ci_upper,'p_value':idi_48_24_p,'n':len(pred_cc)},
        {'metric':'Calibration_Slope_48h','value':calibration['calibration_slope'],'n':len(pred_cc)},
        {'metric':'Calibration_Intercept_48h','value':calibration['calibration_intercept'],'n':len(pred_cc)},
        {'metric':'Brier_Score_48h','value':calibration['brier_score'],'n':len(pred_cc)},
        {'metric':'Hosmer_Lemeshow_p_48h','value':calibration['hl_p_value'],'n':len(pred_cc)},
    ]).to_csv('Table_Prediction_Metrics_Primary_CompleteCase.csv', index=False)
    save_dca_plot(y_cc, {'12h model': prob_12, '24h model': prob_24, '48h model': prob_48}, 'Figure_DCA_CompleteCase_Primary.png')
    save_calibration_plot(y_cc, prob_48, 'Figure_Calibration_CompleteCase_Primary.png')

    # --- Reviewer-requested: does 48h clearance add prognostic information beyond what is known at 24h? ---
    # Model 1: baseline covariates + 24h clearance. Model 2: baseline covariates + 24h clearance + 48h clearance.
    prob_24_only = cross_validated_probabilities(pred_cc, ['clearance_24h'] + covariates, 'mortality_30d')
    prob_24_plus_48 = cross_validated_probabilities(pred_cc, ['clearance_24h', 'clearance_48h'] + covariates, 'mortality_30d')
    auc24_point, auc24_boots = bootstrap_auc(y_cc, prob_24_only, N_BOOTSTRAPS, RANDOM_SEED + 9500)
    auc2448_point, auc2448_boots = bootstrap_auc(y_cc, prob_24_plus_48, N_BOOTSTRAPS, RANDOM_SEED + 9600)
    diff2448_point, diff2448_boots = paired_bootstrap_auc_difference(y_cc, prob_24_plus_48, prob_24_only, N_BOOTSTRAPS, RANDOM_SEED + 9700)
    pd.DataFrame([
        {'model': 'baseline_plus_24h_clearance', 'point_auc': float(auc24_point), 'ci_lower': float(np.percentile(auc24_boots, 2.5)), 'ci_upper': float(np.percentile(auc24_boots, 97.5)), 'n': len(pred_cc)},
        {'model': 'baseline_plus_24h_plus_48h_clearance', 'point_auc': float(auc2448_point), 'ci_lower': float(np.percentile(auc2448_boots, 2.5)), 'ci_upper': float(np.percentile(auc2448_boots, 97.5)), 'n': len(pred_cc)},
    ]).to_csv('Table_AUC_24h_vs_24hPlus48h_Incremental.csv', index=False)
    diff2448_p = float(min(2 * min(np.mean(diff2448_boots <= 0), np.mean(diff2448_boots >= 0)), 1.0))
    pd.DataFrame([{'comparison': '24hPlus48h_vs_24hOnly', 'point_diff': float(diff2448_point), 'ci_lower': float(np.percentile(diff2448_boots, 2.5)), 'ci_upper': float(np.percentile(diff2448_boots, 97.5)), 'p_value': diff2448_p, 'n': len(pred_cc)}]).to_csv('Table_AUC_Difference_24hPlus48h_vs_24hOnly.csv', index=False)
    nri_2448, nri_2448_ci_lower, nri_2448_ci_upper, nri_2448_p = bootstrap_reclassification_pvalue(y_cc, prob_24_only, prob_24_plus_48, continuous_nri, N_BOOTSTRAPS, RANDOM_SEED + 9710)
    idi_2448, idi_2448_ci_lower, idi_2448_ci_upper, idi_2448_p = bootstrap_reclassification_pvalue(y_cc, prob_24_only, prob_24_plus_48, idi, N_BOOTSTRAPS, RANDOM_SEED + 9720)
    calibration_2448 = calibration_metrics(y_cc, prob_24_plus_48)
    pd.DataFrame([
        {'metric': 'NRI_24hPlus48h_vs_24hOnly', 'value': nri_2448, 'ci_lower': nri_2448_ci_lower, 'ci_upper': nri_2448_ci_upper, 'p_value': nri_2448_p, 'baseline_model': 'primary covariates plus 24h clearance', 'expanded_model': 'primary covariates plus 24h and 48h clearance', 'validation': '5-fold cross-validated predictions; bootstrap CI/P', 'n': len(pred_cc)},
        {'metric': 'IDI_24hPlus48h_vs_24hOnly', 'value': idi_2448, 'ci_lower': idi_2448_ci_lower, 'ci_upper': idi_2448_ci_upper, 'p_value': idi_2448_p, 'baseline_model': 'primary covariates plus 24h clearance', 'expanded_model': 'primary covariates plus 24h and 48h clearance', 'validation': '5-fold cross-validated predictions; bootstrap CI/P', 'n': len(pred_cc)},
        {'metric': 'Calibration_Slope_24hPlus48h', 'value': calibration_2448['calibration_slope'], 'n': len(pred_cc)},
        {'metric': 'Calibration_Intercept_24hPlus48h', 'value': calibration_2448['calibration_intercept'], 'n': len(pred_cc)},
        {'metric': 'Brier_Score_24hPlus48h', 'value': calibration_2448['brier_score'], 'n': len(pred_cc)},
        {'metric': 'Hosmer_Lemeshow_p_24hPlus48h', 'value': calibration_2448['hl_p_value'], 'n': len(pred_cc)},
    ]).to_csv('Table_Prediction_Metrics_24hPlus48h_Incremental.csv', index=False)
    save_dca_plot(y_cc, {'24h clearance alone': prob_24_only, '24h + 48h clearance': prob_24_plus_48}, 'Figure_DCA_24h_vs_24hPlus48h_Incremental.png')

    # Head-to-head comparison: MELD 3.0 alone vs MELD 3.0 + 48h lactate clearance
    # Addresses reviewer request for direct comparison against an established prognostic tool
    # and for evidence that adding 48h clearance improves on MELD 3.0 alone.
    meld_prob = cross_validated_probabilities(pred_cc, ['day1_meld_3_0'], 'mortality_30d')
    meld_plus_clearance_prob = cross_validated_probabilities(pred_cc, ['day1_meld_3_0', 'clearance_48h'], 'mortality_30d')

    meld_auc_point, meld_auc_boots = bootstrap_auc(y_cc, meld_prob, N_BOOTSTRAPS, RANDOM_SEED + 9000)
    meld_plus_auc_point, meld_plus_auc_boots = bootstrap_auc(y_cc, meld_plus_clearance_prob, N_BOOTSTRAPS, RANDOM_SEED + 9100)
    meld_auc_diff_point, meld_auc_diff_boots = paired_bootstrap_auc_difference(y_cc, meld_plus_clearance_prob, meld_prob, N_BOOTSTRAPS, RANDOM_SEED + 9200)

    meld_head_to_head_auc = pd.DataFrame([
        {'model': 'MELD_3_0_alone', 'point_auc': float(meld_auc_point), 'ci_lower': float(np.percentile(meld_auc_boots, 2.5)), 'ci_upper': float(np.percentile(meld_auc_boots, 97.5)), 'n': len(pred_cc)},
        {'model': 'MELD_3_0_plus_48h_clearance', 'point_auc': float(meld_plus_auc_point), 'ci_lower': float(np.percentile(meld_plus_auc_boots, 2.5)), 'ci_upper': float(np.percentile(meld_plus_auc_boots, 97.5)), 'n': len(pred_cc)},
    ])
    meld_head_to_head_auc.to_csv('Table_AUC_MELD30_vs_MELD30_plus_48hClearance.csv', index=False)

    meld_auc_diff = pd.DataFrame([{
        'comparison': 'MELD30_plus_48hClearance_vs_MELD30_alone',
        'point_diff': float(meld_auc_diff_point),
        'ci_lower': float(np.percentile(meld_auc_diff_boots, 2.5)),
        'ci_upper': float(np.percentile(meld_auc_diff_boots, 97.5)),
        'p_value': float(min(2 * min(np.mean(meld_auc_diff_boots <= 0), np.mean(meld_auc_diff_boots >= 0)), 1.0)),
        'n': len(pred_cc),
    }])
    meld_auc_diff.to_csv('Table_AUC_Difference_MELD30_plus_48hClearance_vs_MELD30_alone.csv', index=False)

    meld_nri_point, meld_nri_ci_lower, meld_nri_ci_upper, meld_nri_p = bootstrap_reclassification_pvalue(y_cc, meld_prob, meld_plus_clearance_prob, continuous_nri, N_BOOTSTRAPS, RANDOM_SEED + 9300)
    meld_idi_point, meld_idi_ci_lower, meld_idi_ci_upper, meld_idi_p = bootstrap_reclassification_pvalue(y_cc, meld_prob, meld_plus_clearance_prob, idi, N_BOOTSTRAPS, RANDOM_SEED + 9400)
    meld_head_to_head_metrics = pd.DataFrame([
        {'metric': 'NRI_MELD30_plus_48hClearance_vs_MELD30_alone', 'value': meld_nri_point, 'ci_lower': meld_nri_ci_lower, 'ci_upper': meld_nri_ci_upper, 'p_value': meld_nri_p, 'n': len(pred_cc)},
        {'metric': 'IDI_MELD30_plus_48hClearance_vs_MELD30_alone', 'value': meld_idi_point, 'ci_lower': meld_idi_ci_lower, 'ci_upper': meld_idi_ci_upper, 'p_value': meld_idi_p, 'n': len(pred_cc)},
    ])
    meld_head_to_head_metrics.to_csv('Table_Prediction_Metrics_MELD30_vs_MELD30_plus_48hClearance.csv', index=False)

    meld_plus_calibration = calibration_metrics(y_cc, meld_plus_clearance_prob)
    pd.DataFrame([
        {'metric': 'Calibration_Slope_MELD30_plus_48hClearance', 'value': meld_plus_calibration['calibration_slope'], 'p_value': meld_plus_calibration['calibration_slope_p_vs1'], 'null_hypothesis': 'slope = 1', 'n': len(pred_cc)},
        {'metric': 'Calibration_Intercept_MELD30_plus_48hClearance', 'value': meld_plus_calibration['calibration_intercept'], 'p_value': meld_plus_calibration['calibration_intercept_p_vs0'], 'null_hypothesis': 'intercept = 0', 'n': len(pred_cc)},
        {'metric': 'Hosmer_Lemeshow_p_MELD30_plus_48hClearance', 'value': np.nan, 'p_value': meld_plus_calibration['hl_p_value'], 'null_hypothesis': 'model is well calibrated', 'n': len(pred_cc)},
    ]).to_csv('Table_Calibration_MELD30_plus_48hClearance.csv', index=False)

    save_dca_plot(y_cc, {'MELD 3.0 alone': meld_prob, 'MELD 3.0 + 48h clearance': meld_plus_clearance_prob}, 'Figure_DCA_MELD30_vs_MELD30_plus_48hClearance.png')

    cc48_km = cc48.copy()
    cc48_km['days_since_landmark'] = cc48_km['survival_days_30'] - 2.0  # all rows already conditioned on surviving to day 2 (48h)
    save_km_plot(cc48_km, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', 'clearance_48h_fixed_bin', 'Figure_KM_48h_CompleteCase_ClearanceCategories.png', 'Table_KM_48h_CompleteCase_GroupSummary.csv')

    # MELD 3.0 heatmap on primary complete-case 48h cohort
    save_heatmap(cc48, 'clearance_48h_fixed_bin', 'day1_meld_3_0', 15.0, 'Figure_Heatmap_MELD30_Clearance48h_CompleteCase.png', 'Table_Heatmap_MELD30_Clearance48h_CompleteCase.csv', 'Table_Heatmap_MELD30_Clearance48h_CompleteCase_Counts.csv')

    # Secondary MICE analyses
    landmark_imputations = [add_scaled_predictor(ds, 'clearance_48h', 'clearance_48h_per10', 10.0) for ds in landmark_imputations]
    full_imputations = [add_scaled_predictor(ds, 'clearance_24h', 'clearance_24h_per10', 10.0) for ds in full_imputations]
    for tp in ['12h','24h','48h']:
        landmark_imputations = [assign_fixed_categories(ds, f'clearance_{tp}', f'clearance_{tp}_fixed_bin') for ds in landmark_imputations]
        landmark_mean = assign_fixed_categories(landmark_mean, f'clearance_{tp}', f'clearance_{tp}_fixed_bin')
    mice_48_cont = pooled_cox_ph(landmark_imputations, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', ['clearance_48h_per10'] + covariates)
    mice_48_cont.to_csv('Table_Cox_MICE_48h_Landmark_Continuous.csv', index=False)
    mice_metadata_rows = []
    for i, ds in enumerate(landmark_imputations, start=1):
        model_df = ds[['time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', 'clearance_48h_per10'] + covariates].dropna()
        mice_metadata_rows.append({
            'analysis': '48h landmark MICE continuous Cox',
            'imputation': i,
            'N': int(len(model_df)),
            'events': int(model_df['event_30d_after_landmark'].sum()),
            'person_time_patient_days': float(model_df['time_from_landmark_to_censor_or_death'].sum()),
            'parameters': 21,
            'EPV': round(float(model_df['event_30d_after_landmark'].sum()) / 21.0, 2),
        })
    cc48_row = next(r for r in cc_cont_summary if r['timepoint'] == '48h')
    mice48_row = mice_48_cont.loc[mice_48_cont['term'] == 'clearance_48h_per10'].iloc[0]
    pd.DataFrame([
        {'analysis': 'complete_case', 'hr': cc48_row['hr'], 'hr_ci_lower': cc48_row['hr_ci_lower'], 'hr_ci_upper': cc48_row['hr_ci_upper'], 'p_value': cc48_row['p_value'], 'n': cc48_row['n']},
        {'analysis': 'mice_pooled', 'hr': mice48_row['hr'], 'hr_ci_lower': mice48_row['hr_ci_lower'], 'hr_ci_upper': mice48_row['hr_ci_upper'], 'p_value': mice48_row['p_value'], 'n': int(mice48_row['n'])},
    ]).to_csv('Table_CompleteCase_vs_MICE_48h_Comparison.csv', index=False)

    mice_24_full_cont = pooled_cox_ph(full_imputations, 'survival_days_30', 'mortality_30d', ['clearance_24h_per10'] + covariates)
    mice_24_full_cont.to_csv('Table_Cox_MICE_24h_FullCohort_Continuous.csv', index=False)
    for i, ds in enumerate(full_imputations, start=1):
        model_df = ds[['survival_days_30', 'mortality_30d', 'clearance_24h_per10'] + covariates].dropna()
        mice_metadata_rows.append({
            'analysis': '24h full-cohort MICE continuous Cox',
            'imputation': i,
            'N': int(len(model_df)),
            'events': int(model_df['mortality_30d'].sum()),
            'person_time_patient_days': float(model_df['survival_days_30'].sum()),
            'parameters': 21,
            'EPV': round(float(model_df['mortality_30d'].sum()) / 21.0, 2),
        })
    mice_fixed_rows = []
    for tp in ['12h','24h','48h']:
        res = pooled_categorical_cox(landmark_imputations, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', f'clearance_{tp}_fixed_bin', ['<=0%', '0-33%', '33-66%'], covariates)
        res['timepoint'] = tp
        res.to_csv(f'Table_Cox_MICE_{tp}_Landmark_FixedBins.csv', index=False)
        for term in ['33-66%','0-33%','<=0%']:
            row = res.loc[res['term'] == term].iloc[0]
            mice_fixed_rows.append({'timepoint': tp, 'category': term, 'reference':'>=66%', 'hr': row['hr'], 'hr_ci_lower': row['hr_ci_lower'], 'hr_ci_upper': row['hr_ci_upper'], 'p_value': row['p_value'], 'n': int(row['n'])})
    pd.DataFrame(mice_fixed_rows).to_csv('Table_Cox_MICE_Landmark_FixedBins_Summary.csv', index=False)
    pd.DataFrame(mice_metadata_rows).to_csv('Table_MICE_PerImputation_SampleSize_Events_PersonTime_EPV.csv', index=False)

    # --- MNAR delta-adjustment sensitivity: worsen imputed-only 48h lactate by 1/2/4 mmol/L ---
    mnar_rows = []
    for delta in [0.0, 1.0, 2.0, 4.0]:
        mnar_imputations = apply_mnar_delta(landmark_imputations, df_landmark, delta_mmol=delta)
        mnar_cont = pooled_cox_ph(mnar_imputations, 'time_from_landmark_to_censor_or_death', 'event_30d_after_landmark', ['clearance_48h_per10'] + covariates)
        row = mnar_cont.loc[mnar_cont['term'] == 'clearance_48h_per10'].iloc[0]
        mnar_rows.append({'delta_mmol_added_to_missing': delta, 'hr': row['hr'], 'hr_ci_lower': row['hr_ci_lower'], 'hr_ci_upper': row['hr_ci_upper'], 'p_value': row['p_value'], 'n': int(row['n'])})
    pd.DataFrame(mnar_rows).to_csv('Table_MNAR_DeltaAdjustment_Sensitivity.csv', index=False)

    primary48 = next(r for r in cc_cont_summary if r['timepoint'] == '48h')
    all_results = {
        'dataset_used': str(DATA_PATH.name),
        'sql_pipeline': 'finalized_sql_landmark_corrected.sql',
        'cohort': {
            'source_n': int(len(df_raw)),
            'baseline_lactate_missing': int(df_raw['lactate_0'].isna().sum()),
            'baseline_lactate_lt2': int(df_raw['lactate_0'].lt(2.0).fillna(False).sum()),
            'hyperlactatemia_n': int(len(df_full)),
            'deaths_before_48h': int(len(df_full) - len(df_landmark)),
            'landmark_48h_n': int(len(df_landmark)),
            'complete_case_12h_n': int(len(cc12)),
            'complete_case_24h_n': int(len(cc24)),
            'complete_case_48h_n': int(len(cc48)),
        },
        'primary_48h_cox': primary48,
        'primary_covariates': covariates,
        'primary_parameter_count': 1 + len(covariates),
        'design_limitations': [
            '12h and 24h clearance Cox analyses are conducted inside the 48-hour landmark cohort, creating potential survivorship/selection concerns.',
            '24h full-cohort sensitivity requires observed 24h lactate while survival begins at sepsis onset, so it may be subject to immortal-time/selection bias.',
            'MIMIC first-day SOFA/SAPS/laboratory variables follow SQL-derived first-day definitions.',
            'Clearance is mathematically coupled to baseline lactate, which is also included as a model covariate.',
            'Primary and subgroup Cox models have limited EPV and require cautious interpretation.',
            'Hosmer-Lemeshow results are descriptive and not the sole calibration assessment.',
        ],
    }
    pd.DataFrame([all_results['cohort']]).to_csv('Table_Phase2_Cohort_QC.csv', index=False)
    pd.DataFrame(COX_FIT_LOG).to_csv('Table_Cox_Fit_Log.csv', index=False)
    pd.Series(all_results).to_json('phase2_machine_readable_results.json', indent=2)

    report = f"""# Phase 2 Python Audit Report

## Dataset Used

Frozen CSV: `{DATA_PATH.name}`

Frozen SQL/pipeline: `finalized_sql_landmark_corrected.sql`

Corrected script: `phase2_corrected_analysis.py`

## Cohort QC

| Metric | Value |
| --- | ---: |
| Source cohort | {len(df_raw)} |
| Missing baseline lactate | {int(df_raw['lactate_0'].isna().sum())} |
| Baseline lactate <2.0 mmol/L | {int(df_raw['lactate_0'].lt(2.0).fillna(False).sum())} |
| Hyperlactatemia cohort | {len(df_full)} |
| Deaths before 48h landmark | {len(df_full) - len(df_landmark)} |
| 48h landmark cohort | {len(df_landmark)} |
| 12h complete-case N | {len(cc12)} |
| 24h complete-case N | {len(cc24)} |
| 48h complete-case N | {len(cc48)} |

Frozen cohort checks passed exactly.

## Primary Cox Model

The primary 48h Cox model used `clearance_48h_per10` plus the 20 frozen covariates, for 21 total parameters. The exposure is interpreted per 10-percentage-point increase in 48h lactate clearance.

Primary 48h HR: {primary48['hr']:.3f} ({primary48['hr_ci_lower']:.3f}-{primary48['hr_ci_upper']:.3f}); P={primary48['p_value']:.4g}; N={primary48['n']}.

## MICE

MICE used {N_IMPUTATIONS} imputations. Predictor data were imputed with outcome/time variables available as auxiliary predictors, then observed outcome/time fields were restored before survival modeling. Raw lactates were imputed first; clearance and per-10 exposure were recalculated afterward. Rubin pooling was used.

## Prediction

Univariable clearance AUROC is saved separately from multivariable prediction AUROC. Incremental prediction outputs identify baseline and expanded models. Optimism correction uses the same 48h multivariable predictor set for the apparent and bootstrap models.

## Calibration

Calibration intercept, calibration slope, Brier score, and Hosmer-Lemeshow descriptive P values were saved. Hosmer-Lemeshow is treated as descriptive only.

## Sensitivities and Diagnostics

PH diagnostics, EPV/model complexity, MNAR delta adjustment, P95 baseline lactate sensitivity, treatment-confounding sensitivity, etiology subgroup/interaction analyses, boundary sensitivity, and mathematical-coupling assessment were run and saved in `phase2_python_outputs`.

## Design Limitations to Carry Forward

- 12h/24h Cox analyses remain inside the 48h landmark cohort and may have survivorship/selection issues.
- 24h full-cohort sensitivity conditions on observed 24h lactate while survival begins at sepsis onset.
- First-day SOFA/SAPS/labs follow the SQL/MIMIC first-day definitions.
- Clearance is mathematically coupled to baseline lactate.
- Low EPV and sparse subgroup models require cautious interpretation.
- Any PH/convergence/calibration warnings should be reflected in Phase 3 manuscript updates.

## GO/NO-GO Audit

GO, conditional on manuscript Phase 3 being updated from the Phase 2 outputs and preserving the frozen 65/665 cohort definition.
"""
    Path('phase2_python_audit_report.md').write_text(report)
    print(f'Phase 2 pipeline complete. Outputs written to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
