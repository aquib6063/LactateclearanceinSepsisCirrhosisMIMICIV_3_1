#!/usr/bin/env python3
"""
Round-3 reviewer addendum  (WJCCM manuscript 126511)
=====================================================

Stand-alone add-on to `02_lactate_clearance_main_analysis.py`.
It does NOT modify the original script and does NOT write into its output folder.

It imports the original script's functions/constants (without running its main()),
rebuilds the identical frozen cohorts (730 / 665 landmark / 413-334-281 complete-case),
and runs only the two analyses requested in the round-3 response:

  A. Reduced-parameter Cox model (reviewer comment 2: EPV / overfitting / collinearity)
     Exposure + baseline lactate, age, day-1 SOFA, MELD 3.0, vasopressor <=24h,
     mechanical ventilation <=24h  ->  7 parameters.
     Fitted on the SAME complete-case cohorts as the primary 21-parameter model,
     so the two are directly comparable. Done for 12h, 24h and 48h; 48h is the headline.

  B. Restricted cubic spline for 48-hour lactate clearance (reviewer comment 3: nonlinearity)
     3 knots at the 10th/50th/90th percentiles (Harrell), reference = 0% clearance.
     Main: adjusted for the 20 primary covariates. Sensitivity: adjusted for the reduced set.
     Reports overall (2-df) and nonlinearity (1-df) Wald tests, HRs at fixed clearance
     values, and an editable SVG + 300-dpi PNG figure.

Same Cox settings as the original (lifelines CoxPHFitter, penalizer = COX_PENALIZER = 0.01).

Usage (from the folder that contains the original script and the data CSV):
    python 03_lactate_clearance_sensitivity_rcs_reduced.py

Optional environment variables:
    ORIGINAL_SCRIPT   path to the original analysis script (default: ./02_lactate_clearance_main_analysis.py)
    DATA_PATH         path to the extracted CSV (default: the original script's DATA_PATH)
    RCS_REF           reference clearance (%) for the spline HR curve (default: 0)
    SKIP_FROZEN_CHECK set to 1 ONLY for testing on dummy data
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = Path(os.environ.get('ORIGINAL_SCRIPT', SCRIPT_DIR / '02_lactate_clearance_main_analysis.py')).resolve()
OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'sensitivity'
RCS_REF = float(os.environ.get('RCS_REF', '0'))
RCS_KNOT_PERCENTILES = (10, 50, 90)
PLOT_RANGE_PERCENTILES = (2.5, 97.5)
HR_TICKS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10]
HR_TABLE_VALUES = [-50, -25, 0, 10, 25, 33, 50, 66, 75, 90]
SKIP_FROZEN_CHECK = os.environ.get('SKIP_FROZEN_CHECK', '0') == '1'

DUR = 'time_from_landmark_to_censor_or_death'
EVT = 'event_30d_after_landmark'

# Reduced covariate set (exposure is added on top -> 7 parameters).
# SOFA, SAPS II and MELD 3.0 already contain the single labs; SAPS II is dropped
# because it overlaps heavily with SOFA, the single labs are represented by MELD 3.0.
REDUCED_COVARIATES = [
    'lactate_0',
    'age',
    'day1_sofa',
    'day1_meld_3_0',
    'vasopressor_within_24h',
    'mech_vent_within_24h',
]


# --------------------------------------------------------------------------------------
# Load the original script as a module (its main() is guarded by __name__ and does not run)
# --------------------------------------------------------------------------------------
def load_original_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f'Original script not found at {path}. Put this file next to it, or set ORIGINAL_SCRIPT=/path/to/it.'
        )
    spec = importlib.util.spec_from_file_location('phase2_original', str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules['phase2_original'] = module
    spec.loader.exec_module(module)
    return module


orig = load_original_module(ORIGINAL_SCRIPT)
DATA_PATH = Path(os.environ.get('DATA_PATH', orig.DATA_PATH)).resolve()

import matplotlib  # noqa: E402  (original module already set the Agg backend)
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402

plt.rcParams['svg.fonttype'] = 'none'   # keep figure text as editable text in the SVG
plt.rcParams['font.family'] = 'DejaVu Sans'


# --------------------------------------------------------------------------------------
# Cohort construction: identical to original main() lines for df_full / df_landmark / ccXX
# --------------------------------------------------------------------------------------
def build_cohorts() -> dict[str, pd.DataFrame]:
    df_raw = pd.read_csv(DATA_PATH)
    orig.validate_columns(df_raw, orig.REQUIRED_COLUMNS)
    df_raw = df_raw.dropna(subset=['mortality_30d', 'survival_days_30']).copy()
    df_raw = orig.derive_variables(df_raw)

    df_full = df_raw[df_raw['lactate_0'] >= 2.0].copy()
    pre48 = (df_full['mortality_30d'] == 1) & (df_full['survival_days_30'] < 2.0)
    df_landmark = df_full[~pre48].copy()
    df_landmark = orig.recalculate_post_landmark_outcomes(df_landmark, landmark_days=2.0)

    cohorts = {}
    for tp, lac in [('12h', 'lactate_12'), ('24h', 'lactate_24'), ('48h', 'lactate_48')]:
        cc = df_landmark.dropna(subset=[lac] + orig.covariates + ['mortality_30d', 'survival_days_30']).copy()
        cc = orig.add_scaled_predictor(cc, f'clearance_{tp}', f'clearance_{tp}_per10', 10.0)
        cohorts[tp] = cc

    sizes = (len(df_full), int(pre48.sum()), len(df_landmark), len(cohorts['12h']), len(cohorts['24h']), len(cohorts['48h']))
    if sizes != (730, 65, 665, 413, 334, 281):
        msg = f'Frozen cohort check failed (full, pre48, landmark, cc12, cc24, cc48) = {sizes}; expected (730, 65, 665, 413, 334, 281)'
        if SKIP_FROZEN_CHECK:
            warnings.warn(msg + '  [SKIP_FROZEN_CHECK=1, continuing - TEST MODE ONLY]')
        else:
            raise RuntimeError(msg)
    cohorts['_sizes'] = sizes
    return cohorts


# --------------------------------------------------------------------------------------
# A. Reduced-parameter Cox vs primary Cox
# --------------------------------------------------------------------------------------
def summarise_exposure(res: pd.DataFrame, exposure: str, tp: str, model_label: str, n_params: int) -> dict:
    row = res.loc[res['term'] == exposure].iloc[0]
    events = int(row['events'])
    return {
        'timepoint': tp,
        'model': model_label,
        'exposure': exposure,
        'n': int(row['n']),
        'events': events,
        'parameters': n_params,
        'EPV': round(events / n_params, 2),
        'hr_per10': row['hr'],
        'hr_ci_lower': row['hr_ci_lower'],
        'hr_ci_upper': row['hr_ci_upper'],
        'p_value': row['p_value'],
        'convergence_status': row['convergence_status'],
    }


def run_reduced_models(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tp in ['12h', '24h', '48h']:
        cc = cohorts[tp]
        exposure = f'clearance_{tp}_per10'

        full_pred = [exposure] + orig.covariates
        red_pred = [exposure] + REDUCED_COVARIATES

        full_res = orig.complete_case_cox_ph(cc, DUR, EVT, full_pred)
        red_res = orig.complete_case_cox_ph(cc, DUR, EVT, red_pred)

        red_res.to_csv(f'SuppTable_ReducedCox_{tp}_FullCoefficients.csv', index=False)
        rows.append(summarise_exposure(full_res, exposure, tp, 'Primary (21 parameters)', len(full_pred)))
        rows.append(summarise_exposure(red_res, exposure, tp, 'Reduced (7 parameters)', len(red_pred)))

    out = pd.DataFrame(rows)
    out.to_csv('SuppTable_ReducedCox_vs_Primary_EPV.csv', index=False)
    return out


# --------------------------------------------------------------------------------------
# B. Restricted cubic spline (Harrell) for 48h clearance
# --------------------------------------------------------------------------------------
def rcs_nonlinear_terms(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell restricted cubic spline: returns the k-2 nonlinear basis columns
    (the linear term is x itself). Scaled by (t_k - t_1)^2 for numerical stability."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = len(t)
    pos3 = lambda u: np.where(u > 0, u, 0.0) ** 3  # noqa: E731
    scale = (t[-1] - t[0]) ** 2
    cols = []
    for j in range(k - 2):
        term = (
            pos3(x - t[j])
            - pos3(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + pos3(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        )
        cols.append(term / scale)
    return np.column_stack(cols)


def full_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack([x, rcs_nonlinear_terms(x, knots)])


def wald_test(beta: np.ndarray, cov: np.ndarray) -> tuple[float, int, float]:
    beta = np.atleast_1d(beta)
    cov = np.atleast_2d(cov)
    stat = float(beta @ np.linalg.solve(cov, beta))
    df = len(beta)
    return stat, df, float(stats.chi2.sf(stat, df))


def fit_spline_model(cc: pd.DataFrame, knots: np.ndarray, adjust_covs: list[str], label: str) -> dict:
    spline_cols = ['clr48_rcs_linear'] + [f'clr48_rcs_nonlin{j + 1}' for j in range(len(knots) - 2)]
    model_df = cc[[DUR, EVT, 'clearance_48h'] + adjust_covs].dropna().copy()
    model_df = model_df.loc[model_df[DUR].astype(float) > 0].copy()
    basis = full_basis(model_df['clearance_48h'].to_numpy(), knots)
    for i, col in enumerate(spline_cols):
        model_df[col] = basis[:, i]
    predictors = spline_cols + adjust_covs
    fit_df = model_df[[DUR, EVT] + predictors].apply(pd.to_numeric, errors='coerce').dropna()

    cph = CoxPHFitter(penalizer=orig.COX_PENALIZER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cph.fit(fit_df, duration_col=DUR, event_col=EVT, show_progress=False)
    warn_msgs = [str(w.message) for w in caught]

    beta_s = cph.params_.loc[spline_cols].to_numpy(dtype=float)
    cov_s = cph.variance_matrix_.loc[spline_cols, spline_cols].to_numpy(dtype=float)

    overall = wald_test(beta_s, cov_s)
    nonlin = wald_test(beta_s[1:], cov_s[1:, 1:])

    events = int(fit_df[EVT].sum())
    n_params = len(predictors)

    cph.summary.reset_index().to_csv(f'SuppTable_RCS48h_{label}_FullCoefficients.csv', index=False)
    return {
        'label': label,
        'cph': cph,
        'beta_s': beta_s,
        'cov_s': cov_s,
        'n': len(fit_df),
        'events': events,
        'parameters': n_params,
        'EPV': round(events / n_params, 2),
        'overall_chi2': overall[0], 'overall_df': overall[1], 'overall_p': overall[2],
        'nonlinear_chi2': nonlin[0], 'nonlinear_df': nonlin[1], 'nonlinear_p': nonlin[2],
        'convergence_status': 'returned_with_warning' if warn_msgs else 'returned_without_warning',
        'warnings': ' | '.join(warn_msgs),
    }


def hr_curve(fit: dict, knots: np.ndarray, grid: np.ndarray, ref: float) -> pd.DataFrame:
    d = full_basis(grid, knots) - full_basis(np.array([ref]), knots)
    log_hr = d @ fit['beta_s']
    se = np.sqrt(np.einsum('ij,jk,ik->i', d, fit['cov_s'], d))
    z = stats.norm.ppf(0.975)
    return pd.DataFrame({
        'clearance_48h_pct': grid,
        'log_hr': log_hr,
        'se_log_hr': se,
        'hr': np.exp(log_hr),
        'hr_ci_lower': np.exp(log_hr - z * se),
        'hr_ci_upper': np.exp(log_hr + z * se),
        'reference_clearance_pct': ref,
        'model': fit['label'],
    })


def plot_spline(curves: list[pd.DataFrame], fits: list[dict], observed: np.ndarray, knots: np.ndarray, ref: float) -> None:
    titles = {
        'Primary_20covariates': 'A. Adjusted for primary covariates',
        'Reduced_6covariates': 'B. Adjusted for reduced covariate set',
    }
    fig, axes = plt.subplots(1, len(curves), figsize=(6.2 * len(curves), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = curves[0]['clearance_48h_pct'].min(), curves[0]['clearance_48h_pct'].max()
    obs_in = observed[(observed >= lo) & (observed <= hi)]
    for ax, curve, fit in zip(axes, curves, fits):
        ax.fill_between(curve['clearance_48h_pct'], curve['hr_ci_lower'], curve['hr_ci_upper'], color='#9ecae1', alpha=0.55, linewidth=0, label='95% CI')
        ax.plot(curve['clearance_48h_pct'], curve['hr'], color='#08519c', linewidth=2.2, label='Hazard ratio')
        ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1)
        ax.axvline(ref, color='#999999', linestyle=':', linewidth=1)
        for kn in knots:
            ax.axvline(kn, color='#cccccc', linewidth=0.8, zorder=0)
        ax.set_yscale('log')
        # rug of observed values, pinned just above the x-axis in axes coordinates
        ax.plot(obs_in, np.full_like(obs_in, 0.03), '|', color='#333333', markersize=8, alpha=0.6,
                transform=ax.get_xaxis_transform())
        ax.yaxis.set_major_locator(FixedLocator(HR_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('48-hour lactate clearance (%)')
        ax.set_title(titles.get(fit['label'], fit['label']), fontsize=11, loc='left')
        ax.text(
            0.98, 0.97,
            f"P overall {fmt_p_eq(fit['overall_p'])}\nP nonlinearity {fmt_p_eq(fit['nonlinear_p'])}\nN = {fit['n']}, events = {fit['events']}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'),
        )
    all_lo = min(c['hr_ci_lower'].min() for c in curves)
    all_hi = max(c['hr_ci_upper'].max() for c in curves)
    axes[0].set_ylim(all_lo / 1.6, all_hi * 1.15)   # headroom below for the rug
    axes[0].set_ylabel(f'Adjusted hazard ratio for 30-day mortality\n(reference: {ref:g}% clearance)')
    fig.tight_layout()
    fig.savefig('SuppFigure_RCS_48hClearance.png', dpi=300)
    fig.savefig('SuppFigure_RCS_48hClearance.svg')
    plt.close(fig)


def fmt_p_eq(p: float) -> str:
    return '< 0.001' if p < 0.001 else f'= {p:.3f}'


def fmt_p(p: float) -> str:
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def run_spline(cohorts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    cc = cohorts['48h']
    x = cc['clearance_48h'].dropna().to_numpy(dtype=float)
    knots = np.percentile(x, RCS_KNOT_PERCENTILES)
    if len(np.unique(np.round(knots, 8))) < len(knots):
        raise RuntimeError(f'Spline knots are not distinct: {knots}')
    lo, hi = np.percentile(x, PLOT_RANGE_PERCENTILES)
    ref = RCS_REF
    if not (lo <= ref <= hi):
        warnings.warn(f'RCS_REF={ref} lies outside the plotted range [{lo:.1f}, {hi:.1f}]; using the median instead.')
        ref = float(np.median(x))
    grid = np.linspace(lo, hi, 201)

    fits = [
        fit_spline_model(cc, knots, orig.covariates, 'Primary_20covariates'),
        fit_spline_model(cc, knots, REDUCED_COVARIATES, 'Reduced_6covariates'),
    ]
    curves = [hr_curve(f, knots, grid, ref) for f in fits]
    pd.concat(curves, ignore_index=True).to_csv('SuppData_RCS48h_HR_Curve.csv', index=False)

    # HRs at fixed clearance values (for the text / a small table)
    vals = np.array([v for v in HR_TABLE_VALUES if lo <= v <= hi], dtype=float)
    fixed = pd.concat([hr_curve(f, knots, vals, ref) for f in fits], ignore_index=True)
    fixed.to_csv('SuppTable_RCS48h_HR_At_Selected_Clearance.csv', index=False)

    tests = pd.DataFrame([{
        'model': f['label'],
        'n': f['n'], 'events': f['events'], 'parameters': f['parameters'], 'EPV': f['EPV'],
        'knots_pct_clearance': '; '.join(f'{k:.1f}' for k in knots),
        'knot_percentiles': '; '.join(str(p) for p in RCS_KNOT_PERCENTILES),
        'reference_clearance_pct': ref,
        'overall_wald_chi2': f['overall_chi2'], 'overall_df': f['overall_df'], 'overall_p': f['overall_p'],
        'nonlinear_wald_chi2': f['nonlinear_chi2'], 'nonlinear_df': f['nonlinear_df'], 'nonlinear_p': f['nonlinear_p'],
        'convergence_status': f['convergence_status'], 'warnings': f['warnings'],
    } for f in fits])
    tests.to_csv('SuppTable_RCS48h_Tests.csv', index=False)

    plot_spline(curves, fits, x, knots, ref)
    return tests, fixed, fits


# --------------------------------------------------------------------------------------
# Plain-text summary for pasting into the response letter
# --------------------------------------------------------------------------------------
def write_summary(sizes, reduced: pd.DataFrame, tests: pd.DataFrame, fixed: pd.DataFrame) -> None:
    def hr_txt(r):
        return f"HR {r['hr_per10']:.3f} (95% CI {r['hr_ci_lower']:.3f}-{r['hr_ci_upper']:.3f}; p={fmt_p(r['p_value'])})"

    lines = [
        'ROUND-3 REVIEWER ADDENDUM - KEY RESULTS',
        f'Run: {datetime.now():%Y-%m-%d %H:%M}   Original script: {ORIGINAL_SCRIPT.name}   Data: {DATA_PATH.name}',
        f'Cox penalizer: {orig.COX_PENALIZER}   Cohort sizes (full, pre48 deaths, landmark, cc12, cc24, cc48): {sizes}',
        '',
        'A. Reduced-parameter Cox (comment 2), per 10% increase in clearance',
    ]
    for _, r in reduced.iterrows():
        lines.append(f"  {r['timepoint']:>3}  {r['model']:<24} N={r['n']}, events={r['events']}, params={r['parameters']}, EPV={r['EPV']}: {hr_txt(r)}  [{r['convergence_status']}]")
    lines += ['', 'B. Restricted cubic spline, 48h clearance (comment 3)']
    for _, t in tests.iterrows():
        lines.append(
            f"  {t['model']}: N={t['n']}, events={t['events']}, params={t['parameters']}, EPV={t['EPV']}; knots at {t['knots_pct_clearance']}%; "
            f"P overall={fmt_p(t['overall_p'])}, P nonlinearity={fmt_p(t['nonlinear_p'])}  [{t['convergence_status']}]"
        )
    lines += ['', f"  Hazard ratios vs {tests['reference_clearance_pct'].iloc[0]:g}% clearance:"]
    for _, r in fixed.iterrows():
        lines.append(f"    {r['model']:<22} clearance {r['clearance_48h_pct']:>6.0f}%: HR {r['hr']:.3f} ({r['hr_ci_lower']:.3f}-{r['hr_ci_upper']:.3f})")
    Path('Round3_Summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_DIR)
    warnings.filterwarnings('default')
    np.random.seed(orig.RANDOM_SEED)

    cohorts = build_cohorts()
    reduced = run_reduced_models(cohorts)
    tests, fixed, _ = run_spline(cohorts)
    pd.DataFrame(orig.COX_FIT_LOG).to_csv('Table_Cox_Fit_Log_Round3.csv', index=False)
    write_summary(cohorts['_sizes'], reduced, tests, fixed)
    print(f'\nRound-3 addendum complete. Outputs written to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()