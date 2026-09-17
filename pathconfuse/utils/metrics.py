"""
Evaluation Metrics, Bootstrap Confidence Intervals, and Survival Statistics.
"""

import numpy as np
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import pandas as pd


def compute_c_index(y_time, event_observed, risk_scores):
    """
    Computes Harrell's Concordance Index.
    risk_scores: higher value indicates higher hazard / earlier event.
    """
    y_time = np.asarray(y_time)
    event_observed = np.asarray(event_observed)
    risk_scores = np.asarray(risk_scores).flatten()
    return float(concordance_index(y_time, -risk_scores, event_observed))


def bootstrap_c_index_ci(y_time, event_observed, risk_scores, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    Computes 95% patient-level bootstrap confidence interval for Harrell's C-index.
    """
    y_time = np.asarray(y_time)
    event_observed = np.asarray(event_observed)
    risk_scores = np.asarray(risk_scores).flatten()
    n = len(y_time)

    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Check that there are at least 2 events in the bootstrap sample
        if np.sum(event_observed[idx]) >= 2:
            try:
                c = concordance_index(y_time[idx], -risk_scores[idx], event_observed[idx])
                scores.append(c)
            except Exception:
                continue

    if len(scores) < 10:
        c_base = compute_c_index(y_time, event_observed, risk_scores)
        return c_base, c_base, c_base

    low = float(np.percentile(scores, 100 * (alpha / 2)))
    high = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    mean_val = float(np.mean(scores))
    return mean_val, low, high


def logrank_analysis(y_time, event_observed, risk_scores):
    """
    Stratifies patients into High-risk vs Low-risk by median risk score,
    and performs the two-sided log-rank test and univariate Cox PH hazard ratio.
    """
    y_time = np.asarray(y_time)
    event_observed = np.asarray(event_observed)
    risk_scores = np.asarray(risk_scores).flatten()

    median_risk = np.median(risk_scores)
    group_high = (risk_scores >= median_risk).astype(int)

    # Log-rank test
    results = logrank_test(
        y_time[group_high == 1],
        y_time[group_high == 0],
        event_observed_A=event_observed[group_high == 1],
        event_observed_B=event_observed[group_high == 0]
    )
    p_value = float(results.p_value)

    # Cox Hazard Ratio
    df = pd.DataFrame({
        "time": y_time,
        "event": event_observed,
        "group": group_high
    })
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col="time", event_col="event")
        hr = float(cph.hazard_ratios_["group"])
        hr_ci = (float(cph.confidence_intervals_.loc["group"].iloc[0]),
                 float(cph.confidence_intervals_.loc["group"].iloc[1]))
    except Exception:
        hr = 1.0
        hr_ci = (1.0, 1.0)

    return {
        "logrank_p": p_value,
        "hazard_ratio": hr,
        "hr_ci_lower": hr_ci[0],
        "hr_ci_upper": hr_ci[1]
    }
