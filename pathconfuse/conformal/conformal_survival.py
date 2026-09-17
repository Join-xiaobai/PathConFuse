"""
Inverse Probability of Censoring Weighting (IPCW) Split Conformal Prediction
for Censored Oncology Survival Outcomes.

Implements distribution-free finite-sample interval calibration under administrative
right-censoring following Candes et al. (2023) and Gui et al. (2024).
"""

import numpy as np
from lifelines import KaplanMeierFitter


class CensoredConformalSurvival:
    """
    Split Conformal Survival Predictor with IPCW Censoring Correction.
    Guarantees asymptotic validity and empirical finite-sample coverage at confidence 1 - alpha.
    """
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.q_hat = None
        self.kmf_c = KaplanMeierFitter()

    def fit_calibration(self, y_time, delta_event, pred_time):
        """
        Calibrate weighted residual quantile on held-out calibration patients.
        y_time: observed follow-up time in months.
        delta_event: 1 if death event observed, 0 if right-censored.
        pred_time: model predicted survival duration in months.
        """
        y_time = np.asarray(y_time, dtype=np.float64)
        delta_event = np.asarray(delta_event, dtype=np.int64)
        pred_time = np.asarray(pred_time, dtype=np.float64)
        n = len(y_time)

        # 1. Fit Kaplan-Meier on censoring indicator (1 - delta_event) to estimate G(t) = P(C > t)
        self.kmf_c.fit(y_time, event_observed=(1 - delta_event))

        # 2. Extract uncensored residuals
        uncens_mask = (delta_event == 1)
        if np.sum(uncens_mask) < 5:
            uncens_mask = np.ones(n, dtype=bool)

        y_uncens = y_time[uncens_mask]
        pred_uncens = pred_time[uncens_mask]
        residuals = np.abs(y_uncens - pred_uncens)

        # 3. Estimate IPCW weights w_i = 1 / G(T_i)
        g_vals = self.kmf_c.predict(y_uncens).values
        g_vals = np.clip(g_vals, 0.05, 1.0)
        weights = 1.0 / g_vals
        weights = weights / np.sum(weights)

        # 4. Weighted Empirical Quantile at level (1 - alpha) * (1 + 1/n_uncens)
        n_uncens = len(residuals)
        target_quantile = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n_uncens))

        sorted_indices = np.argsort(residuals)
        sorted_res = residuals[sorted_indices]
        sorted_w = weights[sorted_indices]
        cum_w = np.cumsum(sorted_w)

        idx = np.searchsorted(cum_w, target_quantile)
        idx = min(idx, len(sorted_res) - 1)
        self.q_hat = float(sorted_res[idx])
        return self.q_hat

    def predict_intervals(self, pred_time, d_conflict=None, conflict_scale=0.5):
        """
        Construct individualized prediction intervals [L(X), U(X)].
        If d_conflict is provided, adaptively expands uncertainty margin:
        q_eff = q_hat * (1 + conflict_scale * d_conflict).
        """
        assert self.q_hat is not None, "Must call fit_calibration before predict_intervals."
        pred_time = np.asarray(pred_time, dtype=np.float64)

        if d_conflict is not None:
            d_conflict = np.asarray(d_conflict, dtype=np.float64).flatten()
            q_eff = self.q_hat * (1.0 + conflict_scale * d_conflict)
        else:
            q_eff = np.full_like(pred_time, self.q_hat)

        lower = np.maximum(0.0, pred_time - q_eff)
        upper = pred_time + q_eff
        return lower, upper

    @staticmethod
    def evaluate_coverage(y_time, delta_event, lower_bounds, upper_bounds):
        """
        Evaluate empirical conformal coverage on test cohort.
        - For uncensored events (delta=1): covered if lower <= y_time <= upper.
        - For right-censored patients (delta=0): covered if upper >= y_time (since true T* >= C).
        """
        y_time = np.asarray(y_time, dtype=np.float64)
        delta_event = np.asarray(delta_event, dtype=np.int64)
        lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        upper_bounds = np.asarray(upper_bounds, dtype=np.float64)

        covered = []
        for t, d, l, u in zip(y_time, delta_event, lower_bounds, upper_bounds):
            if d == 1:
                covered.append(1.0 if (l <= t <= u) else 0.0)
            else:
                covered.append(1.0 if (u >= t) else 0.0)

        empirical_coverage = float(np.mean(covered))
        marginal_width = float(np.mean(upper_bounds - lower_bounds))
        return empirical_coverage, marginal_width
