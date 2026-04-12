"""Unit tests for src/evaluate.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import (
    christoffersen_cc,
    compare_vol_forecasts,
    compute_var_t,
    evaluate_point_forecast,
    evaluate_vol_forecast,
    kupiec_pof,
    var_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def date_index():
    return pd.date_range("2021-01-01", periods=100, freq="B")


@pytest.fixture()
def returns(date_index):
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 0.01, size=len(date_index)), index=date_index)


@pytest.fixture()
def rolling_df(date_index):
    """Minimal rolling forecast DataFrame with constant volatility."""
    vol = 0.009
    return pd.DataFrame(
        {
            "predicted_volatility": vol,
            "predicted_variance": vol ** 2,
            "realized_return": np.random.default_rng(7).normal(0, 0.01, len(date_index)),
        },
        index=date_index,
    )


# ---------------------------------------------------------------------------
# 1. evaluate_point_forecast
# ---------------------------------------------------------------------------

class TestEvaluatePointForecast:
    def test_perfect_forecast_gives_zero_errors(self, date_index):
        s = pd.Series(np.arange(1.0, 11.0), index=date_index[:10])
        result = evaluate_point_forecast(s, s.copy())
        assert result["MAE"] == pytest.approx(0.0)
        assert result["RMSE"] == pytest.approx(0.0)
        assert result["MAPE (%)"] == pytest.approx(0.0)

    def test_known_values(self, date_index):
        actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=date_index[:4])
        predicted = pd.Series([1.5, 2.5, 3.5, 4.5], index=date_index[:4])
        result = evaluate_point_forecast(actual, predicted)
        assert result["MAE"] == pytest.approx(0.5)
        assert result["RMSE"] == pytest.approx(0.5)

    def test_alignment_by_index(self, date_index):
        actual = pd.Series([1.0, 2.0, 3.0], index=date_index[:3])
        predicted = pd.Series([1.0, 2.0, 3.0, 4.0], index=date_index[:4])
        result = evaluate_point_forecast(actual, predicted)
        assert result["MAE"] == pytest.approx(0.0)

    def test_no_overlap_raises(self, date_index):
        actual = pd.Series([1.0], index=date_index[:1])
        other_index = pd.date_range("2050-01-01", periods=1, freq="B")
        predicted = pd.Series([1.0], index=other_index)
        with pytest.raises(ValueError, match="no common index"):
            evaluate_point_forecast(actual, predicted)

    def test_returns_dict_with_expected_keys(self, date_index):
        s = pd.Series(np.linspace(1, 10, 5), index=date_index[:5])
        result = evaluate_point_forecast(s, s * 1.01)
        assert set(result.keys()) == {"MAE", "RMSE", "MAPE (%)"}


# ---------------------------------------------------------------------------
# 2. evaluate_vol_forecast
# ---------------------------------------------------------------------------

class TestEvaluateVolForecast:
    def test_returns_expected_keys(self, rolling_df, returns):
        result = evaluate_vol_forecast(rolling_df, returns, model_name="test")
        assert set(result.keys()) == {"model", "RMSE", "MAE", "QLIKE", "Corr", "n_obs"}

    def test_model_name_preserved(self, rolling_df, returns):
        result = evaluate_vol_forecast(rolling_df, returns, model_name="GARCH")
        assert result["model"] == "GARCH"

    def test_n_obs_equals_overlap(self, rolling_df, returns):
        result = evaluate_vol_forecast(rolling_df, returns)
        assert result["n_obs"] == len(rolling_df.index.intersection(returns.index))

    def test_missing_columns_raises(self, rolling_df, returns):
        bad_df = rolling_df.drop(columns=["predicted_variance"])
        with pytest.raises(ValueError, match="missing required columns"):
            evaluate_vol_forecast(bad_df, returns)

    def test_no_overlap_raises(self, rolling_df):
        other_returns = pd.Series(
            [0.01] * 5, index=pd.date_range("2050-01-01", periods=5, freq="B")
        )
        with pytest.raises(ValueError, match="no common index"):
            evaluate_vol_forecast(rolling_df, other_returns)

    def test_rmse_mae_positive(self, rolling_df, returns):
        result = evaluate_vol_forecast(rolling_df, returns)
        assert result["RMSE"] >= 0
        assert result["MAE"] >= 0


# ---------------------------------------------------------------------------
# 3. compare_vol_forecasts
# ---------------------------------------------------------------------------

class TestCompareVolForecasts:
    def test_returns_dataframe(self, rolling_df, returns):
        result = compare_vol_forecasts({"A": rolling_df, "B": rolling_df}, returns)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["A", "B"]

    def test_identical_models_identical_metrics(self, rolling_df, returns):
        result = compare_vol_forecasts({"A": rolling_df, "B": rolling_df.copy()}, returns)
        assert result.loc["A", "RMSE"] == pytest.approx(result.loc["B", "RMSE"])


# ---------------------------------------------------------------------------
# 4. compute_var_t
# ---------------------------------------------------------------------------

class TestComputeVarT:
    def test_var_positive_for_standard_alpha(self, rolling_df):
        var = compute_var_t(rolling_df["predicted_volatility"], nu=10.0, alpha=0.01)
        assert (var > 0).all()

    def test_var_larger_for_smaller_alpha(self, rolling_df):
        """1% VaR should be larger (more extreme) than 5% VaR."""
        vol = rolling_df["predicted_volatility"]
        var1 = compute_var_t(vol, nu=10.0, alpha=0.01)
        var5 = compute_var_t(vol, nu=10.0, alpha=0.05)
        assert (var1 > var5).all()

    def test_invalid_nu_raises(self, rolling_df):
        with pytest.raises(ValueError, match="nu must be > 2"):
            compute_var_t(rolling_df["predicted_volatility"], nu=1.5, alpha=0.01)

    def test_invalid_alpha_raises(self, rolling_df):
        with pytest.raises(ValueError, match="alpha must be in"):
            compute_var_t(rolling_df["predicted_volatility"], nu=10.0, alpha=1.5)

    def test_heavier_tail_gives_larger_var(self, rolling_df):
        """Lower nu (heavier tails) ⟹ larger 1% VaR."""
        vol = rolling_df["predicted_volatility"]
        var_heavy = compute_var_t(vol, nu=4.0, alpha=0.01)
        var_light = compute_var_t(vol, nu=30.0, alpha=0.01)
        assert (var_heavy > var_light).all()


# ---------------------------------------------------------------------------
# 5. kupiec_pof
# ---------------------------------------------------------------------------

class TestKupiecPof:
    def test_perfect_calibration_high_pvalue(self):
        # x/T ≈ alpha → should not reject H0
        T, alpha = 1000, 0.01
        x = 10  # exactly 1%
        result = kupiec_pof(T, x, alpha)
        assert result["p_value"] > 0.05

    def test_gross_miscalibration_low_pvalue(self):
        # 10% breaches when 1% expected → strong rejection
        T, alpha = 1000, 0.01
        x = 100
        result = kupiec_pof(T, x, alpha)
        assert result["p_value"] < 0.001

    def test_zero_breaches(self):
        result = kupiec_pof(100, 0, 0.01)
        assert result["LR"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_all_breaches(self):
        result = kupiec_pof(100, 100, 0.01)
        assert result["LR"] > 0

    def test_returns_lr_and_pvalue(self):
        result = kupiec_pof(250, 3, 0.01)
        assert "LR" in result and "p_value" in result
        assert result["LR"] >= 0


# ---------------------------------------------------------------------------
# 6. var_backtest
# ---------------------------------------------------------------------------

class TestVarBacktest:
    def test_returns_dataframe(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0)
        assert isinstance(result, pd.DataFrame)

    def test_default_two_alpha_levels(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0)
        assert set(result["alpha"]) == {0.01, 0.05}

    def test_custom_alpha_levels(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0, alpha_levels=[0.01])
        assert len(result) == 1
        assert result["alpha"].iloc[0] == 0.01

    def test_breach_rate_within_range(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0)
        assert (result["breach_rate"] >= 0).all()
        assert (result["breach_rate"] <= 1).all()

    def test_model_name_in_output(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0, model_name="TestModel")
        assert (result["model"] == "TestModel").all()

    def test_missing_column_raises(self, rolling_df, returns):
        bad_df = rolling_df.drop(columns=["predicted_volatility"])
        with pytest.raises(ValueError, match="missing required columns"):
            var_backtest(bad_df, returns, nu=10.0)


# ---------------------------------------------------------------------------
# 7. christoffersen_cc
# ---------------------------------------------------------------------------

class TestChristoffersenCC:
    def test_returns_dict_with_required_keys(self):
        hits = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)
        result = christoffersen_cc(hits, alpha=0.01)
        for key in ("n_00", "n_01", "n_10", "n_11", "pi_01", "pi_11",
                    "LR_uc", "p_uc", "LR_ind", "p_ind", "LR_cc", "p_cc"):
            assert key in result

    def test_transition_counts_sum_to_T_minus_1(self):
        hits = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0], dtype=float)
        r = christoffersen_cc(hits, alpha=0.01)
        assert r["n_00"] + r["n_01"] + r["n_10"] + r["n_11"] == len(hits) - 1

    def test_known_transition_counts(self):
        # hits = [0,1,0,1,0,1] → pairs: (0,1),(1,0),(0,1),(1,0),(0,1)
        # n_00=0, n_01=3, n_10=2, n_11=0
        hits = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        r = christoffersen_cc(hits, alpha=0.5)
        assert r["n_00"] == 0
        assert r["n_01"] == 3
        assert r["n_10"] == 2
        assert r["n_11"] == 0

    def test_all_zeros_no_breach(self):
        hits = np.zeros(100, dtype=float)
        r = christoffersen_cc(hits, alpha=0.01)
        assert r["n_01"] == 0
        assert r["n_11"] == 0
        assert r["pi_01"] == 0.0
        assert r["pi_11"] == 0.0

    def test_all_ones_all_breach(self):
        hits = np.ones(100, dtype=float)
        r = christoffersen_cc(hits, alpha=0.99)
        assert r["n_00"] == 0
        assert r["n_10"] == 0

    def test_p_values_in_unit_interval(self):
        rng = np.random.default_rng(0)
        hits = (rng.uniform(size=200) < 0.01).astype(float)
        r = christoffersen_cc(hits, alpha=0.01)
        assert 0 <= r["p_uc"] <= 1
        assert 0 <= r["p_ind"] <= 1
        assert 0 <= r["p_cc"] <= 1

    def test_lr_cc_equals_lr_uc_plus_lr_ind(self):
        rng = np.random.default_rng(1)
        hits = (rng.uniform(size=300) < 0.05).astype(float)
        r = christoffersen_cc(hits, alpha=0.05)
        assert abs(r["LR_cc"] - (r["LR_uc"] + r["LR_ind"])) < 1e-3  # rounding to 4dp

    def test_well_calibrated_model_does_not_reject(self):
        # With ~1% breach rate, CC test should not reject at 5%
        rng = np.random.default_rng(42)
        hits = (rng.uniform(size=1200) < 0.01).astype(float)
        r = christoffersen_cc(hits, alpha=0.01)
        assert r["p_cc"] > 0.05

    def test_perfectly_clustered_breaches_rejects_independence(self):
        # 50 consecutive breaches then 150 non-breaches → strong clustering
        hits = np.array([1.0] * 50 + [0.0] * 150)
        r = christoffersen_cc(hits, alpha=0.25)
        # pi_11 >> pi_01 → independence rejected
        assert r["p_ind"] < 0.05

    def test_accepts_pandas_series(self):
        s = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=float)
        r = christoffersen_cc(s, alpha=0.01)
        assert isinstance(r, dict)

    def test_invalid_alpha_raises(self):
        hits = np.zeros(50, dtype=float)
        with pytest.raises(ValueError, match="alpha"):
            christoffersen_cc(hits, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            christoffersen_cc(hits, alpha=1.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            christoffersen_cc(np.array([1.0]), alpha=0.01)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            christoffersen_cc(np.array([]), alpha=0.01)

    def test_lr_statistics_non_negative(self):
        rng = np.random.default_rng(99)
        hits = (rng.uniform(size=500) < 0.05).astype(float)
        r = christoffersen_cc(hits, alpha=0.05)
        assert r["LR_uc"] >= 0
        assert r["LR_cc"] >= 0


# ---------------------------------------------------------------------------
# 8. var_backtest – CC columns
# ---------------------------------------------------------------------------

class TestVarBacktestCC:
    def test_cc_columns_present(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0)
        for col in ("LR_ind", "p_ind", "reject_ind", "LR_cc", "p_cc", "reject_cc"):
            assert col in result.columns

    def test_reject_cc_is_bool(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0)
        assert result["reject_cc"].dtype == bool

    def test_p_cc_in_unit_interval(self, rolling_df, returns):
        result = var_backtest(rolling_df, returns, nu=10.0)
        assert (result["p_cc"] >= 0).all()
        assert (result["p_cc"] <= 1).all()
