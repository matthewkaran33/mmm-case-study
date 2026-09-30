"""Shared builders for the Everline Foods (SIMULATED) MMM case study.

Used by 02_fit_meridian.py (fit) and 03_readout.py (reload without refitting).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmm_config as C

from meridian import backend
from meridian import constants
from meridian.data import data_frame_input_data_builder as builder_mod
from meridian.model import prior_distribution as prior_dist_mod
from meridian.model import spec as spec_mod

tfd = backend.tfd


def build_input_data(df: pd.DataFrame):
    df = df.copy()
    df["week"] = pd.to_datetime(df["week"])
    b = builder_mod.DataFrameInputDataBuilder(kpi_type=constants.REVENUE)
    (
        b.with_kpi(df, kpi_col="revenue", time_col="week")
        .with_media(
            df,
            media_cols=C.SPEND_COLS,
            media_spend_cols=C.SPEND_COLS,  # spend as execution metric (no impression data)
            media_channels=C.CHANNELS,
            time_col="week",
        )
        .with_controls(df, control_cols=C.CONTROL_COLS, time_col="week")
    )
    return b.build()


def build_prior() -> prior_dist_mod.PriorDistribution:
    f = np.float64  # backend default float dtype
    return prior_dist_mod.PriorDistribution(
        roi_m=tfd.LogNormal(
            # loc is log(prior MEDIAN), not log(mean); see mmm_config.py
            loc=np.log(C.ROI_PRIOR_MEDIAN).astype(f),
            scale=f(C.ROI_SIGMA),
        ),
        alpha_m=tfd.Beta(
            concentration1=C.DECAY_A.astype(f),
            concentration0=C.DECAY_B.astype(f),
        ),
        ec_m=tfd.LogNormal(
            # loc is log(prior MEDIAN), not log(mean); see mmm_config.py
            loc=np.log(C.EC_PRIOR_MEDIAN).astype(f),
            scale=f(C.EC_SIGMA),
        ),
        slope_m=tfd.LogNormal(
            # loc is log(prior median slope); see mmm_config.py
            loc=f(C.SLOPE_LOG_MEDIAN),
            scale=f(C.SLOPE_SIGMA),
        ),
    )


def build_model_spec() -> spec_mod.ModelSpec:
    return spec_mod.ModelSpec(
        prior=build_prior(),
        media_prior_type="roi",
        max_lag=8,
    )
