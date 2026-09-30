"""Shared configuration for the Everline Foods (SIMULATED) MMM case study."""
import numpy as np

# --- channels (order matters everywhere) ------------------------------------
CHANNELS = ["Paid Search", "Paid Social", "TV", "Display", "Online Video"]
SPEND_COLS = [
    "spend_paid_search",
    "spend_paid_social",
    "spend_tv",
    "spend_display",
    "spend_video",
]
CONTROL_COLS = ["price", "promo"]

# --- informed priors ("domain knowledge", not the simulation truth) ---------
# roi_mu: reasonable industry guesses — search/video most efficient,
# display prospecting near breakeven. Wide sigma lets the data speak.
ROI_MU = np.array([4.0, 3.0, 2.5, 1.2, 3.5])
ROI_SIGMA = 0.8

# adstock decay: TV carries over longest, search/display shortest.
# Beta(a, b) with mean a/(a+b).
DECAY_A = np.array([3.0, 5.0, 7.0, 2.0, 5.5])
DECAY_B = np.array([7.0, 5.0, 3.0, 8.0, 4.5])

# half-saturation ec: scaled to typical weekly spend levels per channel.
# v2: TIGHTENED (sigma 0.9 -> 0.45). In v1 the wide ec prior let search's
# response curve extrapolate wildly past observed spend (unidentified Hill
# region). The tighter prior keeps saturation inside plausible spend levels
# while still letting the data move it ~1.5x either way.
EC_MU = np.array([25000.0, 20000.0, 100000.0, 8000.0, 18000.0])
EC_SIGMA = 0.45

# hill slope: diminishing returns kick in gradually; modestly informative.
# v2: tightened (0.6 -> 0.35) for the same identification reason.
SLOPE_MU = np.log(2.0)
SLOPE_SIGMA = 0.35

# --- sampling ----------------------------------------------------------------
SEED = 123
# v2: longer MCMC for anything public-facing — 4 chains, deeper warmup, more draws.
N_CHAINS = 4
N_ADAPT = 1000
N_BURNIN = 1000
N_KEEP = 2000

# --- paths -------------------------------------------------------------------
DATA_CSV = "data/simulated_mmm_weekly.csv"
INFERENCE_NC = "data/inference_data.nc"
CHART_DIR = "charts"
