"""
Fit a Meridian MMM on the simulated Everline Foods weekly dataset.

SIMULATED DATA — not real company data.

Steps:
  1. Load data, build Meridian InputData (national model).
  2. Set informed priors (see mmm_config.py).
  3. Sample prior + posterior (MCMC).
  4. Save inference data (data/inference_data.nc) for the readout step.
  5. Write convergence + posterior-predictive diagnostics to charts/.

Run from the repo root:  python scripts/02_fit_meridian.py
"""
import os
import sys

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import mmm_config as C
import mmm_lib

from meridian.model import model as model_mod
from meridian.analysis import analyzer as analyzer_mod


def main():
    os.makedirs(C.CHART_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(C.INFERENCE_NC), exist_ok=True)

    df = pd.read_csv(C.DATA_CSV)
    print(f"loaded {len(df)} weeks of SIMULATED data")

    input_data = mmm_lib.build_input_data(df)
    print("input data built:",
          f"kpi shape={dict(input_data.kpi.sizes)}",
          f"media channels={input_data.media.coords['media_channel'].values.tolist()}")

    model_spec = mmm_lib.build_model_spec()
    mmm = model_mod.Meridian(input_data=input_data, model_spec=model_spec)

    print("sampling prior (500 draws)...")
    mmm.sample_prior(n_draws=500, seed=C.SEED)

    print(f"sampling posterior: chains={C.N_CHAINS}, "
          f"adapt={C.N_ADAPT}, burnin={C.N_BURNIN}, keep={C.N_KEEP} ...")
    mmm.sample_posterior(
        n_chains=C.N_CHAINS,
        n_adapt=C.N_ADAPT,
        n_burnin=C.N_BURNIN,
        n_keep=C.N_KEEP,
        seed=C.SEED,
    )

    az.to_netcdf(mmm.inference_data, C.INFERENCE_NC)
    print(f"saved inference data -> {C.INFERENCE_NC}")

    # ---- diagnostics -------------------------------------------------------
    analyzer = analyzer_mod.Analyzer(model_context=mmm.model_context,
                                     inference_data=mmm.inference_data)

    rhat = analyzer.rhat_summary()
    print("\n--- R-hat summary ---")
    print(rhat.to_string() if hasattr(rhat, "to_string") else rhat)

    pa = analyzer.predictive_accuracy()
    print("\n--- predictive accuracy ---")
    print(pa)

    # Charts are built with matplotlib in 03_readout.py (Meridian's built-in
    # visualizers return Altair objects here). Quick ROI point estimates:
    roi = analyzer.roi()
    print("\n--- posterior mean ROI by channel ---")
    print(np.asarray(roi).reshape(-1, len(C.CHANNELS)).mean(axis=0))

    print("\nDONE. Next: python scripts/03_readout.py")


if __name__ == "__main__":
    main()
