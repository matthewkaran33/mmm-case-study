"""
Synthetic transportability check for the Everline Foods (SIMULATED) MMM case
study.

SIMULATED DATA - not real company data.

What it does:
  1. Takes the national v2 posterior (fit on national weekly data) as given.
  2. Applies the fitted national response-curve shape to a SEPARATELY
     simulated 12-geo dataset (same true parameters, geo-sized saturation).
  3. Scores two regions of that dataset:
       - EVAL region  = geo_04 + geo_09 (2 geos)
       - OTHER region = the remaining 10 geos
  4. Compares predicted vs true per-channel incremental revenue per
     posterior draw.

What it is NOT: a holdout validation. The model was never trained on ANY of
the 12 geos - not the 2 eval geos and not the other 10 - so nothing here is
"unseen" relative to a geo training set. This tests whether the national
curve shape transfers to a separately simulated geo dataset where the truth
is known. A real holdout would fit a geo-level model on 10 geos and evaluate
on the excluded 2; a real geo test would randomize the holdout.

Scale note: saturation is scale-dependent (a $10k/week geo and the $3M/week
nation sit at different points on the absolute spend curve), so region media
is rescaled to national-equivalent before going through the fitted model,
and predicted incremental is scaled back. Without that, the comparison would
confound curve-shape error with a pure scale mismatch.

Run from the repo root (after 02_fit_meridian.py):
  python scripts/04_transportability_check.py
"""
import json
import os
import sys

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import mmm_config as C  # noqa: E402
import mmm_lib  # noqa: E402
from meridian import backend  # noqa: E402
from meridian.analysis import tensors as tensors_mod  # noqa: E402
from meridian.model import model as model_mod  # noqa: E402
from meridian.analysis import analyzer as analyzer_mod  # noqa: E402

SIM_FOOTNOTE = "Simulated data - 'Everline Foods' is a fictional brand."
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold"})
PALETTE = ["#2a6f97", "#61a5c2", "#f4a259", "#bc4b51", "#5b8e7d"]
GEO_CSV = "data/simulated_mmm_geo.csv"


def load_national():
    df = pd.read_csv(C.DATA_CSV, parse_dates=["week"])
    input_data = mmm_lib.build_input_data(df)
    idata = az.from_netcdf(C.INFERENCE_NC)
    mmm = model_mod.Meridian(input_data=input_data,
                             model_spec=mmm_lib.build_model_spec(),
                             inference_data=idata)
    an = analyzer_mod.Analyzer(model_context=mmm.model_context,
                               inference_data=idata)
    return an


def region_media(df_geo, geos, share):
    """Aggregate a set of geos into national-equivalent media.

    Sums the geos' spends, then divides by the region's size share so the
    series sits at national scale - the scale the model's saturation curve
    was estimated at. (Saturation is scale-dependent; see module docstring.)
    """
    sub = df_geo[df_geo["geo"].isin(geos)].sort_values(["week", "geo"])
    piv = sub.groupby("week")[C.SPEND_COLS].sum().reset_index()
    arr = piv[C.SPEND_COLS].to_numpy(dtype=float) / share  # (104, 5)
    return backend.to_tensor(arr[np.newaxis, :, :], dtype=backend.float_dtype)


def main():
    os.makedirs(C.CHART_DIR, exist_ok=True)
    an = load_national()
    df_geo = pd.read_csv(GEO_CSV, parse_dates=["week"])
    with open("data/dgp_truth_geo.yaml") as f:
        gtruth = yaml.safe_load(f)

    eval_geos = gtruth["holdout_geos"]
    other_geos = [g for g in gtruth["geos"] if g not in eval_geos]
    key = {"Paid Search": "paid_search", "Paid Social": "paid_social",
           "TV": "tv", "Display": "display", "Online Video": "video"}

    results = {}
    for name, geos in [("eval", eval_geos), ("other", other_geos)]:
        share = sum(gtruth["geos"][g]["share"] for g in geos)
        print(f"computing per-draw incremental outcomes for {name} region "
              f"({len(geos)} geos, share={share:.2f}) ...")
        inc = np.asarray(an.incremental_outcome(
            new_data=tensors_mod.DataTensors(media=region_media(df_geo, geos, share)),
            aggregate_geos=True, aggregate_times=True))
        # predictions are at national scale; scale back to the region
        draws = inc.reshape(-1, inc.shape[-1]) * share  # (n_draws, 5)
        rows = []
        for i, ch in enumerate(C.CHANNELS):
            truth_c = sum(gtruth["geos"][g]["total_true_contribution"][key[ch]] for g in geos)
            rows.append({
                "channel": ch,
                "true_incremental": float(truth_c),
                "pred_mean": float(draws[:, i].mean()),
                "pred_lo": float(np.percentile(draws[:, i], 5)),
                "pred_hi": float(np.percentile(draws[:, i], 95)),
                "in_90ci": bool(np.percentile(draws[:, i], 5) <= truth_c
                                <= np.percentile(draws[:, i], 95)),
            })
        results[name] = rows
        hit = sum(r["in_90ci"] for r in rows)
        print(f"  {name}: {hit}/5 channels' truth inside 90% CI")

    # ---- chart: predicted vs true (eval region) ----------------------------
    h = pd.DataFrame(results["eval"])
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    xs = h["true_incremental"] / 1e6
    ys = h["pred_mean"] / 1e6
    yerr = [(h["pred_mean"] - h["pred_lo"]) / 1e6, (h["pred_hi"] - h["pred_mean"]) / 1e6]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", ms=8, capsize=5, color="#2a6f97",
                ecolor="#2a6f97", elinewidth=1.6)
    for _, r in h.iterrows():
        ax.annotate(r["channel"], (r["true_incremental"] / 1e6, r["pred_mean"] / 1e6),
                    xytext=(7, 5), textcoords="offset points", fontsize=9)
    lim = [0, max(xs.max(), ys.max()) * 1.15]
    ax.plot(lim, lim, ls="--", color="#999", lw=1.2, label="Perfect recovery (45°)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True incremental revenue, eval geos ($M) - known from simulation")
    ax.set_ylabel("Model-predicted incremental revenue ($M, posterior mean ± 90% CI)")
    ax.set_title("Synthetic transportability check: national response-curve\n"
                 "shape applied to a separately simulated geo dataset")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.01, -0.20,
            SIM_FOOTNOTE + " Eval-region media rescaled to national-equivalent "
            "before prediction (saturation is scale-dependent). The model was "
            "fit on national data only - never on these geos. Curve-shape "
            "transport test on simulated data, not a randomized experiment.",
            transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(f"{C.CHART_DIR}/09_transportability_check.png", bbox_inches="tight")
    plt.close(fig)
    print("chart written: charts/09_transportability_check.png")

    with open("data/transportability_check.json", "w") as f:
        json.dump({"note": ("SIMULATED DATA - Everline Foods is fictional. "
                            "Synthetic transportability check: the national v2 "
                            "posterior (fit on national weekly data) applied "
                            "to a separately simulated 12-geo dataset. Not a "
                            "holdout validation - the model never trained on "
                            "any of the 12 geos."),
                   "eval_geos": eval_geos, "results": results}, f, indent=2)
    print("saved data/transportability_check.json")

    print("\n=== eval region: predicted vs true incremental ===")
    print(h.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
