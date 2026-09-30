"""
Read-out for the Everline Foods (SIMULATED) MMM case study - v2.

SIMULATED DATA - not real company data.

Reloads the fitted Meridian model (no refitting) and produces:
  1. Channel contribution & ROI table (posterior means + 90% CI)
  2. Calibration check against the simulation ground truth
  3. Budget scenario: shift 15% of display spend into online video,
     with uncertainty from FULL posterior draws (not interpolated curve CIs)
  4. Matplotlib charts in charts/ (v1 charts archived under charts/v1_before/)
  5. Headline numbers saved to data/readout.json

v2 rebuild log (everything that changed vs v1 - this is a rebuild, not a
controlled experiment, so the before/after charts cannot isolate any single
change):
  - simulated dataset: TV flights staggered against promo weeks (v1 ran
    every flight on top of promo weeks); new random seed
  - saturation priors tightened (ec sigma 0.9 -> 0.45, slope sigma 0.6 -> 0.35)
  - MCMC config: 2 chains x 500 kept draws -> 4 chains x 2000
  - response curves are shown only within ~observed spend range (<=1.25x),
    not extrapolated to 1.5x
  - new chart 06: search response curve v1 vs v2 rebuild (multiple things
    changed - do not read the difference as the effect of the priors alone)
  - new charts 07/08: prior-vs-posterior for ROI and half-saturation (ec)
  - scenario interval comes from propagating every posterior draw through
    Meridian's own incremental_outcome() on scenario media

Run from the repo root:  python scripts/03_readout.py
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
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold",
})
PALETTE = ["#2a6f97", "#61a5c2", "#f4a259", "#bc4b51", "#5b8e7d"]
# response curves are only presented inside observed spend variation
RC_MULTS = np.linspace(0.5, 1.25, 16)
RC_MULTS_FULL = np.linspace(0.5, 1.5, 21)  # for the v1/v2 rebuild comparison only


def load():
    df = pd.read_csv(C.DATA_CSV, parse_dates=["week"])
    input_data = mmm_lib.build_input_data(df)
    idata = az.from_netcdf(C.INFERENCE_NC)
    mmm = model_mod.Meridian(
        input_data=input_data,
        model_spec=mmm_lib.build_model_spec(),
        inference_data=idata,
    )
    an = analyzer_mod.Analyzer(model_context=mmm.model_context,
                               inference_data=idata)
    return df, an, idata


def posterior_summary(an):
    """Per-channel posterior summary from summary_metrics()."""
    sm = an.summary_metrics()
    post = sm.sel(distribution="posterior")
    channels = [c for c in sm.channel.values.tolist() if c != "All Channels"]
    rows = []
    for ch in channels:
        rows.append({
            "channel": ch,
            "spend": float(sm["spend"].sel(channel=ch)),
            "pct_spend": float(sm["pct_of_spend"].sel(channel=ch)),
            "incremental_revenue": float(post["incremental_outcome"].sel(channel=ch, metric="mean")),
            "incr_lo": float(post["incremental_outcome"].sel(channel=ch, metric="ci_lo")),
            "incr_hi": float(post["incremental_outcome"].sel(channel=ch, metric="ci_hi")),
            "pct_contribution": float(post["pct_of_contribution"].sel(channel=ch, metric="mean")),
            "roi_mean": float(post["roi"].sel(channel=ch, metric="mean")),
            "roi_lo": float(post["roi"].sel(channel=ch, metric="ci_lo")),
            "roi_hi": float(post["roi"].sel(channel=ch, metric="ci_hi")),
            "mroi_mean": float(post["mroi"].sel(channel=ch, metric="mean")),
        })
    return pd.DataFrame(rows)


def calibration(df_summary):
    """Compare posterior ROI against the simulation ground truth."""
    with open("data/dgp_truth.yaml") as f:
        truth = yaml.safe_load(f)
    key = {"Paid Search": "paid_search", "Paid Social": "paid_social",
           "TV": "tv", "Display": "display", "Online Video": "video"}
    rows = []
    for _, r in df_summary.iterrows():
        t = truth["channels"][key[r["channel"]]]
        true_roi = t["total_true_contribution"] / t["total_spend"]
        rows.append({"channel": r["channel"], "true_roi": true_roi,
                     "post_roi": r["roi_mean"],
                     "in_90ci": r["roi_lo"] <= true_roi <= r["roi_hi"]})
    return pd.DataFrame(rows)


def budget_scenario_full_posterior(an, df):
    """Shift 15% of display spend into online video, full-posterior version.

    Every posterior draw flows through Meridian's own incremental_outcome()
    evaluated on scenario media (passed as new_data), so the net-gain
    distribution reflects the joint uncertainty in ROI, adstock, and
    saturation - not an interpolation of curve CIs.

    net_d = [video(scenario) - video(baseline)] - [display(baseline) - display(scenario)]
    """
    spend = dict(zip(df_summary_channel_order(), [float(df[c].sum()) for c in C.SPEND_COLS]))
    shift_amt = 0.15 * spend["Display"]
    m_video = 1 + shift_amt / spend["Online Video"]

    base_media = df[C.SPEND_COLS].to_numpy(dtype=float)  # (104, 5) original dollars
    scen_media = base_media.copy()
    vi = C.CHANNELS.index("Online Video")
    di = C.CHANNELS.index("Display")
    scen_media[:, vi] *= m_video
    scen_media[:, di] *= 0.85

    def to_t(a):
        return backend.to_tensor(a[np.newaxis, :, :], dtype=backend.float_dtype)

    print("  computing baseline incremental outcomes per draw ...")
    inc_base = np.asarray(an.incremental_outcome(
        aggregate_geos=True, aggregate_times=True))
    print("  computing scenario incremental outcomes per draw ...")
    inc_scen = np.asarray(an.incremental_outcome(
        new_data=tensors_mod.DataTensors(media=to_t(scen_media)),
        aggregate_geos=True, aggregate_times=True))
    assert inc_base.shape == inc_scen.shape, (inc_base.shape, inc_scen.shape)

    d0 = inc_base.reshape(-1, inc_base.shape[-1])
    d1 = inc_scen.reshape(-1, inc_scen.shape[-1])
    gain = d1[:, vi] - d0[:, vi]
    loss = d0[:, di] - d1[:, di]
    net = gain - loss
    return {
        "method": "full posterior draws through analyzer.incremental_outcome(new_data=scenario_media)",
        "shift_amount": shift_amt, "from": "Display", "to": "Online Video",
        "video_multiplier": m_video, "display_multiplier": 0.85,
        "n_draws": int(net.shape[0]),
        "expected_gain": float(gain.mean()),
        "expected_gain_lo": float(np.percentile(gain, 5)),
        "expected_gain_hi": float(np.percentile(gain, 95)),
        "expected_loss": float(loss.mean()),
        "net": float(net.mean()),
        "net_lo": float(np.percentile(net, 5)),
        "net_hi": float(np.percentile(net, 95)),
        "_net_draws": net,  # stripped before JSON
    }


def df_summary_channel_order():
    return list(C.CHANNELS)


# ---------------------------------------------------------------- charts ---

def chart_actual_vs_predicted(df, an):
    eo = np.asarray(an.expected_outcome(aggregate_times=False))
    draws = eo.reshape(-1, eo.shape[-1])
    mean = draws.mean(axis=0)
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df["week"], df["revenue"] / 1e6, color="#222", lw=1.4, label="Actual revenue")
    ax.plot(df["week"], mean / 1e6, color="#2a6f97", lw=1.4, label="Posterior mean")
    ax.fill_between(df["week"], lo / 1e6, hi / 1e6, color="#2a6f97", alpha=0.18,
                    label="90% credible interval")
    ax.set_title("Model fit: actual vs predicted weekly revenue")
    ax.set_ylabel("Revenue ($M)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.01, -0.18, SIM_FOOTNOTE, transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(f"{C.CHART_DIR}/01_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)


def chart_roi(df_summary):
    s = df_summary.sort_values("roi_mean")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(y, s["roi_mean"], xerr=[s["roi_mean"] - s["roi_lo"],
                                   s["roi_hi"] - s["roi_mean"]],
            color=PALETTE, capsize=4)
    ax.axvline(1.0, color="#bc4b51", ls="--", lw=1.2, label="Breakeven (ROI = 1)")
    ax.set_yticks(y, s["channel"])
    ax.set_xlabel("ROI - incremental revenue \\$ per \\$1 spent (posterior mean, 90% CI)")
    ax.set_title("ROI by channel")
    ax.legend(frameon=False)
    ax.text(0.01, -0.18, SIM_FOOTNOTE, transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(f"{C.CHART_DIR}/02_roi_by_channel.png", bbox_inches="tight")
    plt.close(fig)


def chart_contribution(df_summary):
    s = df_summary.sort_values("incremental_revenue")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(y, s["incremental_revenue"] / 1e6, color=PALETTE)
    for b, (_, r) in zip(bars, s.iterrows()):
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
                f"${r['incremental_revenue']/1e6:.1f}M ({r['pct_contribution']:.1f}%)",
                va="center", fontsize=9)
    ax.set_yticks(y, s["channel"])
    ax.set_xlabel("Incremental revenue over 2 years ($M, posterior mean)")
    ax.set_title("Incremental contribution by channel")
    ax.text(0.01, -0.18, SIM_FOOTNOTE, transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(f"{C.CHART_DIR}/03_contribution.png", bbox_inches="tight")
    plt.close(fig)


def chart_response_curves(an):
    rc = an.response_curves(spend_multipliers=list(RC_MULTS))
    mean_curve = rc["incremental_outcome"].sel(metric="mean")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, ch in enumerate(C.CHANNELS):
        ax.plot(mean_curve["spend_multiplier"], mean_curve.sel(channel=ch) / 1e6,
                marker="o", ms=3, label=ch, color=PALETTE[i])
    ax.axvline(1.0, color="#999", ls=":", lw=1)
    ax.set_xlabel("Spend multiplier (1.0 = historical spend)")
    ax.set_ylabel("Incremental revenue ($M)")
    ax.set_title("Response curves: diminishing returns by channel")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.01, -0.20,
            SIM_FOOTNOTE + " Shown only within observed spend variation (≤1.25x); "
            "extrapolation beyond that is not presented.",
            transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(f"{C.CHART_DIR}/04_response_curves.png", bbox_inches="tight")
    plt.close(fig)


def chart_search_before_after(an):
    """v1 vs v2 search response curve - a rebuild comparison, NOT controlled.

    v1 and v2 differ in the simulated dataset (seed, TV/promo timing), the
    saturation priors, AND the MCMC config. Same analyzer machinery, same
    multiplier grid - but the visible difference cannot be attributed to any
    single change. The extrapolation zone past 1.25x is shaded: v1's curve
    takes off there, v2's stays inside plausible spend levels.
    """
    z = np.load("data/v1_response_curves.npz", allow_pickle=True)
    chs = [str(c) for c in z["channels"]]
    si = chs.index("Paid Search")
    mults_v1 = z["mults"]
    v1_m, v1_lo, v1_hi = z["mean"][:, si], z["lo"][:, si], z["hi"][:, si]

    rc = an.response_curves(spend_multipliers=list(RC_MULTS_FULL))
    mc = rc["incremental_outcome"].sel(metric="mean")
    v2_m = mc.sel(channel="Paid Search").values / 1e6
    v2_lo = rc["incremental_outcome"].sel(metric="ci_lo").sel(channel="Paid Search").values / 1e6
    v2_hi = rc["incremental_outcome"].sel(metric="ci_hi").sel(channel="Paid Search").values / 1e6

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(mults_v1, v1_m / 1e6, ls="--", color="#bc4b51", lw=1.8,
            label="v1")
    ax.fill_between(mults_v1, v1_lo / 1e6, v1_hi / 1e6, color="#bc4b51", alpha=0.15)
    ax.plot(RC_MULTS_FULL, v2_m, color="#2a6f97", lw=1.8,
            label="v2 (rebuilt: new data, seed, priors, MCMC)")
    ax.fill_between(RC_MULTS_FULL, v2_lo, v2_hi, color="#2a6f97", alpha=0.15)
    ax.axvspan(1.25, 1.5, color="#999", alpha=0.12)
    ax.text(1.375, ax.get_ylim()[1] * 0.92, "extrapolation\nzone", ha="center",
            fontsize=8, color="#666")
    ax.axvline(1.0, color="#999", ls=":", lw=1)
    ax.set_xlabel("Spend multiplier (1.0 = historical spend)")
    ax.set_ylabel("Incremental revenue, paid search ($M)")
    ax.set_title("Paid search response curve: v1 vs v2 rebuild\n"
                 "(dataset, seed, priors, and MCMC all changed - not controlled)")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.01, -0.20,
            SIM_FOOTNOTE + " v1's curve explodes past observed spend; v2's stays "
            "inside plausible levels. Curves are posterior means with 90% CIs. "
            "Multiple things changed between versions, so the difference is not "
            "the effect of the prior change alone.",
            transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(f"{C.CHART_DIR}/06_search_before_after.png", bbox_inches="tight")
    plt.close(fig)


def _prior_posterior_fig(idata, var, title, fname, truth_map, logx=False):
    post = idata.posterior[var].stack(s=("chain", "draw")).transpose("s", "media_channel")
    prior = idata.prior[var].stack(s=("chain", "draw")).transpose("s", "media_channel")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for i, ch in enumerate(C.CHANNELS):
        ax = axes[i]
        pr = prior.sel(media_channel=ch).values.ravel()
        po = post.sel(media_channel=ch).values.ravel()
        rng = (min(pr.min(), po.min()), max(pr.max(), po.max()))
        ax.hist(pr, bins=40, range=rng, color="#999", alpha=0.5, label="Prior", density=True)
        ax.hist(po, bins=40, range=rng, color="#2a6f97", alpha=0.65, label="Posterior", density=True)
        if truth_map is not None:
            ax.axvline(truth_map[ch], color="#bc4b51", ls="--", lw=1.4, label="Sim truth")
        ax.set_title(ch, fontsize=10)
        if logx:
            ax.set_xscale("log")
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.text(0.01, 0.01, SIM_FOOTNOTE + " 'Sim truth' is the simulation's known "
             "parameter - the model never sees it.", fontsize=8, color="#666")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(f"{C.CHART_DIR}/{fname}", bbox_inches="tight")
    plt.close(fig)


def chart_prior_posterior(idata):
    with open("data/dgp_truth.yaml") as f:
        truth = yaml.safe_load(f)
    key = {"Paid Search": "paid_search", "Paid Social": "paid_social",
           "TV": "tv", "Display": "display", "Online Video": "video"}
    roi_truth = {ch: truth["channels"][key[ch]]["roi_true_per_adstocked_dollar"]
                 for ch in C.CHANNELS}
    ec_truth = {ch: truth["channels"][key[ch]]["hill_ec"] for ch in C.CHANNELS}
    _prior_posterior_fig(idata, "roi_m", "Prior vs posterior: ROI by channel",
                         "07_prior_posterior_roi.png", roi_truth)
    _prior_posterior_fig(idata, "ec_m", "Prior vs posterior: half-saturation (ec) by channel",
                         "08_prior_posterior_ec.png", ec_truth, logx=True)


def chart_scenario(scen):
    net = scen["_net_draws"] / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(net, bins=40, color="#2a6f97", alpha=0.7)
    ax.axvline(scen["net"] / 1e6, color="#222", lw=1.8,
               label=f"Mean: +${scen['net']/1e6:.1f}M")
    ax.axvline(scen["net_lo"] / 1e6, color="#bc4b51", ls="--", lw=1.4,
               label=f"90% CI: ${scen['net_lo']/1e6:.1f}M – ${scen['net_hi']/1e6:.1f}M")
    ax.axvline(scen["net_hi"] / 1e6, color="#bc4b51", ls="--", lw=1.4)
    ax.axvline(0, color="#999", lw=1)
    ax.set_xlabel("Net incremental revenue, 2 years ($M)")
    ax.set_ylabel("Posterior draws")
    ax.set_title("Budget scenario: 15% of Display → Online Video\n"
                 "full-posterior uncertainty on the net gain")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.01, -0.22,
            SIM_FOOTNOTE + " Every posterior draw propagated through the model's "
            "own adstock + saturation transform. Directional, not a media plan.",
            transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(f"{C.CHART_DIR}/05_budget_scenario.png", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(C.CHART_DIR, exist_ok=True)
    df, an, idata = load()

    summary = posterior_summary(an)
    print("\n=== Channel summary (posterior) ===")
    print(summary.round(3).to_string(index=False))

    cal = calibration(summary)
    print("\n=== Calibration vs simulation truth ===")
    print(cal.round(2).to_string(index=False))
    print(f"  true ROIs inside 90% CI: {int(cal['in_90ci'].sum())}/{len(cal)}")

    scen = budget_scenario_full_posterior(an, df)
    print("\n=== Budget scenario (full posterior): 15% of Display -> Online Video ===")
    print(f"  draws: {scen['n_draws']}")
    print(f"  expected gain (video): ${scen['expected_gain']:,.0f} "
          f"(90% CI ${scen['expected_gain_lo']:,.0f} – ${scen['expected_gain_hi']:,.0f})")
    print(f"  expected loss (display): ${scen['expected_loss']:,.0f}")
    print(f"  NET: ${scen['net']:,.0f} (90% CI ${scen['net_lo']:,.0f} – ${scen['net_hi']:,.0f})")

    chart_actual_vs_predicted(df, an)
    chart_roi(summary)
    chart_contribution(summary)
    chart_response_curves(an)
    chart_search_before_after(an)
    chart_prior_posterior(idata)
    chart_scenario(scen)
    print("\ncharts written to charts/")

    scen_out = {k: v for k, v in scen.items() if not k.startswith("_")}
    readout = {
        "note": "SIMULATED DATA - Everline Foods is fictional.",
        "version": "v2",
        "channels": summary.to_dict(orient="records"),
        "calibration": cal.to_dict(orient="records"),
        "scenario": scen_out,
    }
    with open("data/readout.json", "w") as f:
        json.dump(readout, f, indent=2, default=float)
    print("saved data/readout.json")


if __name__ == "__main__":
    main()
