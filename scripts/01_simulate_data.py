"""
Simulate a weekly CPG/retail-style MMM dataset (~2 years).

*** ALL DATA IN THIS FILE IS SIMULATED. ***
It is NOT real company data. It exists only to demonstrate an MMM workflow
for a portfolio case study.

True data-generating process (known to us, NOT to the model):
  revenue_t = base * trend_t * seasonality_t * price_effect_t * (1 + promo_lift*promo_t)
              + sum_c roi_c * adstock_c(spend_c, t) * hill_c(adstocked spend)
              + noise

  - adstock: geometric decay per channel (carryover effect)
  - hill: diminishing returns (saturation curve)
  - promo/price/seasonality/trend: non-media controls

Channels: paid_search, paid_social, tv, display, video (online/CTV).
Outcome: weekly revenue ($).
"""
import numpy as np
import pandas as pd
import yaml

RNG = np.random.default_rng(43)

N_WEEKS = 104
START = "2024-01-01"
BRAND = "Everline Foods (FICTIONAL - simulated data)"

# --- v2 design: TV flights are STAGGERED vs promo weeks -----------------------
# In v1, every TV flight ran exactly on top of promo weeks, so the model had
# to split TV-vs-promo credit by guesswork. In v2 each year has:
#   TV flights:      weeks (6,12), (28,34), (44,50)
#   Promo weeks:     weeks (9,15), (22,25), (40,46)
# That yields clean TV-only windows (6-8, 28-33, 46-49), a clean promo-only
# window (22-24), and partial overlaps (9-11, 44-45) - so the confounding is
# now a testable feature: the model should recover TV's ROI better than v1.
TV_FLIGHTS = ((6, 12), (28, 34), (44, 50))
PROMO_WINDOWS = ((9, 15), (22, 25), (40, 46))

# ----------------------------------------------------------------------------
# Channel configuration: spend patterns, adstock decay, saturation, true ROI
# ----------------------------------------------------------------------------
CHANNELS = {
    # name:        spend_fn params,          decay, hill_ec, hill_slope, roi_true
    "paid_search": {"base": 25000, "decay": 0.30, "ec": 22000, "slope": 2.2, "roi": 4.0},
    "paid_social": {"base": 18000, "decay": 0.50, "ec": 16000, "slope": 2.5, "roi": 2.8},
    "tv":          {"base": 130000, "decay": 0.70, "ec": 90000, "slope": 2.0, "roi": 2.2},
    "display":     {"base": 8000,  "decay": 0.20, "ec": 7000,  "slope": 3.0, "roi": 1.2},
    "video":       {"base": 12000, "decay": 0.55, "ec": 14000, "slope": 2.4, "roi": 3.2},
}

BASE_REVENUE = 2_100_000   # $/week baseline
PRICE_BASE = 3.49
PRICE_PROMO = 2.99
PRICE_ELASTICITY = -1.8
PROMO_LIFT = 0.12          # extra lift beyond the price cut (feature/display support)
NOISE_SD = 90_000          # ~3-4% of weekly revenue


def hill(x, ec, slope):
    """Hill saturation function in (0, 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / (1.0 + (x / ec) ** (-slope))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def geometric_adstock(spend, decay):
    out = np.zeros_like(spend, dtype=float)
    for t in range(len(spend)):
        out[t] = spend[t] + (decay * out[t - 1] if t > 0 else 0.0)
    return out


def simulate():
    t = np.arange(N_WEEKS)
    weeks = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    woy = weeks.isocalendar().week.to_numpy().astype(float)

    # ---- media spend -------------------------------------------------------
    spend = {}

    # paid search: always-on, gentle growth + noise
    s = CHANNELS["paid_search"]["base"] * (1 + 0.003 * t) * (1 + 0.15 * RNG.normal(size=N_WEEKS))
    spend["paid_search"] = np.clip(s, 5000, None)

    # paid social: always-on with pulsing bursts (~13-week cycle)
    pulse = 1 + 0.9 * np.maximum(0, np.sin(2 * np.pi * t / 13 + 1.0)) ** 2
    s = CHANNELS["paid_social"]["base"] * pulse * (1 + 0.15 * RNG.normal(size=N_WEEKS))
    spend["paid_social"] = np.clip(s, 3000, None)

    # tv: flighted - 3 flights per year, dark otherwise (staggered vs promo, see v2 design note)
    s = np.zeros(N_WEEKS)
    for year_start in (0, 52):
        for fstart, fend in TV_FLIGHTS:
            a, b = year_start + fstart, min(year_start + fend, N_WEEKS)
            s[a:b] = CHANNELS["tv"]["base"] * (1 + 0.10 * RNG.normal(size=b - a))
    spend["tv"] = np.clip(s, 0, None)

    # display: low always-on
    s = CHANNELS["display"]["base"] * (1 + 0.20 * RNG.normal(size=N_WEEKS))
    spend["display"] = np.clip(s, 2000, None)

    # video (CTV/OLV): seasonal bursts, heavy in Q4
    q4 = ((woy >= 44) | (woy <= 2)).astype(float)
    s = CHANNELS["video"]["base"] * (1 + 1.8 * q4) * (1 + 0.20 * RNG.normal(size=N_WEEKS))
    spend["video"] = np.clip(s, 2000, None)

    # ---- controls ----------------------------------------------------------
    # promo weeks: staggered vs TV flights (v2 design) - clean TV-only,
    # clean promo-only, and partial-overlap windows each year
    promo = np.zeros(N_WEEKS)
    for year_start in (0, 52):
        for fstart, fend in PROMO_WINDOWS:
            a, b = year_start + fstart, min(year_start + fend, N_WEEKS)
            promo[a:b] = 1.0

    price = np.where(promo == 1, PRICE_PROMO, PRICE_BASE).astype(float)
    # small week-to-week price variation (shelf-price noise) so price isn't
    # perfectly collinear with the promo flag
    price = price * (1 + 0.01 * RNG.normal(size=N_WEEKS))
    price_effect = (price / PRICE_BASE) ** PRICE_ELASTICITY

    trend = 1 + 0.0015 * t  # slow brand growth (~8%/yr)

    # seasonality: December holiday spike, summer lift, January dip
    seasonality = (
        1
        + 0.20 * np.exp(-(((woy - 52) / 5.0) ** 2))
        + 0.20 * np.exp(-(((woy - 1) / 5.0) ** 2))   # wraps around new year
        + 0.08 * np.exp(-(((woy - 28) / 5.0) ** 2))
        - 0.06 * np.exp(-(((woy - 4) / 3.0) ** 2))
    )

    # ---- media contributions (the "truth") ---------------------------------
    contributions = {}
    for name, cfg in CHANNELS.items():
        ads = geometric_adstock(spend[name], cfg["decay"])
        sat = hill(ads, cfg["ec"], cfg["slope"])
        contributions[name] = cfg["roi"] * ads * sat

    media_total = sum(contributions.values())

    base_component = BASE_REVENUE * trend * seasonality * price_effect * (1 + PROMO_LIFT * promo)
    noise = RNG.normal(0, NOISE_SD, N_WEEKS)
    revenue = base_component + media_total + noise

    df = pd.DataFrame({
        "week": weeks,
        "revenue": revenue,
        "spend_paid_search": spend["paid_search"],
        "spend_paid_social": spend["paid_social"],
        "spend_tv": spend["tv"],
        "spend_display": spend["display"],
        "spend_video": spend["video"],
        "price": price,
        "promo": promo,
    })
    df["is_simulated"] = True

    # ground-truth summary (kept for calibration checks, not shown to model)
    truth = {
        "brand": BRAND,
        "note": "SIMULATED data. Ground truth of the simulation, for calibration checks only.",
        "version": "v2",
        "design_notes": (
            "v2 staggers TV flights vs promo weeks so the TV/promo confounding is a "
            "testable feature: each year has clean TV-only windows (weeks 6-8, 28-33, "
            "46-49), a clean promo-only window (22-24), and partial overlaps (9-11, "
            "44-45). In v1 every TV flight ran exactly on top of promo weeks. "
            "Same DGP parameters as v1; fresh RNG seed (43)."
        ),
        "tv_flights": [list(w) for w in TV_FLIGHTS],
        "promo_windows": [list(w) for w in PROMO_WINDOWS],
        "weeks": N_WEEKS,
        "channels": {
            name: {
                "decay": cfg["decay"], "hill_ec": cfg["ec"], "hill_slope": cfg["slope"],
                "roi_true_per_adstocked_dollar": cfg["roi"],
                "total_spend": float(spend[name].sum()),
                "total_true_contribution": float(contributions[name].sum()),
            }
            for name, cfg in CHANNELS.items()
        },
        "controls": {
            "base_revenue": BASE_REVENUE,
            "price_elasticity": PRICE_ELASTICITY,
            "promo_lift": PROMO_LIFT,
            "noise_sd": NOISE_SD,
        },
    }
    return df, truth


if __name__ == "__main__":
    df, truth = simulate()
    df.to_csv("data/simulated_mmm_weekly.csv", index=False)
    with open("data/dgp_truth.yaml", "w") as f:
        yaml.dump(truth, f, default_flow_style=False)

    print(f"brand: {BRAND}")
    print(f"weeks: {len(df)}  ({df['week'].min().date()} -> {df['week'].max().date()})")
    print(f"revenue: mean=${df['revenue'].mean():,.0f}  total=${df['revenue'].sum():,.0f}")
    print("\nspend by channel:")
    for c in ["paid_search", "paid_social", "tv", "display", "video"]:
        col = f"spend_{c}"
        print(f"  {c:12s} total=${df[col].sum():>11,.0f}  weekly_mean=${df[col].mean():>8,.0f}")
    print("\ntrue contribution share (of revenue):")
    total_rev = df["revenue"].sum()
    for name, cfg in truth["channels"].items():
        share = cfg["total_true_contribution"] / total_rev * 100
        print(f"  {name:12s} ${cfg['total_true_contribution']:>11,.0f}  ({share:4.1f}%)")
