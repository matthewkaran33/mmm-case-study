"""
Simulate a GEO-structured variant of the Everline Foods MMM dataset.

*** ALL DATA IN THIS FILE IS SIMULATED. ***
It is NOT real company data. "Everline Foods" is a fictional brand.

Purpose: build the dataset for a synthetic transportability check. 12 geos
x 104 weeks, same DGP parameters as the national v2 simulation (same true
ROI / adstock / Hill per channel), but geo-specific spend patterns, size
scales, and noise. Two geos are designated EVAL: the transportability script
(04_transportability_check.py) applies the national posterior - fit on
national weekly data, never on any of these 12 geos - to the eval geos and
checks the predicted channel effects against the known truth.

Run from the repo root:  python scripts/01b_simulate_data.py
"""
import numpy as np
import pandas as pd
import yaml

# import the national v2 DGP building blocks
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "sim01", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_simulate_data.py"))
sim01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim01)

RNG = np.random.default_rng(44)

N_GEOS = 12
# size shares (sum to 1) - a couple of big geos, a long tail of small ones
GEO_SHARES = np.array([0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
GEO_SHARES = GEO_SHARES / GEO_SHARES.sum()
GEO_IDS = [f"geo_{i+1:02d}" for i in range(N_GEOS)]
HOLDOUT_GEOS = ["geo_04", "geo_09"]  # mid-size geos, held out of the validation fit

BRAND = "Everline Foods (FICTIONAL - simulated data)"


def simulate_geo():
    t = np.arange(sim01.N_WEEKS)
    weeks = pd.date_range(sim01.START, periods=sim01.N_WEEKS, freq="W-MON")
    woy = weeks.isocalendar().week.to_numpy().astype(float)

    # national promo/price/seasonality/trend are shared (retailer-wide promos)
    promo = np.zeros(sim01.N_WEEKS)
    for year_start in (0, 52):
        for fstart, fend in sim01.PROMO_WINDOWS:
            a, b = year_start + fstart, min(year_start + fend, sim01.N_WEEKS)
            promo[a:b] = 1.0
    price = np.where(promo == 1, sim01.PRICE_PROMO, sim01.PRICE_BASE).astype(float)
    price = price * (1 + 0.01 * RNG.normal(size=sim01.N_WEEKS))
    price_effect = (price / sim01.PRICE_BASE) ** sim01.PRICE_ELASTICITY
    trend = 1 + 0.0015 * t
    seasonality = (
        1
        + 0.20 * np.exp(-(((woy - 52) / 5.0) ** 2))
        + 0.20 * np.exp(-(((woy - 1) / 5.0) ** 2))
        + 0.08 * np.exp(-(((woy - 28) / 5.0) ** 2))
        - 0.06 * np.exp(-(((woy - 4) / 3.0) ** 2))
    )
    base_component = (sim01.BASE_REVENUE * trend * seasonality
                      * price_effect * (1 + sim01.PROMO_LIFT * promo))

    rows = []
    truth_geos = {}
    for gi, (geo, share) in enumerate(zip(GEO_IDS, GEO_SHARES)):
        g = np.random.default_rng(1000 + gi)  # per-geo stream, reproducible
        spend = {}
        # search: always-on + growth, geo-level variation around the national shape
        s = (sim01.CHANNELS["paid_search"]["base"] * share * (1 + 0.003 * t)
             * (1 + 0.15 * g.normal(size=sim01.N_WEEKS))
             * (1 + 0.25 * g.normal()))
        spend["paid_search"] = np.clip(s, 500, None)
        # social: pulsing
        pulse = 1 + 0.9 * np.maximum(0, np.sin(2 * np.pi * t / 13 + 1.0)) ** 2
        s = (sim01.CHANNELS["paid_social"]["base"] * share * pulse
             * (1 + 0.15 * g.normal(size=sim01.N_WEEKS)) * (1 + 0.25 * g.normal()))
        spend["paid_social"] = np.clip(s, 300, None)
        # tv: flighted (same windows, geo-level intensity variation)
        s = np.zeros(sim01.N_WEEKS)
        intensity = 1 + 0.3 * g.normal()
        for year_start in (0, 52):
            for fstart, fend in sim01.TV_FLIGHTS:
                a, b = year_start + fstart, min(year_start + fend, sim01.N_WEEKS)
                s[a:b] = (sim01.CHANNELS["tv"]["base"] * share * intensity
                          * (1 + 0.10 * g.normal(size=b - a)))
        spend["tv"] = np.clip(s, 0, None)
        # display: low always-on
        s = (sim01.CHANNELS["display"]["base"] * share
             * (1 + 0.20 * g.normal(size=sim01.N_WEEKS)) * (1 + 0.25 * g.normal()))
        spend["display"] = np.clip(s, 200, None)
        # video: Q4 bursts
        q4 = ((woy >= 44) | (woy <= 2)).astype(float)
        s = (sim01.CHANNELS["video"]["base"] * share * (1 + 1.8 * q4)
             * (1 + 0.20 * g.normal(size=sim01.N_WEEKS)) * (1 + 0.25 * g.normal()))
        spend["video"] = np.clip(s, 200, None)

        contributions = {}
        for name, cfg in sim01.CHANNELS.items():
            ads = sim01.geometric_adstock(spend[name], cfg["decay"])
            # Saturation scales with market size: a geo with 6% of national
            # spend saturates at 6% of the national half-saturation point.
            # (Without this, tiny geos would sit deep in the linear region and
            # a national saturation curve could never transport to them.)
            ec_geo = cfg["ec"] * share
            contributions[name] = cfg["roi"] * ads * sim01.hill(ads, ec_geo, cfg["slope"])

        geo_base = base_component * share
        noise = g.normal(0, sim01.NOISE_SD * np.sqrt(share), sim01.N_WEEKS)
        revenue = geo_base + sum(contributions.values()) + noise

        for i in range(sim01.N_WEEKS):
            rows.append({
                "geo": geo, "week": weeks[i], "revenue": revenue[i],
                "spend_paid_search": spend["paid_search"][i],
                "spend_paid_social": spend["paid_social"][i],
                "spend_tv": spend["tv"][i],
                "spend_display": spend["display"][i],
                "spend_video": spend["video"][i],
                "price": price[i], "promo": promo[i],
                "is_simulated": True,
                "holdout": geo in HOLDOUT_GEOS,
            })
        truth_geos[geo] = {
            "share": float(share),
            "holdout": geo in HOLDOUT_GEOS,
            "total_spend": {n: float(spend[n].sum()) for n in sim01.CHANNELS},
            "total_true_contribution": {n: float(contributions[n].sum())
                                        for n in sim01.CHANNELS},
        }

    df = pd.DataFrame(rows)
    truth = {
        "brand": BRAND,
        "note": ("SIMULATED data. Geo variant of the v2 DGP - same true ROI/decay/slope, "
                 "geo-specific spend draws and noise. Saturation (ec) scales with geo "
                 "size: ec_geo = ec_national * geo_share, so each geo sits at the same "
                 "relative point on its saturation curve as the nation does."),
        "n_geos": N_GEOS,
        "weeks": sim01.N_WEEKS,
        "holdout_geos": HOLDOUT_GEOS,
        "geos": truth_geos,
    }
    return df, truth


if __name__ == "__main__":
    df, truth = simulate_geo()
    df.to_csv("data/simulated_mmm_geo.csv", index=False)
    with open("data/dgp_truth_geo.yaml", "w") as f:
        yaml.dump(truth, f, default_flow_style=False)
    print(f"geos: {N_GEOS} x {sim01.N_WEEKS} weeks = {len(df)} rows")
    print(f"holdout: {HOLDOUT_GEOS}")
    print(f"revenue total: ${df['revenue'].sum():,.0f}")
    # sanity: geo revenue should roughly tile the national scale
    print(f"geo shares sum: {GEO_SHARES.sum():.3f}")
