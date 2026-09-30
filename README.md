# MMM Case Study — "Everline Foods" (SIMULATED DATA)

> **All data in this project is simulated.** "Everline Foods" is a fictional
> brand. Nothing here is real company data. The datasets were generated with a
> known data-generating process (adstock + saturation + seasonality + promo +
> noise) so the modeling workflow can be demonstrated end to end — including
> checking the model's answers against the truth, which you can't do with
> real data.

## What this is

An end-to-end Marketing Mix Modeling case study built with
[Google Meridian](https://github.com/google/meridian):

1. **Data prep** — 104 weeks of weekly CPG-style data: 5 media channels
   (paid search, paid social, TV, display, online video), price, promo,
   seasonality, and weekly revenue. Plus a geo-structured variant
   (12 geos × 104 weeks, with 2 holdout geos) for a holdout-validation demo.
2. **Model** — Meridian (Bayesian hierarchical MMM), 4 chains × 2000 kept
   draws, with informed priors on ROI, adstock decay, and saturation per
   channel. Saturation priors were deliberately tightened in v2 after v1's
   wide priors let one channel's response curve extrapolate wildly.
3. **Diagnostics** — convergence (R-hat), posterior predictive checks,
   prior-vs-posterior shift charts. Limitations are noted honestly.
4. **Read-out** — channel contribution, ROI per channel, calibration of the
   model's answers against the simulation's known truth.
5. **Scenario** — budget reallocation ("what if we shifted 15% of display
   spend into online video?") with uncertainty from full posterior draws.
6. **Geo-holdout demo** — do national estimates transport to unseen geos?
7. **Business summary** — plain-English takeaways (`docs/case_study.md`).

## Repo layout

```
mmm-case-study/
├── data/
│   ├── simulated_mmm_weekly.csv   # national dataset (SIMULATED)
│   ├── dgp_truth.yaml             # ground truth of the national simulation
│   ├── simulated_mmm_geo.csv      # geo variant: 12 geos x 104 weeks (SIMULATED)
│   ├── dgp_truth_geo.yaml         # ground truth of the geo simulation
│   ├── inference_data.nc          # fitted posterior + prior draws
│   ├── v1_response_curves.npz     # v1 search response curve (before/after)
│   ├── readout.json               # headline numbers (national read-out)
│   └── geo_holdout.json           # holdout validation numbers
├── scripts/
│   ├── 01_simulate_data.py        # builds the national dataset (v2: TV flights
│   │                              #   staggered vs promo weeks)
│   ├── 01b_simulate_geo.py        # builds the geo variant dataset
│   ├── 02_fit_meridian.py         # trains the Meridian model (4 chains x 2000
│   │                              #   draws), saves diagnostics
│   ├── 03_readout.py              # contribution, ROI, full-posterior scenario,
│   │                              #   charts (incl. prior-vs-posterior, before/after)
│   └── 04_geo_holdout.py          # holdout-region validation vs known truth
├── charts/                        # output figures (v1 archived in v1_before/)
├── docs/
│   └── case_study.md              # one-page write-up
└── README.md
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # google-meridian + deps
python scripts/01_simulate_data.py
python scripts/01b_simulate_geo.py
python scripts/02_fit_meridian.py  # MCMC; ~15-25 min on CPU (4 chains x 2000 draws)
python scripts/03_readout.py       # tables, charts, data/readout.json
python scripts/04_geo_holdout.py   # holdout validation, data/geo_holdout.json
```

Environment note: on small-/tmp machines (e.g. 512MB tmpfs), point pip's temp
dir at real disk or the Meridian download will fail with "No space left on
device": `TMPDIR=~/.pip-tmp pip install -r requirements.txt`.

## What changed in v2 (and why)

- **Staggered TV/promo design.** In v1 every TV flight ran exactly on top of
  promo weeks, so TV credit leaked into promo. v2 has clean TV-only windows,
  a clean promo-only window, and partial overlaps each year — the confounding
  is now a testable feature.
- **Tighter saturation priors** (`ec` sigma 0.9 → 0.45, slope sigma 0.6 →
  0.35) after v1's wide priors let paid search's response curve explode past
  observed spend. Response curves are now presented only within observed
  spend variation (≤1.25x); the v1-vs-v2 before/after is chart 06.
- **Longer MCMC**: 4 chains × 2000 kept draws (was 2 × 500), plus
  prior-vs-posterior shift charts (07, 08).
- **Honest scenario uncertainty**: the budget reallocation pushes every
  posterior draw through the model's own adstock + saturation transform
  instead of interpolating curve CIs (chart 05 is now the net-gain
  distribution).
- **Geo-holdout demo** (chart 09): national estimates validated against
  unseen geos on data where the truth is known. Framed as a transportability
  check, not a randomized experiment.

## Limitations (read before citing any number)

- Simulated data: clean by construction. Real data has messier confounders
  (competitor spend, distribution changes, creative quality shifts) that this
  demo does not include.
- 104 weeks is the minimum viable history; real MMMs prefer 2–3+ years.
- ROI estimates are posterior means with wide credible intervals — treat the
  ranking as more reliable than any single point estimate.
- The budget scenario is a directional illustration, not a media plan.
  Marginal-return estimates are the least-identified part of any MMM; the
  full-posterior interval is honest about that and it is wide.
- The geo-holdout demo tests transportability to unseen geos on simulated
  data. A real geo test randomizes the holdout — this demo does not.
