# Where should the next media dollar go? An MMM read on a CPG portfolio

**Note: every number below comes from simulated data.** "Everline Foods" is a
fictional brand. I built the dataset with a known data-generating process
(adstock, diminishing returns, seasonality, promo, noise) so I could test the
modeling workflow end to end — including checking the model's answers against
the truth, which you can't do with real data.

## Setup

- 104 weeks of national data, 5 channels: paid search, paid social, TV,
  display, online video. Total media: $12.5M. Total revenue: $342M.
- Google Meridian (Bayesian MMM), fitted with MCMC — 4 chains, 2000 kept
  draws each. Priors were set from domain knowledge, not from the simulation
  truth: search/video expected most efficient, display near breakeven, TV
  with the longest carryover.
- One deliberate design choice worth flagging: TV flights are staggered
  against promo weeks (clean TV-only windows, a clean promo-only window, and
  partial overlaps each year). In the first version of this project every TV
  flight ran exactly on top of promo weeks and TV credit leaked into promo —
  the stagger makes that confounding testable instead of accidental.
- Convergence is clean (R-hat ≈ 1.00 on every parameter, 0% bad) and the
  model tracks history well: R² 0.94, weighted MAPE 3.6% on weekly revenue.

## What the model says

| Channel | 2-yr spend | Incremental revenue | Share of revenue | ROI (90% CI) |
|---|---|---|---|---|
| Online video | $1.7M | $15.7M | 4.6% | 9.4 (6.1–14.4) |
| Paid search | $3.0M | $27.3M | 8.0% | 9.2 (5.4–14.3) |
| TV | $4.8M | $25.5M | 7.4% | 5.4 (4.7–6.1) |
| Paid social | $2.3M | $10.4M | 3.0% | 4.6 (1.8–8.5) |
| Display | $0.9M | $1.1M | 0.3% | 1.2 (0.3–3.1) |

Media overall drove roughly 23% of revenue; the rest is baseline (brand,
seasonality, promo, price).

## Where the model misses

Because the data is simulated, I know the true ROIs — and the model gets
four out of five inside its 90% intervals. The remaining miss is paid
search — the model says 9.2, truth is 4.5. Search spend grew smoothly alongside the brand's growth trend for two
straight years, so the model can't fully separate the two. That's not a
modeling failure, it's the data telling the truth about what 104 weeks of
observational data can and can't identify. The intervals carry the real
information here: search's 90% interval (5.4–14.3) spans nearly a 3x range,
so the 9.2 point estimate should not drive any budget decision on its own.
TV's estimate is 5.4 against a truth of 6.0; v2 staggers TV flights against
promo weeks (v1 ran every flight on top of promo weeks) to make TV/promo
confounding testable, but the dataset, seed, and priors all changed between
versions, so no single change gets the credit.

## What changed between v1 and v2

v1 had a real problem: the priors on the saturation curve were wide enough
that paid search's response curve exploded once you pushed spend past what
was ever observed — the classic unidentified-Hill problem. The v2 rebuild
changed several things at once: tighter saturation priors, response curves
presented only inside observed spend variation (≤1.25x), staggered TV/promo
timing in the simulated dataset, a new random seed, and longer MCMC. Because
all of that changed together, the v1-vs-v2 before/after chart documents the
rebuild — it cannot isolate the effect of the prior change. The chart is in
the charts folder; read it as a log, not an experiment.

## Budget scenario

Shift 15% of display spend ($128k) into online video. Instead of
interpolating curve confidence bands, I pushed all 8,000 posterior draws
through the model's own adstock + saturation transform, so the interval
reflects the joint uncertainty in ROI, carryover, and saturation. Net gain:
**+$2.6M incremental revenue** over two years, 90% interval $1.9M–$3.2M.
Directional, not a media plan: marginal-return estimates are the
least-identified part of any MMM, and I'd want a holdout test before moving
real money on the strength of one model run.

## Geo-holdout validation

I also built a 12-geo variant of the dataset (same true parameters,
geo-sized saturation, two holdout geos) and asked: does the national
response-curve shape transport to geos the model never saw? With holdout
media rescaled to national-equivalent — saturation is scale-dependent, so
this is the correct comparison — the model recovers 4 of 5 channels inside
its 90% intervals in both the holdout and training regions. The misses are
the same two channels the national calibration already flagged (search and
video get over-credited), which is exactly what a validation should do:
reproduce the known weaknesses, not hide them. This is a transportability
check on simulated data, not a randomized experiment — a real geo test would
randomize the holdout.

## Bottom line for a CMO

- **Online video is the strongest read.** Highest estimated ROI, and it's not
  saturated — it has headroom before diminishing returns bite. The estimate
  runs hot (9.4 vs 6.1 true), so treat the ranking as the signal, not the
  point estimate.
- **Paid search is materially unidentified.** The model says 9.2; the truth
  is 4.5 — roughly double. Search spend grew smoothly alongside the brand's
  growth trend for two straight years, so the model can't separate the two.
  Do not use this ROI to guide allocation without a calibration experiment.
- **TV works, expensively.** Biggest absolute contributor, middling
  efficiency — with TV/promo confounding addressed in the v2 design, the
  5.4x read is one you can actually use.
- **Display is on the bubble.** ROI interval straddles breakeven (0.3–3.1).
  Cut it or prove it with a geo test — don't scale it.
- **Paid social is the fuzziest read** (ROI 1.8–8.5). It needs an
  incrementality test before any budget decision.
- **Display-to-video is the strongest hypothesis for an incrementality
  test:** the full-posterior scenario puts the expected gain at +$2.6M
  (90% interval $1.9M–$3.2M) for a $128k shift. Directional, not a media
  plan — the next step is a test, not a reallocation.
