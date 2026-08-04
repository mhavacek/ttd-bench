# Phase-2 analysis plan — documented deviations from the frozen §3

**Date: 2026-07-24.** Written after the frozen Phase-2 data collection completed
(3400 runs, config hash `20bdb7a3775a1c3d`, git `356842d`, all on konos /
GTX 1080 Ti) and **before** any final inferential statistic is reported.

The experimental design (§2) and the data are untouched. What follows are
changes to the *analysis* of that data, each with the reason it became
necessary. Every quantity is computed post-hoc from the immutable raw logs.

---

## D1 — Runs that processed zero frames are excluded (n = 775, 22.8 %)

775 of 3400 logs contain a header and nothing else: zero frames, no `end`
record. These are infrastructure failures of the job (the run never started
streaming), not detection failures, and they must not be scored as misses.

Failure rate is strongly configuration-dependent — local-gpu 7.2 %, edge-sim
15.3 %, remote-wifi 24.9 %, remote-lan 28.2 %, remote-cellular-4g 38.4 % —
i.e. it rises with the number of subprocesses a configuration has to start
(mediamtx, remote inference server, network proxy).

Exclusion is defensible only if failure is independent of scene content, which
was tested:

| test | result |
|---|---|
| failure vs. scenario (all runs) | chi2 = 41.1, df = 33, **p = 0.157** |
| failure vs. repetition | **p = 0.302** |
| failure vs. clip length | Spearman rho = −0.024, **p = 0.894** |
| failure vs. model | p = 0.0074 (range 19.1–25.1 %) |
| failure vs. scenario *within* remote-lan | p = 1.3e-8 |

Content-independence holds overall; the two significant results (a narrow
model effect and one configuration's scenario dependence) are reported as a
limitation. Port collision was tested as a cause and **ruled out**:
remote-lan straddles the cell-index boundary at which the port offset wraps,
and its failure rate is 28.1 % below vs 30.0 % above the boundary.

**Consequence:** all rates below are computed on the 2625 valid runs. This
matters — on the contaminated set the local-vs-4G detection gap reads 25.0 pp,
on valid runs only 17.6 pp.

## D2 — TTD is analysed as a censored time-to-event variable

**Reason.** TTD is observed only on runs where an alarm fired, and the alarm
rate differs by configuration (45.3 % local-gpu vs 27.7 % 4G). A median over
observed TTDs therefore averages over *different subsets of scenarios*: the
hard scenarios drop out of the slow configurations, biasing their median
downwards. The naive medians (481 ms local vs 774 ms 4G) understate the gap
and are not comparable across configurations.

**Change.** Time from onset to alarm, with non-detections right-censored.
Kaplan–Meier estimates of P(alarm by t), pairwise log-rank, and a Cox
proportional-hazards model with configuration and model as covariates.
Detection rate and detection latency stop being two metrics and become one
curve. Implemented in `scripts/survival_phase2.py`.

**Censoring is administrative and set by the clip, not by the 30 s timeout.**
SCVD clips end a median of 3.37 s after the onset (IQR 2.57–4.81 s; 77 % of
runs have less than 30 s of post-onset footage). A miss is censored at
`followup_ms` = onset → last processed frame. Follow-up is near-identical
across configurations (3.32–3.57 s median), so censoring is non-differential.

The nominal 30 s timeout of §2 is therefore never the binding constraint and
the reported "miss within 30 s" of the operational definition should be read
as "no alarm within the available footage".

## D3 — Dependence: repetitions are clustered, not independent

5 repetitions of the same scenario are not 5 independent observations. Cox
models use robust standard errors clustered on `scenario`. Log-rank p-values
assume independence, are anticonservative, and are reported for reference
only; **the clustered Cox model is the inferential statement.**

## D4 — Paired tests, where still used, pair per configuration PAIR

The intersection of scenarios detected in *all five* configurations is too
small for a five-way paired design (1/34 for yolov12m, 15/34 for yolov8s).
Pairwise intersections are usable (up to 24/34). Therefore: paired tests per
configuration pair, McNemar for detection-rate differences on scenario level,
**Bonferroni across all 10 configuration pairs** (not per pair).

With D2 adopted this becomes a secondary analysis — log-rank/Cox under
censoring use all 34 scenarios in every configuration and need no intersection.

## D5 — Alarm-rule ablation (K-consecutive is not assumed)

**Reason.** Under 55–70 % frame drop the K = 3 *consecutive* rule is close to
unsatisfiable by construction, so a hit-rate drop cannot be attributed to the
network without showing it survives other rules.

**Change.** The same logs are re-scored under K-within-a-window rules and at
the policy-free bound (K = 1, i.e. a single detection anywhere after onset),
which upper-bounds what *any* alarm policy could achieve on the frames that
configuration actually saw. Implemented via `win_hit`/`ttd_win_ms` in
`metrics.py` (additive; the §2 and measurement definitions are unchanged).

## D6 — Cross-model claims are made at matched false-alarm rates

**Reason.** The detector logged detections from conf 0.1 while the alarm used
0.5; 94.5 % of logged detections fall below the alarm threshold. Comparing
architectures at one fixed threshold confounds architecture with calibration —
a model whose scores sit higher fires more on weapons *and* on parked cars.

**Change.** Thresholds are swept post-hoc (0.1–0.8) and models compared at
thresholds calibrated to a common pre-onset false-alarm rate.
Implemented in `scripts/threshold_sweep.py`.

This reverses the fixed-threshold ranking and **retracts** any claim that
yolov8s is the most sensitive model: at 0.5 it leads on hit rate (66.0 %) with
a 51.9 % pre-onset false-alarm rate, but at matched FP it is the weakest or
second weakest of the four.

## D7 — The 30 s timeout of §2 is replaced by end-of-observation

**Reason.** §2 defines a miss as "no alarm within 30 s of the onset". That
threshold is never reached: SCVD clips end a median of 3.37 s after the onset,
so the timeout never binds and no run was ever terminated by it. Reporting a
"miss rate within 30 s" would imply 30 s of evidence that does not exist.

**Change.** The observation window ends at the last processed frame. The
metric is renamed accordingly:

| §2 (frozen) | reported as |
|---|---|
| miss rate within 30 s | **no alarm before end of observation** |
| TTD (alarm only) | P(alarm by t), t up to the censoring horizon |

The 30 s timeout stays in the code as a safety bound and is documented as
inactive for this dataset. Any future dataset with longer post-onset footage
makes it active again without a code change.

**Consequence for the paper.** Absolute detection rates are conditional on
~3.4 s of post-onset footage and are NOT comparable to studies that allow
longer observation. Comparisons *between* configurations remain valid because
follow-up is non-differential (3.32–3.57 s median across configurations).

## D8 — Attrition: repair-and-rerun, not exclusion alone (pending)

D1's exclusion is a stopgap. Because attrition correlates with the
experimental condition (7.2 % local vs 38.4 % 4G), the surviving runs cannot
be *proven* unbiased — and if the failure is a startup race, the survivors are
those whose startup was fast, which is plausibly correlated with node load and
therefore with the latency behaviour being measured.

Plan: diagnose from the PBS stderr, fix the harness, re-run the 775 failed
cells (`hpc/rerun/cells_failed.txt`) into a separate results directory, and
re-run a 50-cell control sample of cells that already succeeded
(`hpc/rerun/cells_control.txt`) to test whether the fix perturbs the
measurement path. Repaired and frozen cells are merged **only if** the control
sample's latency distributions match the frozen run; otherwise the experiment
is re-run in full. Generated by `scripts/rerun_failed.py`.

Note on the failure-rate ordering: local-gpu < edge-sim < {remote-lan ~
remote-wifi} < remote-4g. Every step is significant except lan vs wifi
(z = 1.41, p = 0.158, Wilson CIs [25.0, 31.7] vs [21.8, 28.2]), so the
apparent lan > wifi inversion is noise and does not contradict a
startup-related mechanism.

## Unchanged

§2 (design, deployments, network profiles, matrix, seeds), the operational
alarm definition, the measurement definition, the five-component decomposition
and the TTD identity, timestamping, and the raw logs themselves.

---

## Addendum 2026-08-03 — the cause of the D1 exclusions is now known

Added **after** the plan above was written; the plan's text is left untouched
as the dated pre-registration it is. Full analysis:
[`DIAGNOSTIKA-VYPADKU.md`](DIAGNOSTIKA-VYPADKU.md).

The 775 zero-frame runs are **not** of unknown origin. PBS output was
retrieved and categorised: 775/775 died at `mtx.start()` on
`listen udp :8000: bind: address already in use`, before a single frame was
decoded. `mediamtx` binds fixed UDP ports :8000/:8001 that the config template
never parameterises, and the retry loop randomises only the TCP port. The
trigger is the scheduler placing a **second concurrent job on the node**;
the mechanism was reproduced off-cluster with the same binary.

Two claims in D1 above are superseded by this:

1. **"Failure rate is strongly configuration-dependent"** — it is not. The
   ordering local-gpu < edge-sim < lan ~ wifi < 4G is a confound with time:
   configuration is the slowest-varying factor in the cell enumeration, so
   local-gpu (indices 0–679) ran while nodes were still single-slot and 4G
   (2720–3399) ran when they were not. In the single-slot regime **every
   configuration failed 0 %** (0/1597 runs); in the two-slot regime all of
   them fail 35–48 % with no monotone ordering.
2. **The two significant results reported as a limitation** — the narrow
   model effect (p = 0.0074) and the scenario dependence within remote-lan
   (p = 1.3e-8) — **both vanish** once the concurrency regime is conditioned
   on: p = 0.627 and p = 0.963 respectively. Scenario overall goes from
   p = 0.157 to p = 0.995.

The exclusion itself stands, and its justification is now mechanistic rather
than merely empirical: the failure precedes any contact with the video, the
model, or the alarm logic, so independence from content is a necessity of the
mechanism and not an observed coincidence. The missingness is MAR conditional
on configuration (not MCAR — probability of failure depends on *when* a cell
ran, which is tied to configuration), which does not bias estimates computed
within configuration; it costs precision, unequally across configurations.
Wording for the limitations section is in §6 of the diagnostic protocol.

A one-line fix (`protocols: [tcp]`, verified measurement-neutral because both
ends of the chain already force TCP) removes the failure mode; see §7–§8 there
for the rerun cost/benefit.

## Open — requires supervisor sign-off before the final numbers are written up

- D2 (survival formulation) replaces the median-TTD tables of §3.
- D4 (pairwise + Bonferroni over 10 pairs) as the secondary paired analysis.
- Double annotation of onsets is still outstanding; onset uncertainty
  (±2–3 frames ≈ 66–100 ms) is of the same order as the differences between
  configurations at the fast end and must be reported as a systematic
  component of measurement uncertainty.
