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

## D9 — Cross-model claims are calibrated to false alarms per camera-hour on a disjoint alarm-free corpus (added 2026-08-05)

**Reason.** D6 matched models at a common *pre-onset false-positive rate*
measured on the same 34 event clips on which TTD is then evaluated. That is
calibration on the evaluation data, and the calibration basis is thin: the
unique pre-onset footage behind the 30 % FP target is **85.8 s** (34 clips,
median 2.56 s before onset, range 0.5–6.0 s). A per-run FP probability over a
~2.5 s window is also not an operational quantity — control-room practice uses
**false alarms per camera-hour**.

**Corrected 2026-08-16 against the source.** This deviation previously cited
"FAR per hour reported by only 2 of 35 studies" from our own survey. The
survey (Havacek et al., AI Review 2026, doi:10.1007/s10462-026-11653-z) says
something different and stronger: of 36 representative studies, TTD is
reported by **0 %**, FAR-type false-alarm reporting appears in **3 of 36
(8 %)**, and **no study at all reports false alarms per operational hour**.
The 2/35 figure was wrong in both numbers and substance and must not appear
anywhere in the paper.

**Change.** Thresholds are calibrated post hoc on a **disjoint alarm-free
corpus**: all 246 SCVD *Normal*-class clips (200 train + 46 test,
SCVD_converted, all 30 fps), total **0.2728 h**. UCF-Crime contributes
nothing — the local copy holds only anomaly categories (RoadAccidents,
Robbery, Shooting; verified in both the extracted tree and the archive), so
there is **no** normal-class UCF-Crime footage to add without expanding the
data beyond what exists on disk.

Disjointness from the 34 event clips is established three ways (script
`common/scripts/far_corpus.py`, report
`results-far-calib/disjointness_report.txt`): (1) by construction — event
clips come exclusively from `Weaponized/` class folders, the corpus
exclusively from `Normal/`; (2) no basename overlap; (3) no MD5 overlap.

**Unit and rule.** FAR = alarm *episodes* per camera-hour: an episode starts
when K = 3 consecutive processed frames reach the threshold and re-arms on
the first frame below it; streaks do not cross clip boundaries. Calibration
inference is offline score collection (no mediamtx/RTSP/network emulation,
hardware-independent), logged from conf 0.1 like the frozen Phase-2 logs
(`far_infer.py`; ultralytics backend, the Phase-2 local-gpu path).
**Corrected 2026-09-27 (D11):** same backend, but not the same input size for
yolov8s — calibration scores every model at 640, whereas the Phase-2
local-gpu/remote replay ran yolov8s at its stored 416. Left as written above
so the record shows what was assumed.

**What 0.27 h can and cannot resolve.** A threshold with zero observed
episodes only supports "FAR < 11/h" (95 % one-sided); ±50 % relative
precision needs FAR ≥ ~59/h. The original plan's 0.1/h and 1/h targets would
need ≥ 30 h / 3 h of footage for a mere zero-count bound and ~160 h / 16 h
for a ±50 % estimate — **they are not calibratable from this corpus and are
not reported.** The sweep grid is 12 / 30 / 60 / 120 / 300 / 600 alarms/h;
the 12/h point rests on 2–3 episodes and is reported as a bound, not an
estimate. Uncertainty: Garwood exact Poisson CI plus a clip-level bootstrap
(episodes cluster within clips; the bootstrap is authoritative). All curves
in `far_curves.parquet`, calibrated thresholds in `thresholds.csv`.

**What the old operating points mean in the new unit.** The published fixed
conf 0.5 corresponds to **271–2438 false alarms per camera-hour** at 30 fps
(v26m 271, v12m 645, v8m 1239, v8s 2438). The D6 "matched FP 30 %"
thresholds (0.195–0.645) correspond to **1005–2082/h** — and differ between
models by a factor of ~2, i.e. the D6 comparison was not actually matched in
operational units (v8m sat at 2082/h while v26m sat at 1005/h).

**Published numbers, old vs new (pooled over configurations, defect cell of
D10 excluded everywhere; `old_vs_new.csv`, `phase2_far_summary.csv`):**

| model | fixed 0.5 hit % / P(≤1 s) | FP30 (D6) hit % / P(≤1 s) | FAR 120/h (b) | FAR 30/h (b) |
|---|---|---|---|---|
| yolov8s  | 66.0 / 0.458 | 38.5 / 0.154 | 14.5 / 0.106 | 6.4 / 0.055 |
| yolov8m  | 38.2 / 0.237 | 56.9 / 0.368 | 13.9 / 0.081 | 5.1 / 0.027 |
| yolov12m | 20.4 / 0.105 | 33.0 / 0.198 |  6.0 / 0.033 | 3.1 / 0.020 |
| yolov26m | 17.1 / 0.083 | 50.5 / 0.239 |  8.0 / 0.036 | 2.2 / 0.000 |

**Does the D6 ranking survive?** No. The D6 order
(v8m > v26m > v12m > v8s at matched FP) occurs at **no point** of the FAR
sweep. At 12–60/h, in both variants, v8m and v8s lead, v12m is third and
v26m last — v26m's D6 rank was carried by the saturated edge-sim cell of D10
and disappears with it. From 120/h up, v8s and v8m stay on top in variant
(b), while v12m and v26m swap places depending on variant and target; the one
point where v26m leads is 600/h variant (a), at 33.2 %.

The more consequential casualty is D6's own retraction. D6 declared yolov8s
"the weakest or second weakest of the four" at matched FP; at FAR-matched
operating points **v8s is first or second at 10 of the 12 sweep points**. The
D6 conclusion was an artifact of calibrating on a 2.5 s pre-onset window of
the evaluation clips, and should be retracted in turn rather than carried
into the paper.

Two things are stable across the whole sweep: yolov12m never leaves the
bottom two, and no model reaches 15 % hit rate at FAR ≤ 120/h. Below
~120/h the model differences are **not statistically separable**
(clustered-Cox CIs span 1 by a wide margin, e.g. v8m vs v8s at 30/h:
HR 1.05 [0.30, 3.67]); the honest statement is that 0.27 h of calibration
footage ranks models only coarsely.

**Consequence for the paper.** At every honestly-calibrated operating point
the absolute hit rates collapse to 2–15 % (conditional on ~3.4 s of
post-onset footage, D7) — against 38–66 % at the published operating points,
whose implied false-alarm load (≥ 270/h per camera) no control room would
accept. The FAR-per-hour framing converts the D6 fix from a patch into a
contribution (2/35 studies report it), but the numbers it produces are much
smaller than the frozen §3 tables.

**Limitations.** (i) The corpus is 246 short clips (median 4.2 s), so
per-hour rates are extrapolated from second-scale snippets and episode
streaks reset at clip boundaries; (ii) frame-rate emulation uses uniform
stride (edge-sim median 1.76 fps → stride 17 at 30 fps), while real loss is
bursty; (iii) calibration runs one backend (ultralytics .pt) — see the
backend spot-check in `results-far-calib/backend_check.csv`; (iv) SCVD
Normal is domain-close CCTV but contains no long uneventful surveillance
recordings; a production calibration would need tens of hours per site.

## D10 — the ONNX backend misparses end2end exports; yolov26m × edge-sim is invalid and excluded (added 2026-08-05)

**Found while implementing D9.** In the frozen Phase-2 logs, every detection
frame of yolov26m under edge-sim (the ONNX-on-CPU backend) carries
max_conf = 1.000 — all 1318 detection frames, against a mean detection
confidence of 0.37–0.39 for the same model on every other configuration.

**Root cause — a parsing bug in our harness, not a bad checkpoint.**
`OnnxBackend.infer` (`common/src/ttdbench/inference.py:212-219`) hard-codes
the YOLOv8 raw output layout `(1, 4+nc, N)` and transposes it. The yolov26m
export is **end2end** (NMS baked in): its output is `(1, 300, 6)`, i.e.
`[x1, y1, x2, y2, conf, cls]` per detection. Transposing gives a `(6, 300)`
array whose six *rows* are then read as six detections: the class-id row has
maximum 1.0 and its argmax index falls into `names` as 0/1, so **every frame
with any detection at all yields a phantom "Knife"/"Handgun" at conf exactly
1.000**, with a degenerate box (e.g. `[1.0, -279.0, 3.0, -277.0]`), while the
coordinate rows are logged as `class142` etc. at conf ≈ 660.

Verified three ways: (1) the ONNX file itself is fine — read through
ultralytics it agrees with the .pt checkpoint to a mean absolute confidence
difference of 0.024 and 98.96 % threshold agreement at 0.5
(`backend_check.csv`); (2) output shapes confirm the layout split — v8s/v8m/
v12m are `(1, 6, 8400)`, v26m is `(1, 300, 6)`; (3) running our own
`OnnxBackend` over **alarm-free** corpus footage containing no weapon
reproduces the artifact: 11 of 20 frames report a weapon at conf 1.000 for
v26m, 0 of 20 for v8m.

The alarm therefore fires on the presence of *any* detection, not on a
weapon. No threshold in (0, 1] can separate anything, the cell shows 91.8 %
hit rate at *every* threshold, and it cannot be re-thresholded at all.

**Consequences for already-computed numbers (defect cell = 134 valid runs):**

| quantity (conf 0.5) | with defect cell (as frozen) | without |
|---|---|---|
| edge-sim KM row: hit % | 38.5 | 22.4 |
| edge-sim KM row: P(alarm ≤ 1 s) | 0.250 | 0.131 |
| yolov26m pooled hit % | 32.8 | 17.1 |

Roughly **40 % of the apparent edge-sim detection performance in the frozen
KM table is this artifact**, and it is the reason v26m ranked second in the
D6 FP-matched comparison. All D9 quantities exclude the cell; the frozen
logs are untouched.

**Phase 1 is affected too.** `rtdetr.onnx` is also an end2end export
(`(1, 300, 6)`) and rtdetr is in the Phase-1 active models with an
onnxruntime deployment, so `results/phase1-final` edge-sim × rtdetr logs
carry the same garbage detections (`sandwich` at conf 1.0, `class238` at
conf 67.0). Phase 1 reports latency, not detection, so its headline numbers
survive — but the post-processing component is measurably distorted: NMS runs
over 6 misparsed rows instead of 300 detections, giving
`pf_dt_post_ms = 0.28` for edge-sim × rtdetr against 1.58–1.65 for the
correctly-parsed models on the same configuration and 5.23 for rtdetr on
local-gpu (ultralytics backend). The effect on total latency (855 ms) is
negligible; the post-processing column for that one cell should not be
reported as a measurement.

**Fix APPLIED 2026-08-10** (commit 018b3d7): `OnnxBackend` dispatches on the
output layout — `(1, N, 6)` is read row-wise with no transpose and no second
NMS — with unit tests for both layouts and a check on real weapon-free footage
(`common/scripts/verify_onnx_decode.py`). Measured there: the old path reports
a detection at conf ≥ 0.999 on **60 of 60** frames for yolov26m, the new path
on **0 of 60**; the three raw-head models are unaffected, as they must be. The
fix exists for paper 3, which runs on this code. **The frozen cell stays
excluded either way** — re-running it would be new data collection against a
frozen design.

**Consequences of the exclusion, computed 2026-08-10** (derived layer
`results-d10/`, see its README; 170 runs dropped, 2491 analysed):

1. **edge-sim moves from mid-table to worst.** Cox HR 0.752 (p = 0.11) →
   **0.3158 (p < 0.0001)**, below remote-cellular-4g (0.4967). The thesis of
   the results section changes from "the effect is not monotonic in link
   quality" to "the compute-bound constraint dominates the network one", which
   is what Phase 1 predicted all along (94–96 % frame loss on edge-sim).
   Pairwise, edge vs 4G is **not** significant (log-rank p = 0.0592), so this
   is the worst point estimate, not a demonstrated worst.
2. **"The 1 s window destroys edge-sim" was the artifact.** Frozen it read
   38.5 % → 16.3 %; excluded it reads 22.4 % → 21.3 %, i.e. −1.1 pp. Phantom
   detections satisfied the consecutive rule but not the windowed one, which
   manufactured the collapse. The surviving claim is that the window buys fast
   pipelines 6–12 pp and edge-sim nothing.
3. **The "~275 ms detector constant" holds for three configurations, not
   four.** edge-sim moves 272.4 → 168.2 ms on 99 surviving hits of 442 runs —
   a selected subset, since the scenarios where a ~2 fps pipeline still lands
   three consecutive qualifying frames are the easy ones. The residual split
   is conditional on a hit by construction and therefore inherits exactly the
   survivorship bias the survival analysis was introduced to remove; it must
   not be compared across configurations whose hit rates differ twofold.
4. **Unaffected:** the local−4G gap (17.6 pp at K=3, 12.4 pp at K=1) and all
   local-gpu / remote-lan / remote-wifi rows.

Scope: D10 is an analysis-level exclusion applied *after* the D1 validity
filter. The campaign disposition (3400 submitted, 775 zero-frame, 2625 valid)
describes the campaign and stays on the full set.

## D11 — imgsz mismatch: yolov8s ran at 416 outside edge-sim, but was calibrated at 640 (discovered 2026-09-27)

**Found** during the Sci Rep pre-submission check, after all analyses. Full
audit, scripts and intermediates: [`IMGSZ-AUDIT-2026-09-27.md`](IMGSZ-AUDIT-2026-09-27.md)
and `imgsz-audit-2026-09-27/`.

**What happened.** `UltralyticsBackend.infer` calls `predict()` without
`imgsz`. For a `.pt` checkpoint ultralytics 8.4.103 keeps the checkpoint's
training `imgsz` (`Model._reset_ckpt_args`), so the default 640 applies only
when none is stored. yolov8s-weapon stores 416. In Phase 2 yolov8s therefore
ran at **416** on local-gpu and remote (lan/wifi/4g), and at **640** on
edge-sim (the ONNX export has a fixed 640 input); yolov8m/v12m/v26m ran at 640
everywhere. The D9 calibration scored all four at 640. For yolov8s the
thresholds calibrated at 640 are applied to 416 scores in 4 of 5
configurations. Phase 1 (COCO checkpoints, 640) is unaffected. Confirmed
independently by matching the frozen replay traces to offline scores: r = 0.954
with 416 vs 0.541 with 640 on local-gpu; edge-sim the reverse (0.493 vs 0.679).

**Measured local impact (indicative only; MPS, 34 event clips, 0.483 h corpus
subset; `common/scripts/scirep_imgsz_facts.py` →
`results/scirep-stats/imgsz_impact.txt`).** Decision disagreement 416 vs 640
at 0.5 on event frames 30.8 %. Achieved v8s FAR at the paper thresholds is
0.36–4.5 × nominal (higher at 12–120/h, lower at 300/h, 600/h and fixed 0.5).
local-gpu v8s hit rate with 416-matched thresholds: 120/h 5.9 % (unchanged),
300/h 32.4 % [8.8, 44.1] vs 17.6 %, 600/h 64.7 % [55.9, 73.5] vs 32.4 %.

**Handling: option (c), approved by the author 2026-09-27.** Reported as a
limitation; no cluster, no re-scoring, no frozen number or generated table
changed. Manuscript: the false sentence "inference was performed at 640 pixels
for all models" corrected; limitation paragraph added; claims about v8s at
≥ 300/h (ranking, v8m-vs-v8s HR at 600/h, the 1248/h at 0.5 as the deployed
rate) qualified; new SI Supplementary Note "Input size of YOLOv8s (deviation
D11)" with the numbers above; D11 row in the SI deviation table.

**Pending:** option (a), re-scoring the calibration corpus with yolov8s at 416
on MetaCentrum (~10–20 GPU-h L40) and recomputing variant a/b for v8s, awaits
the author's decision. Option (b), re-running the replay at 640, is not
recommended (changes frozen primary data).

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
  **Measured 2026-09-21, and the ±2–3 frames above was wrong by an order of
  magnitude.** Blind second annotation of 17 clips: usability agreement 13/17,
  median |difference| 25 frames (833 ms), mean 37.77, bias −30.54 frames
  (`results/annotation-agreement/compare.txt`). The pre-registered sensitivity
  analysis (paper §3.7.6, §4.4.5; `results/onset-sensitivity/`) keeps the
  configuration ordering and the wifi, 4G and edge-sim HRs; LAN is flagged on
  sign only, never significant. Absolute levels move. Left as written above
  so the record shows what was assumed.
- D9/D10 as a whole (new calibration unit, corpus, exclusions).
- **D9 frame-rate variants (a) vs (b)** — the K = 3 consecutive rule spans
  ~100 ms at 30 fps but ~1.7 s at the edge configuration's 1.76 fps, so one
  threshold does not give one operating point across configurations. Both
  variants are computed in `results-far-calib/thresholds.csv`:
  **(a)** one threshold per model, calibrated at full frame rate; the FAR it
  yields per configuration is a derived quantity and differs by roughly an
  order of magnitude (e.g. yolov8m at target 120/h: 106/h local-gpu but
  18/h edge-sim, 44/h 4G). Configurations then share the *detector* operating
  point, and "matched operating point" means matched threshold — the
  configuration comparison keeps the network as the only moving part, but
  operators of different deployments live at different false-alarm loads.
  **(b)** one threshold per model *and* configuration at equal FAR (e.g.
  yolov26m at 120/h: thr 0.745 local vs 0.130 edge). Operators share the
  false-alarm load, and "matched operating point" means matched alarm
  economics — but the configuration comparison now moves two things at once
  (network *and* threshold), so a TTD difference between configurations is no
  longer attributable to the deployment alone. Decision deferred to the
  supervisor consultation; every downstream table exists for both variants.
