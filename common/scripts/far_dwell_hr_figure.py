#!/usr/bin/env python3
"""Figure 6: hazard ratio against requested dwell, and alarm probability against
realised dwell K/f (variant a, 120 false alarms per camera-hour).

Left: HR of each deployment configuration relative to local-gpu for each
requested dwell t (clustered Cox, 95 % CI). The region t < 1/f_min is shaded:
there K cannot fall below one on the slowest configuration, so it enforces
K/f > t and the comparison is not interpretable. Open markers at the left show
the pre-registered K = 3 frame rule, whose realised dwell differs per
configuration (0.1-1.7 s) and so has no position on the t axis.
Right: pooled Kaplan-Meier P(alarm within 3 s) against the realised dwell K/f.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from far_dwell import F_MIN, OUT  # noqa: E402
# house style of the Scientific Reports figures (fonts, TrueType embedding, colours)
from scirep_figures import COLOR, MM, panel, plt  # noqa: E402

CFG = ["remote-lan", "remote-wifi", "remote-cellular-4g", "edge-sim"]
TGT = 120.0


def main() -> int:
    cx = pd.read_csv(OUT / "phase2_far_dwell_cox.csv", index_col=0)
    cx = cx[(cx.variant == "a") & (cx.target_far_h == TGT)]
    pl = pd.read_csv(OUT / "phase2_far_dwell_pooled.csv")
    pl = pl[(pl.variant == "a") & (pl.target_far_h == TGT)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(183 * MM, 68 * MM))
    tmin = 1 / F_MIN
    for c in CFG:
        g = cx[(cx.index == "config_label_" + c) & (cx["mode"] != "fixed3")]
        g = g.sort_values("t_requested_s")
        a.plot(g.t_requested_s, g.HR, "-o", ms=2.5, color=COLOR[c], label=c)
        a.fill_between(g.t_requested_s, g.CI_lo, g.CI_hi, color=COLOR[c], alpha=.12, lw=0)
        f3 = cx[(cx.index == "config_label_" + c) & (cx["mode"] == "fixed3")].iloc[0]
        a.plot([0.07], [f3.HR], "o", mfc="none", color=COLOR[c], ms=4)
    a.axvspan(0.05, tmin, color="0.93", zorder=0)
    a.text(0.2, 12, "not enforceable\n(t < 1/f$_{min}$ = %.2f s)" % tmin, ha="center")
    a.axhline(1, color="k", lw=.7)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.05, 1.7)
    a.set_xlabel("Requested dwell t (s); open markers: K = 3 frames")
    a.set_ylabel("Hazard ratio vs local-gpu (120/h)")
    a.legend(fontsize=7, frameon=False, loc="lower right")
    for c in ["local-gpu", *CFG]:
        g = pl[(pl.config == c) & (pl["mode"] != "fixed3")].sort_values("realised_dwell_s")
        b.plot(g.realised_dwell_s, g.P_alarm_3s, "-o", ms=2.5, color=COLOR[c], label=c)
    b.set_xscale("log")
    b.set_xlabel("Realised dwell K/f (s)")
    b.set_ylabel("P(alarm within 3 s), pooled detectors")
    b.set_ylim(0, 0.7)
    b.legend(frameon=False, loc="upper left")
    for x, letter in ((a, "a"), (b, "b")):
        x.grid(alpha=.25, linewidth=0.4)
        panel(x, letter)
    fig.tight_layout()
    f = OUT / "fig_hr_vs_dwell.pdf"
    fig.savefig(f); fig.savefig(f.with_suffix(".png"), dpi=150)
    print("wrote", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
