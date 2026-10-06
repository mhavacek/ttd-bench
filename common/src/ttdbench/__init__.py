"""TTD-Bench: empirical validation of Time-to-Detection decomposition.

TTD = t_alarm - t_onset = dt_acq + dt_transfer + dt_infer + dt_post + dt_alarm

All timestamps are time.monotonic_ns() on a single physical node (one clock
domain). See README.md, section "Clock-domain methodology".
"""

__version__ = "0.1.0"
