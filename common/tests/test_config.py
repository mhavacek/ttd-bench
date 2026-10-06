"""Config parsing and deterministic matrix enumeration."""

from pathlib import Path

import pytest

from ttdbench.config import ExperimentConfig

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    return ExperimentConfig.load(REPO / "configs" / "experiment.yaml")


def test_loads(cfg):
    assert cfg.repetitions >= 5
    assert cfg.seed_base == 42
    assert cfg.alarm["k_consecutive"] >= 1
    assert 0 < cfg.alarm["confidence_threshold"] < 1


def test_network_profiles(cfg):
    for name in ("lan", "wifi", "cellular-4g"):
        assert name in cfg.network_profiles
    assert cfg.network_profiles["lan"]["rtt_ms"] == 1.0
    assert cfg.network_profiles["cellular-4g"]["loss_pct"] == 1.0


def test_scenarios_have_onset(cfg):
    for sid in cfg.active_scenarios:
        s = cfg.scenario_spec(sid)
        assert s["t_onset_frame"] >= 0
        assert s["fps"] > 0
        assert "tags" in s


def test_matrix_size(cfg):
    cells = cfg.enumerate_cells()
    # deployments expand to: local-gpu + edge-sim + remote×3 profiles = 5 configs
    n_configs = 5
    expected = n_configs * len(cfg.active_models) * len(cfg.active_scenarios) * cfg.repetitions
    assert len(cells) == expected


def test_enumeration_deterministic(cfg):
    a = cfg.enumerate_cells()
    b = ExperimentConfig.load(REPO / "configs" / "experiment.yaml").enumerate_cells()
    assert [c.run_id() for c in a] == [c.run_id() for c in b]
    assert [c.index for c in a] == list(range(len(a)))


def test_seeds(cfg):
    for c in cfg.enumerate_cells():
        assert c.seed == 42 + c.repetition


def test_remote_cells_have_profiles(cfg):
    for c in cfg.enumerate_cells():
        if c.remote:
            assert c.network_profile in cfg.network_profiles
        else:
            assert c.network_profile is None


def test_cell_lookup_bounds(cfg):
    with pytest.raises(IndexError):
        cfg.cell(10**6)


def test_config_hash_stable(cfg):
    h1 = cfg.config_hash()
    h2 = ExperimentConfig.load(REPO / "configs" / "experiment.yaml").config_hash()
    assert h1 == h2
