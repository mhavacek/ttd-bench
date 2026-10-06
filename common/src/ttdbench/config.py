"""Configuration loading and deterministic experiment-matrix enumeration.

A *cell* is one point of the experiment matrix:
    (deployment, network_profile | None, model, scenario, repetition)

Cells are enumerated in a fixed, documented order so that an HPC array index
maps to exactly one cell on every machine and every run:

    for deployment in deployments:              # config order
        for profile in deployment.network_profiles or [None]:
            for model in active_models:         # config order
                for scenario in active_scenarios:
                    for rep in range(repetitions):
                        yield Cell(...)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class Cell:
    index: int
    deployment: str
    backend: str
    device: str
    cpu_threads: Optional[int]
    remote: bool
    network_profile: Optional[str]  # None unless remote
    model: str
    scenario: str
    repetition: int
    seed: int

    def run_id(self) -> str:
        prof = self.network_profile or "none"
        return (
            f"{self.deployment}__{prof}__{self.model}__{self.scenario}"
            f"__rep{self.repetition}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]
    root: Path
    scenarios: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_scenarios: list[str] = field(default_factory=list)
    network_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        root = path.parent.parent if path.parent.name == "configs" else path.parent
        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg = cls(raw=raw, root=root)

        scen_path = root / raw["scenarios_file"]
        with open(scen_path) as f:
            scen_raw = yaml.safe_load(f)
        cfg.scenarios = {s["id"]: s for s in scen_raw["scenarios"]}
        cfg.active_scenarios = scen_raw.get(
            "active_scenarios", list(cfg.scenarios.keys())
        )

        net_path = root / raw["network_profiles_file"]
        with open(net_path) as f:
            net_raw = yaml.safe_load(f)
        cfg.network_profiles = net_raw["profiles"]

        return cfg

    # -------------------------------------------------------------- shortcuts
    @property
    def repetitions(self) -> int:
        return int(self.raw["repetitions"])

    @property
    def seed_base(self) -> int:
        return int(self.raw["seed_base"])

    @property
    def alarm(self) -> dict[str, Any]:
        return self.raw["alarm"]

    @property
    def active_models(self) -> list[str]:
        return list(self.raw["active_models"])

    def model_spec(self, name: str) -> dict[str, Any]:
        for m in self.raw["models"]:
            if m["name"] == name:
                return m
        raise KeyError(f"unknown model {name!r}")

    def scenario_spec(self, scenario_id: str) -> dict[str, Any]:
        return self.scenarios[scenario_id]

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "experiment": self.raw,
                "scenarios": self.scenarios,
                "active_scenarios": self.active_scenarios,
                "network_profiles": self.network_profiles,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ------------------------------------------------------------ enumeration
    def enumerate_cells(self) -> list[Cell]:
        cells: list[Cell] = []
        idx = 0
        for dep in self.raw["deployments"]:
            profiles = dep.get("network_profiles") if dep.get("remote") else None
            for profile in profiles or [None]:
                for model in self.active_models:
                    for scenario in self.active_scenarios:
                        for rep in range(self.repetitions):
                            cells.append(
                                Cell(
                                    index=idx,
                                    deployment=dep["name"],
                                    backend=dep["backend"],
                                    device=str(dep.get("device", "cpu")),
                                    cpu_threads=dep.get("cpu_threads"),
                                    remote=bool(dep.get("remote", False)),
                                    network_profile=profile,
                                    model=model,
                                    scenario=scenario,
                                    repetition=rep,
                                    seed=self.seed_base + rep,
                                )
                            )
                            idx += 1
        return cells

    def cell(self, index: int) -> Cell:
        cells = self.enumerate_cells()
        if not 0 <= index < len(cells):
            raise IndexError(
                f"cell index {index} out of range (matrix has {len(cells)} cells)"
            )
        return cells[index]
