"""What a confidence means: the calibrated confidence, the decline threshold and the held-out
record of answers at that confidence.

The Platt calibrator and the decline threshold were fitted on the calibration split; the bands
are the reliability table of the held-out split, where nothing was fitted. A band's interval is
the Wilson 95% interval of its held-out accuracy. A band with few answers says so rather than
give a rate that looks firmer than it is.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

Z = 1.959963984540054  # the 95% normal quantile
ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = ROOT / "configs/serving.yaml"


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def wilson(k: int, n: int, z: float = Z) -> list[float] | None:
    """Wilson score interval for k of n (as src/stats/report.py; kept here so the service needs
    no numerical library); None when n is 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return [round(lo, 4), round(hi, 4)]


@dataclass(frozen=True)
class Band:
    low: float
    high: float
    n: int
    correct: int
    interval: tuple[float, float] | None

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None

    def to_dict(self, min_n: int) -> dict[str, Any]:
        return {
            "range": [self.low, self.high],
            "n": self.n,
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "interval": None if self.interval is None else list(self.interval),
            "thin": self.n < min_n,
        }


class Meter:
    """Maps a stated confidence to its calibrated value, its band and its status."""

    def __init__(self, calibration: dict[str, Any], cfg: dict[str, Any] | None = None):
        cfg = cfg or config()
        self.min_n = cfg["meter"]["min_band_n"]
        platt = calibration["calibration_split"]["calibrators"]["platt"]
        self.intercept, self.slope = platt["intercept"], platt["slope"]
        held = calibration["decline"]["held_out"]
        self.threshold = held["threshold"]
        self.fitted_on = calibration["calibration_split"]["questions"]
        self.target_accuracy = calibration["decline"]["target_accuracy"]
        self.held_out = {
            "questions": held["questions"],
            "answered": held["answered"],
            "accuracy": held["accuracy"],
            "coverage": held["coverage"],
        }
        self.larger: dict[str, float] | None = None  # a router's larger model's Platt calibrator
        self.bands = self._bands(calibration["held_out"]["stated_platt"]["reliability"])

    def _bands(self, reliability: list[dict[str, Any]]) -> list[Band]:
        return [
            Band(
                low=b["range"][0],
                high=b["range"][1],
                n=b["n"],
                correct=round(b["accuracy"] * b["n"]),
                interval=tuple(wilson(round(b["accuracy"] * b["n"]), b["n"]) or ()) or None,
            )
            for b in reliability
        ]

    @classmethod
    def load(cls, cfg: dict[str, Any] | None = None) -> Meter:
        cfg = cfg or config()
        path = ROOT / cfg["meter"]["calibration"]
        return cls(json.loads(path.read_text(encoding="utf-8")), cfg)

    @classmethod
    def from_registry(
        cls, resolved: dict[str, Any], evaluation: dict[str, Any], cfg: dict[str, Any] | None = None
    ) -> Meter:
        """The meter of a registered agent configuration: its calibrators and decline threshold
        (fitted and chosen on the calibration split) and the held-out record of its answers, from
        its evaluation file. For the configuration without a router this is exactly the meter
        `load` builds from the calibration results."""
        conf = evaluation["confidence"]
        held = conf["held_out"]
        calibration = {
            "calibration_split": {
                "calibrators": {"platt": resolved["calibration"]["calibrator"]},
                "questions": resolved["calibration"]["fitted_on_questions"],
            },
            "decline": {
                "target_accuracy": resolved["calibration"]["target_accuracy"],
                "held_out": {
                    "threshold": resolved["decline_threshold"],
                    "questions": held["questions"],
                    "answered": held["answered"],
                    "accuracy": held["accuracy"],
                    "coverage": held["coverage"],
                },
            },
            "held_out": {"stated_platt": {"reliability": held["reliability"]}},
        }
        meter = cls(calibration, cfg)
        if resolved["router"]:
            meter.larger = resolved["router"]["calibrator"]
        return meter

    def calibrate(self, stated: float, larger: bool = False) -> float:
        """A stated confidence made calibrated: by the primary model's Platt curve, or (`larger`)
        by the router's larger model's."""
        if larger:
            if self.larger is None:
                raise ValueError("this meter has no larger model")
            slope, intercept = self.larger["slope"], self.larger["intercept"]
        else:
            slope, intercept = self.slope, self.intercept
        return 1.0 / (1.0 + math.exp(-(slope * float(stated) + intercept)))

    def band(self, calibrated: float) -> Band | None:
        """The held-out band a calibrated confidence falls in (a band's upper end is open)."""
        for b in self.bands:
            if b.low <= calibrated < b.high:
                return b
        # above the highest band the held-out answers reach: no held-out record at all
        return None

    def describe(
        self, stated: float | None, declined: bool, larger: bool = False
    ) -> dict[str, Any]:
        """The confidence block of an evidence record. `larger`: the answer is a router's larger
        model's, so its confidence is calibrated by that model's own curve."""
        if declined or stated is None:
            return {
                "stated": stated,
                "calibrated": None,
                "band": None,
                "threshold": self.threshold,
                "withheld": False,
            }
        cal = self.calibrate(stated, larger)
        band = self.band(cal)
        return {
            "stated": stated,
            "calibrated": round(cal, 4),
            "band": None if band is None else band.to_dict(self.min_n),
            "threshold": round(self.threshold, 4),
            "withheld": cal < self.threshold,
        }

    def summary(self) -> dict[str, Any]:
        """What the meter's wording rests on, for the client."""
        return {
            "threshold": round(self.threshold, 4),
            "held_out": self.held_out,
            "bands": [b.to_dict(self.min_n) for b in self.bands],
            "min_band_n": self.min_n,
            # how the calibrated confidence is made, for the page to explain and to work an example
            "calibration": {
                "method": "platt",
                "slope": self.slope,
                "intercept": self.intercept,
                "fitted_on": self.fitted_on,
                "target_accuracy": self.target_accuracy,
            },
        }
