import random
import math
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from models import WaterLevelReading, DataSource


@dataclass
class SimulatorConfig:
    node_id: str = "NODE-001"
    interval_ms: int = 1000
    start_level: float = 20.0
    rise_rate: float = 0.5
    noise: float = 0.3
    mode: str = "rising"


class WaterLevelSimulator:
    PRESETS: Dict[str, Dict] = {
        "demo": {
            "start_level": 20,
            "rise_rate": 0.5,
            "noise": 0.2,
            "interval_ms": 1000
        },
        "rapid": {
            "start_level": 25,
            "rise_rate": 3,
            "noise": 0.5,
            "interval_ms": 800
        },
        "slow": {
            "start_level": 25,
            "rise_rate": 0.3,
            "noise": 0.1,
            "interval_ms": 1500
        },
        "stable": {
            "start_level": 22,
            "rise_rate": 0.05,
            "noise": 0.3,
            "interval_ms": 1000
        }
    }

    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()

        self.node_id = self.config.node_id
        self.interval_ms = self.config.interval_ms
        self.mode = self.config.mode
        self.current_level = self.config.start_level
        self.start_level = self.config.start_level
        self.rise_rate = self.config.rise_rate
        self.noise = self.config.noise

        self._callbacks: List[Callable] = []
        self._running = False
        self._interval_id: Optional[object] = None
        self._reading_count = 0

    @property
    def is_active(self) -> bool:
        return self._running

    def use_preset(self, preset_name: str) -> None:
        if preset_name not in self.PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available: {list(self.PRESETS.keys())}"
            )

        preset = self.PRESETS[preset_name]

        self.start_level = preset["start_level"]
        self.current_level = preset["start_level"]
        self.rise_rate = preset["rise_rate"]
        self.noise = preset["noise"]
        self.interval_ms = preset["interval_ms"]
        self.mode = preset_name

        print(f"[Simulator] Using preset: {preset_name}")

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        print(f"[Simulator] Mode set to: {mode}")

    def on_data(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def _emit(self, reading: WaterLevelReading) -> None:
        for callback in self._callbacks:
            try:
                callback(reading)
            except Exception as e:
                print(f"[Simulator] Callback error: {e}")

    def start(self) -> None:
        if self._running:
            print("[Simulator] Already running")
            return

        print(f"[Simulator] Starting with mode: {self.mode}")
        print(f"[Simulator] Interval: {self.interval_ms}ms")
        print(f"[Simulator] Starting level: {self.start_level} cm")

        self._running = True
        self.current_level = self.start_level

        self._generate_reading()

    def stop(self) -> None:
        if not self._running:
            print("[Simulator] Not running")
            return

        self._running = False
        print(
            f"[Simulator] Stopped. "
            f"Total readings: {self._reading_count}"
        )

    def _generate_reading(
        self
    ) -> Optional[WaterLevelReading]:
        if not self._running:
            return None

        self._reading_count += 1

        self.current_level = self._calculate_next_level()

        noise_offset = (
            (random.random() - 0.5)
            * self.noise
            * 2
        )

        final_level = max(
            0,
            self.current_level + noise_offset
        )

        rounded_level = round(final_level, 1)

        reading = WaterLevelReading(
            node_id=self.node_id,
            water_level=rounded_level,
            timestamp=datetime.now(),
            source=DataSource.SIMULATOR
        )

        print(
            f"[Simulator] Reading "
            f"#{self._reading_count}: "
            f"{rounded_level} cm ({self.mode})"
        )

        self._emit(reading)

        return reading

    def _calculate_next_level(self) -> float:
        if self.mode == "rising":
            return self.current_level + self.rise_rate

        elif self.mode == "falling":
            return max(
                0,
                self.current_level - self.rise_rate
            )

        elif self.mode == "stable":
            drift = (
                (random.random() - 0.5)
                * 0.2
            )

            return self.current_level + drift

        elif self.mode == "fluctuating":
            wave = (
                math.sin(
                    self._reading_count * 0.3
                )
                * 1.5
            )

            return (
                self.start_level
                + wave
                + (random.random() - 0.5)
                * self.noise
            )

        elif self.mode == "rapid":
            return self.current_level + self.rise_rate

        elif self.mode == "emergency":
            return (
                self.current_level
                + self.rise_rate * 2
            )

        return self.current_level + self.rise_rate

    def inject_level(
        self,
        level: float
    ) -> WaterLevelReading:
        self.current_level = level

        reading = WaterLevelReading(
            node_id=self.node_id,
            water_level=level,
            timestamp=datetime.now(),
            source=DataSource.SIMULATOR
        )

        print(
            f"[Simulator] Injected level: "
            f"{level} cm"
        )

        self._emit(reading)

        return reading

    def reset(self) -> None:
        self.current_level = self.start_level
        self._reading_count = 0

        print(
            f"[Simulator] Reset to "
            f"{self.start_level} cm"
        )

    def get_stats(self) -> Dict:
        return {
            "active": self._running,
            "mode": self.mode,
            "current_level": self.current_level,
            "readings_generated": self._reading_count,
            "interval_ms": self.interval_ms
        }


class AsyncWaterLevelSimulator(
    WaterLevelSimulator
):
    def __init__(
        self,
        config: Optional[SimulatorConfig] = None
    ):
        super().__init__(config)
        self._task = None

    async def start_async(self) -> None:
        import asyncio

        if self._running:
            print("[Simulator] Already running")
            return

        print(
            f"[Simulator] Starting async "
            f"with mode: {self.mode}"
        )

        print(
            f"[Simulator] Interval: "
            f"{self.interval_ms}ms"
        )

        print(
            f"[Simulator] Starting level: "
            f"{self.start_level} cm"
        )

        self._running = True
        self.current_level = self.start_level

        self._generate_reading()

        while self._running:
            await asyncio.sleep(
                self.interval_ms / 1000
            )

            if self._running:
                self._generate_reading()

    def stop(self) -> None:
        self._running = False

        print(
            f"[Simulator] Stopped. "
            f"Total readings: {self._reading_count}"
        )