from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import uuid


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DataSource(str, Enum):
    SIMULATOR = "simulator"
    SENSOR = "sensor"
    API = "api"
    FILE = "file"


@dataclass
class WaterLevelReading:
    node_id: str
    water_level: float
    timestamp: datetime = field(default_factory=datetime.now)
    source: DataSource = DataSource.SIMULATOR
    id: str = field(
        default_factory=lambda: str(uuid.uuid4())[:12]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "nodeId": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "waterLevel": self.water_level,
            "source": self.source.value
        }


@dataclass
class MonitoringNode:
    id: str
    name: str
    latitude: float = -6.2
    longitude: float = 106.8
    thresholds: Dict[str, int] = field(
        default_factory=lambda: {
            "watch": 30,
            "warning": 50,
            "critical": 70
        }
    )
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "thresholds": self.thresholds,
            "active": self.active
        }


@dataclass
class ProcessedReading:
    reading: WaterLevelReading
    raw_water_level: float
    smoothed_water_level: float
    processed_at: datetime = field(
        default_factory=datetime.now
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.reading.to_dict(),
            "rawWaterLevel": self.raw_water_level,
            "smoothedWaterLevel": self.smoothed_water_level,
            "processedAt": self.processed_at.isoformat()
        }


@dataclass
class Alert:
    severity: RiskLevel
    title: str
    message: str
    node_id: str
    confidence: float = 1.0
    timestamp: datetime = field(
        default_factory=datetime.now
    )
    water_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "nodeId": self.node_id,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "waterLevel": self.water_level
        }