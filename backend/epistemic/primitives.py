from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Any, List
from enum import Enum, auto


class EntityType(Enum):
    MEASUREMENT = auto()
    CLAIM = auto()
    EVIDENCE = auto()
    ASSUMPTION = auto()
    INFERENCE = auto()
    DECISION = auto()
    ACTION = auto()
    NODE = auto()
    CALIBRATION = auto()
    WATERLINE = auto()


class RelationType(Enum):
    DERIVED_FROM = auto()
    JUSTIFIES = auto()
    REQUIRES = auto()
    SUPPORTS = auto()
    CONTRADICTS = auto()
    CAUSED_BY = auto()
    REFINES = auto()
    BASED_ON = auto()


class UnknownCategory(Enum):
    OUT_OF_SCOPE = auto()
    GEOMETRY_VIOLATION = auto()
    INSUFFICIENT_DATA = auto()
    CONFLICTING_EVIDENCE = auto()
    UNIDENTIFIABLE = auto()
    MODEL_LIMITATION = auto()


class ValidityStatus(Enum):
    VALID = auto()
    INVALID = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class EntityId:
    namespace: str
    local_id: str

    def __str__(self) -> str:
        return f"{self.namespace}:{self.local_id}"

    def __repr__(self) -> str:
        return f"EntityId({self.namespace!r}, {self.local_id!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EntityId):
            return NotImplemented
        return self.namespace == other.namespace and self.local_id == other.local_id

    def __hash__(self) -> int:
        return hash((self.namespace, self.local_id))


@dataclass(frozen=True)
class ClaimId:
    namespace: str
    local_id: str

    def __str__(self) -> str:
        return f"claim:{self.namespace}:{self.local_id}"

    def __repr__(self) -> str:
        return f"ClaimId({self.namespace!r}, {self.local_id!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ClaimId):
            return NotImplemented
        return self.namespace == other.namespace and self.local_id == other.local_id

    def __hash__(self) -> int:
        return hash(("claim", self.namespace, self.local_id))


@dataclass(frozen=True)
class EvidenceId:
    namespace: str
    local_id: str

    def __str__(self) -> str:
        return f"ev:{self.namespace}:{self.local_id}"

    def __repr__(self) -> str:
        return f"EvidenceId({self.namespace!r}, {self.local_id!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EvidenceId):
            return NotImplemented
        return self.namespace == other.namespace and self.local_id == other.local_id

    def __hash__(self) -> int:
        return hash(("ev", self.namespace, self.local_id))


@dataclass
class ValidityScope:
    valid_for: Set[str] = field(default_factory=set)
    unknown_for: Set[str] = field(default_factory=set)
    invalid_for: Set[str] = field(default_factory=set)

    def is_valid_for(self, context: str) -> bool:
        if context in self.invalid_for:
            return False
        if context in self.unknown_for:
            return False
        return True

    def is_invalid_for(self, context: str) -> bool:
        return context in self.invalid_for

    def is_unknown_for(self, context: str) -> bool:
        return context in self.unknown_for

    def is_bounded(self) -> bool:
        return bool(self.valid_for or self.unknown_for or self.invalid_for)


@dataclass
class ScopedValidity:
    status: ValidityStatus
    scope: ValidityScope
    reason: Optional[str] = None

    @staticmethod
    def valid(scope: Optional[ValidityScope] = None, reason: Optional[str] = None) -> 'ScopedValidity':
        return ScopedValidity(
            status=ValidityStatus.VALID,
            scope=scope or ValidityScope(),
            reason=reason
        )

    @staticmethod
    def invalid(reason: str, scope: Optional[ValidityScope] = None) -> 'ScopedValidity':
        return ScopedValidity(
            status=ValidityStatus.INVALID,
            scope=scope or ValidityScope(),
            reason=reason
        )

    @staticmethod
    def unknown(reason: Optional[str] = None, scope: Optional[ValidityScope] = None) -> 'ScopedValidity':
        return ScopedValidity(
            status=ValidityStatus.UNKNOWN,
            scope=scope or ValidityScope(),
            reason=reason
        )

    def is_valid(self, context: Optional[str] = None) -> bool:
        if self.status != ValidityStatus.VALID:
            return False
        if context is None:
            return True
        return self.scope.is_valid_for(context)

    def is_invalid(self, context: Optional[str] = None) -> bool:
        if self.status == ValidityStatus.INVALID:
            return True
        if context is not None:
            return self.scope.is_invalid_for(context)
        return False

    def is_unknown(self, context: Optional[str] = None) -> bool:
        if self.status == ValidityStatus.UNKNOWN:
            return True
        if context is not None:
            return self.scope.is_unknown_for(context)
        return False


@dataclass
class Metric:
    value: float
    unit: str
    timestamp: float
    source: EntityId
    provenance: Optional['Provenance'] = None

    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Metric value must be numeric, got {type(self.value)}")
        if not self.unit:
            raise ValueError("Metric unit cannot be empty")

    def as_claim(self, claim_id: ClaimId) -> 'Claim':
        return Claim(
            id=claim_id,
            metric=self,
            justification=f"Derived from {self.source}",
            evidence=[],
            assumptions=[],
            validity=ScopedValidity.valid()
        )


@dataclass
class ProbabilityClaim:
    probability: float
    basis: str
    confidence: float
    timestamp: float
    source: EntityId
    provenance: Optional['Provenance'] = None

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be in [0,1], got {self.probability}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        return self.confidence >= threshold

    def as_epistemic_range(self) -> tuple:
        spread = 1.0 - self.confidence
        return (
            max(0.0, self.probability - spread),
            min(1.0, self.probability + spread)
        )


@dataclass
class Provenance:
    source: EntityId
    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    upstream: List['Provenance'] = field(default_factory=list)

    def trace(self) -> List['Provenance']:
        lineage = [self]
        for up in self.upstream:
            lineage.extend(up.trace())
        return lineage


@dataclass
class Requirement:
    id: str
    description: str
    is_optional: bool = False
    blocking: bool = True
    satisfied_by: List[EntityId] = field(default_factory=list)

    def is_satisfied(self) -> bool:
        return len(self.satisfied_by) > 0

    def is_blocking(self) -> bool:
        return self.blocking and not self.is_satisfied()


@dataclass
class ValidationRecord:
    validator: str
    timestamp: float
    result: ValidityStatus
    checks_performed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    id: EntityId
    type: EntityType
    validity: ScopedValidity = field(default_factory=ScopedValidity.valid())
    provenance: Optional[Provenance] = None
    requirements: List[Requirement] = field(default_factory=list)
    validation_history: List[ValidationRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self, context: Optional[str] = None) -> bool:
        return self.validity.is_valid(context)

    def is_invalid(self, context: Optional[str] = None) -> bool:
        return self.validity.is_invalid(context)

    def is_unknown(self, context: Optional[str] = None) -> bool:
        return self.validity.is_unknown(context)

    def get_missing_requirements(self) -> List[Requirement]:
        return [r for r in self.requirements if r.is_blocking() and not r.is_satisfied()]

    def add_requirement(self, requirement: Requirement) -> None:
        self.requirements.append(requirement)

    def add_validation(self, record: ValidationRecord) -> None:
        self.validation_history.append(record)


@dataclass
class Evidence(Entity):
    content: Any = None
    raw_signal: Any = field(default=None, repr=False)
    quality_score: float = 0.0
    detection_method: Optional[str] = None

    def __post_init__(self):
        if self.type != EntityType.EVIDENCE:
            raise TypeError("Evidence entity must have type EntityType.EVIDENCE")


@dataclass
class Claim(Entity):
    metric: Optional[Metric] = None
    probability: Optional[ProbabilityClaim] = None
    justification: str = ""
    evidence: List[EvidenceId] = field(default_factory=list)
    assumptions: List['AssumptionId'] = field(default_factory=list)

    def __post_init__(self):
        if self.type != EntityType.CLAIM:
            raise TypeError("Claim entity must have type EntityType.CLAIM")

    def has_metric(self) -> bool:
        return self.metric is not None

    def has_probability(self) -> bool:
        return self.probability is not None

    def has_justification(self) -> bool:
        return bool(self.justification)

    def is_well_formed(self) -> bool:
        return (
            self.has_metric() or self.has_probability() or self.has_justification()
        ) and self.is_valid()


@dataclass(frozen=True)
class AssumptionId:
    namespace: str
    local_id: str

    def __str__(self) -> str:
        return f"assume:{self.namespace}:{self.local_id}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, AssumptionId):
            return NotImplemented
        return self.namespace == other.namespace and self.local_id == other.local_id

    def __hash__(self) -> int:
        return hash(("assume", self.namespace, self.local_id))


@dataclass
class Assumption(Entity):
    statement: str = ""
    assumption_id: Optional[AssumptionId] = None
    is_held: bool = True

    def __post_init__(self):
        if self.type != EntityType.ASSUMPTION:
            raise TypeError("Assumption entity must have type EntityType.ASSUMPTION")

    def is_held(self, context: Optional[str] = None) -> bool:
        return self.is_held and self.is_valid(context)


@dataclass(frozen=True)
class FailureModeId:
    namespace: str
    local_id: str

    def __str__(self) -> str:
        return f"fail:{self.namespace}:{self.local_id}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, FailureModeId):
            return NotImplemented
        return self.namespace == other.namespace and self.local_id == other.local_id

    def __hash__(self) -> int:
        return hash(("fail", self.namespace, self.local_id))


@dataclass
class FailureMode(Entity):
    failure_id: Optional[FailureModeId] = None
    description: str = ""
    severity: str = "UNKNOWN"
    detected_at: Optional[float] = None
    resolved_at: Optional[float] = None

    def __post_init__(self):
        if self.type != EntityType.CLAIM:
            object.__setattr__(self, 'type', EntityType.CLAIM)

    def is_active(self) -> bool:
        return self.resolved_at is None

    def is_resolved(self) -> bool:
        return self.resolved_at is not None


@dataclass
class GranularUnknown:
    category: UnknownCategory
    reason: str
    permitted_inferences: Set[str] = field(default_factory=set)
    forbidden_inferences: Set[str] = field(default_factory=set)
    resolution_requirements: List[Requirement] = field(default_factory=list)
    provenance: Optional[Provenance] = None

    def can_infer(self, inference_type: str) -> bool:
        if inference_type in self.forbidden_inferences:
            return False
        if self.permitted_inferences:
            return inference_type in self.permitted_inferences
        return True

    def get_resolution_path(self) -> List[str]:
        return [req.id for req in self.resolution_requirements if not req.is_satisfied()]

    def is_resolvable(self) -> bool:
        return len(self.get_resolution_path()) > 0

    @staticmethod
    def out_of_scope(reason: str) -> 'GranularUnknown':
        return GranularUnknown(
            category=UnknownCategory.OUT_OF_SCOPE,
            reason=reason,
            permitted_inferences=set(),
            forbidden_inferences={'water_level', 'risk_level', 'decision'},
        )

    @staticmethod
    def geometry_violation(reason: str) -> 'GranularUnknown':
        return GranularUnknown(
            category=UnknownCategory.GEOMETRY_VIOLATION,
            reason=reason,
            permitted_inferences={'calibration_needed'},
            forbidden_inferences={'water_level', 'risk_level'},
            resolution_requirements=[
                Requirement(id='geometry_calibration', description='Calibrate scene geometry', blocking=True),
            ]
        )

    @staticmethod
    def insufficient_data(reason: str) -> 'GranularUnknown':
        return GranularUnknown(
            category=UnknownCategory.INSUFFICIENT_DATA,
            reason=reason,
            permitted_inferences={'buffer_accumulation'},
            forbidden_inferences={'water_level', 'risk_level'},
            resolution_requirements=[
                Requirement(id='more_readings', description='Accumulate temporal buffer', blocking=True),
            ]
        )

    @staticmethod
    def conflicting_evidence(reason: str) -> 'GranularUnknown':
        return GranularUnknown(
            category=UnknownCategory.CONFLICTING_EVIDENCE,
            reason=reason,
            permitted_inferences={'uncertainty_propagation'},
            forbidden_inferences={'water_level', 'risk_level'},
            resolution_requirements=[
                Requirement(id='resolve_conflict', description='Reconcile conflicting evidence', blocking=True),
            ]
        )

    @staticmethod
    def unidentifiable(reason: str) -> 'GranularUnknown':
        return GranularUnknown(
            category=UnknownCategory.UNIDENTIFIABLE,
            reason=reason,
            permitted_inferences=set(),
            forbidden_inferences={'water_level', 'risk_level', 'decision'},
        )

    @staticmethod
    def model_limitation(reason: str) -> 'GranularUnknown':
        return GranularUnknown(
            category=UnknownCategory.MODEL_LIMITATION,
            reason=reason,
            permitted_inferences={'model_bounds'},
            forbidden_inferences={'water_level'},
            resolution_requirements=[
                Requirement(id='model_improvement', description='Improve model capability', blocking=False),
            ]
        )
