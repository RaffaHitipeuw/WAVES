from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Any, List, Tuple, FrozenSet
from enum import Enum, auto
import copy


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
    UNKNOWN_STATE = auto()


class RelationType(Enum):
    DERIVED_FROM = auto()
    JUSTIFIES = auto()
    REQUIRES = auto()
    SUPPORTS = auto()
    CONTRADICTS = auto()
    CAUSED_BY = auto()
    REFINES = auto()
    BASED_ON = auto()
    ENTAILED_BY = auto()
    CONFLICTS_WITH = auto()
    REFINED_BY = auto()


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


class EvidenceQuality(Enum):
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    UNUSABLE = auto()
    UNASSESSED = auto()


class InferenceStatus(Enum):
    ALLOWED = auto()
    BLOCKED = auto()
    UNCERTAIN = auto()


@dataclass(frozen=True)
class EntityId:
    namespace: str
    local_id: str

    def __post_init__(self):
        if not self.namespace:
            raise ValueError("EntityId namespace cannot be empty")
        if not self.local_id:
            raise ValueError("EntityId local_id cannot be empty")

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

    def child(self, suffix: str) -> "EntityId":
        return EntityId(self.namespace, f"{self.local_id}/{suffix}")

    def parent(self) -> Optional["EntityId"]:
        parts = self.local_id.rsplit("/", 1)
        if len(parts) == 1:
            return None
        return EntityId(self.namespace, parts[0])


@dataclass(frozen=True)
class ClaimId:
    namespace: str
    local_id: str

    def __post_init__(self):
        if not self.namespace or not self.local_id:
            raise ValueError("ClaimId parts cannot be empty")

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

    def as_entity_id(self) -> EntityId:
        return EntityId(f"claim:{self.namespace}", self.local_id)


@dataclass(frozen=True)
class EvidenceId:
    namespace: str
    local_id: str

    def __post_init__(self):
        if not self.namespace or not self.local_id:
            raise ValueError("EvidenceId parts cannot be empty")

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

    def as_entity_id(self) -> EntityId:
        return EntityId(f"ev:{self.namespace}", self.local_id)


@dataclass(frozen=True)
class AssumptionId:
    namespace: str
    local_id: str

    def __post_init__(self):
        if not self.namespace or not self.local_id:
            raise ValueError("AssumptionId parts cannot be empty")

    def __str__(self) -> str:
        return f"assume:{self.namespace}:{self.local_id}"

    def __repr__(self) -> str:
        return f"AssumptionId({self.namespace!r}, {self.local_id!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, AssumptionId):
            return NotImplemented
        return self.namespace == other.namespace and self.local_id == other.local_id

    def __hash__(self) -> int:
        return hash(("assume", self.namespace, self.local_id))


@dataclass(frozen=True)
class FailureModeId:
    namespace: str
    local_id: str

    def __post_init__(self):
        if not self.namespace or not self.local_id:
            raise ValueError("FailureModeId parts cannot be empty")

    def __str__(self) -> str:
        return f"fail:{self.namespace}:{self.local_id}"

    def __repr__(self) -> str:
        return f"FailureModeId({self.namespace!r}, {self.local_id!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, FailureModeId):
            return NotImplemented
        return self.namespace == other.namespace and self.local_id == other.local_id

    def __hash__(self) -> int:
        return hash(("fail", self.namespace, self.local_id))


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

    def get_status(self, context: str) -> ValidityStatus:
        if context in self.invalid_for:
            return ValidityStatus.INVALID
        if context in self.unknown_for:
            return ValidityStatus.UNKNOWN
        return ValidityStatus.VALID

    def is_context_covered(self, context: str) -> bool:
        return (context in self.valid_for or context in self.unknown_for or context in self.invalid_for)

    def union(self, other: "ValidityScope") -> "ValidityScope":
        return ValidityScope(
            valid_for=self.valid_for | other.valid_for,
            unknown_for=self.unknown_for | other.unknown_for,
            invalid_for=self.invalid_for | other.invalid_for,
        )

    def intersection(self, other: "ValidityScope") -> "ValidityScope":
        return ValidityScope(
            valid_for=self.valid_for & other.valid_for,
            unknown_for=self.unknown_for & other.unknown_for,
            invalid_for=self.invalid_for & other.invalid_for,
        )

    def restrict_to(self, contexts: Set[str]) -> "ValidityScope":
        return ValidityScope(
            valid_for=self.valid_for & contexts,
            unknown_for=self.unknown_for & contexts,
            invalid_for=self.invalid_for & contexts,
        )

    def expand_valid(self, contexts: Set[str]) -> "ValidityScope":
        new_valid = (self.valid_for | contexts) - self.invalid_for - self.unknown_for
        return ValidityScope(valid_for=new_valid, unknown_for=self.unknown_for, invalid_for=self.invalid_for)

    def negate(self) -> "ValidityScope":
        return ValidityScope(
            valid_for=set(),
            unknown_for=self.valid_for,
            invalid_for=self.invalid_for | self.unknown_for,
        )

    def all_contexts(self) -> FrozenSet[str]:
        return frozenset(self.valid_for | self.unknown_for | self.invalid_for)


@dataclass
class ScopedValidity:
    status: ValidityStatus
    scope: ValidityScope
    reason: Optional[str] = None

    @staticmethod
    def valid(scope: Optional[ValidityScope] = None, reason: Optional[str] = None) -> "ScopedValidity":
        return ScopedValidity(status=ValidityStatus.VALID, scope=scope or ValidityScope(), reason=reason)

    @staticmethod
    def invalid(reason: str, scope: Optional[ValidityScope] = None) -> "ScopedValidity":
        return ScopedValidity(status=ValidityStatus.INVALID, scope=scope or ValidityScope(), reason=reason)

    @staticmethod
    def unknown(reason: Optional[str] = None, scope: Optional[ValidityScope] = None) -> "ScopedValidity":
        return ScopedValidity(status=ValidityStatus.UNKNOWN, scope=scope or ValidityScope(), reason=reason)

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

    def get_status(self, context: Optional[str] = None) -> ValidityStatus:
        if self.status != ValidityStatus.VALID:
            return self.status
        if context is not None:
            return self.scope.get_status(context)
        return self.status

    def merge(self, other: "ScopedValidity") -> "ScopedValidity":
        combined_scope = self.scope.union(other.scope)
        if self.status == ValidityStatus.INVALID or other.status == ValidityStatus.INVALID:
            merged_status = ValidityStatus.INVALID
        elif self.status == ValidityStatus.UNKNOWN or other.status == ValidityStatus.UNKNOWN:
            merged_status = ValidityStatus.UNKNOWN
        else:
            merged_status = ValidityStatus.VALID
        return ScopedValidity(status=merged_status, scope=combined_scope, reason=self.reason or other.reason)

    def restrict(self, contexts: Set[str]) -> "ScopedValidity":
        return ScopedValidity(status=self.status, scope=self.scope.restrict_to(contexts), reason=self.reason)

    def is_bounded(self) -> bool:
        return self.scope.is_bounded()

    def requires_context(self, context: str) -> bool:
        return self.scope.is_context_covered(context)

    def to_tuple(self) -> Tuple[ValidityStatus, FrozenSet[str], FrozenSet[str], FrozenSet[str]]:
        return (self.status, frozenset(self.scope.valid_for), frozenset(self.scope.unknown_for), frozenset(self.scope.invalid_for))


@dataclass
class Provenance:
    source: EntityId
    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    upstream: List["Provenance"] = field(default_factory=list)

    def trace(self) -> List["Provenance"]:
        lineage = [self]
        for up in self.upstream:
            lineage.extend(up.trace())
        return lineage

    def depth(self) -> int:
        if not self.upstream:
            return 1
        return 1 + max(up.depth() for up in self.upstream)

    def flatten(self) -> List[EntityId]:
        return [p.source for p in self.trace()]

    def contains_cycle(self) -> bool:
        visited: Set[EntityId] = set()
        stack: List["Provenance"] = [self]
        while stack:
            current = stack.pop()
            if current.source in visited:
                return True
            visited.add(current.source)
            stack.extend(current.upstream)
        return False

    def with_upstream(self, upstream: List["Provenance"]) -> "Provenance":
        return Provenance(source=self.source, method=self.method, parameters=self.parameters.copy(), upstream=self.upstream + upstream)


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

    def satisfy_with(self, entity_id: EntityId) -> None:
        if entity_id not in self.satisfied_by:
            self.satisfied_by.append(entity_id)

    def unsatisfy(self) -> None:
        self.satisfied_by.clear()


@dataclass
class ValidationRecord:
    validator: str
    timestamp: float
    result: ValidityStatus
    checks_performed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def passed(self) -> bool:
        return self.result == ValidityStatus.VALID

    def failed(self) -> bool:
        return self.result == ValidityStatus.INVALID


@dataclass
class Metric:
    value: float
    unit: str
    timestamp: float
    source: EntityId
    provenance: Optional["Provenance"] = None

    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Metric value must be numeric, got {type(self.value)}")
        if not self.unit:
            raise ValueError("Metric unit cannot be empty")

    def add(self, other: "Metric") -> "Metric":
        if self.unit != other.unit:
            raise ValueError(f"Cannot add metrics with different units: {self.unit} vs {other.unit}")
        return Metric(value=self.value + other.value, unit=self.unit, timestamp=max(self.timestamp, other.timestamp), source=self.source, provenance=self.provenance)

    def subtract(self, other: "Metric") -> "Metric":
        if self.unit != other.unit:
            raise ValueError(f"Cannot subtract metrics with different units: {self.unit} vs {other.unit}")
        return Metric(value=self.value - other.value, unit=self.unit, timestamp=max(self.timestamp, other.timestamp), source=self.source, provenance=self.provenance)

    def scale(self, factor: float) -> "Metric":
        return Metric(value=self.value * factor, unit=self.unit, timestamp=self.timestamp, source=self.source, provenance=self.provenance)

    def is_compatible_unit(self, other: "Metric") -> bool:
        return self.unit == other.unit

    def as_claim(self, claim_id: ClaimId) -> "Claim":
        return Claim(id=claim_id, metric=self, justification=f"Derived from {self.source}", evidence=[], assumptions=[], validity=ScopedValidity.valid())


@dataclass
class ProbabilityClaim:
    probability: float
    basis: str
    confidence: float
    timestamp: float
    source: EntityId
    provenance: Optional["Provenance"] = None

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be in [0,1], got {self.probability}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        return self.confidence >= threshold

    def is_high_probability(self, threshold: float = 0.7) -> bool:
        return self.probability >= threshold

    def is_low_probability(self, threshold: float = 0.3) -> bool:
        return self.probability <= threshold

    def as_epistemic_range(self) -> Tuple[float, float]:
        spread = 1.0 - self.confidence
        return (max(0.0, self.probability - spread), min(1.0, self.probability + spread))

    def and_then(self, other: "ProbabilityClaim") -> "ProbabilityClaim":
        return ProbabilityClaim(
            probability=self.probability * other.probability,
            basis=f"{self.basis} AND {other.basis}",
            confidence=min(self.confidence, other.confidence),
            timestamp=max(self.timestamp, other.timestamp),
            source=self.source,
        )

    def or_else(self, other: "ProbabilityClaim") -> "ProbabilityClaim":
        combined = self.probability + other.probability - (self.probability * other.probability)
        return ProbabilityClaim(
            probability=combined,
            basis=f"{self.basis} OR {other.basis}",
            confidence=min(self.confidence, other.confidence),
            timestamp=max(self.timestamp, other.timestamp),
            source=self.source,
        )

    def negate(self) -> "ProbabilityClaim":
        return ProbabilityClaim(
            probability=1.0 - self.probability,
            basis=f"NOT {self.basis}",
            confidence=self.confidence,
            timestamp=self.timestamp,
            source=self.source,
        )


@dataclass
class Entity:
    id: EntityId
    type: EntityType
    validity: ScopedValidity = field(default_factory=ScopedValidity.valid)
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

    def get_status(self, context: Optional[str] = None) -> ValidityStatus:
        return self.validity.get_status(context)

    def get_missing_requirements(self) -> List[Requirement]:
        return [r for r in self.requirements if r.is_blocking() and not r.is_satisfied()]

    def add_requirement(self, requirement: Requirement) -> None:
        self.requirements.append(requirement)

    def add_validation(self, record: ValidationRecord) -> None:
        self.validation_history.append(record)

    def get_latest_validation(self) -> Optional[ValidationRecord]:
        if not self.validation_history:
            return None
        return max(self.validation_history, key=lambda v: v.timestamp)

    def has_blocking_gaps(self) -> bool:
        return any(r.is_blocking() for r in self.requirements)

    def clone(self, new_id: Optional[EntityId] = None) -> "Entity":
        return Entity(
            id=new_id or self.id,
            type=self.type,
            validity=copy.deepcopy(self.validity),
            provenance=copy.deepcopy(self.provenance),
            requirements=copy.deepcopy(self.requirements),
            validation_history=list(self.validation_history),
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "type": self.type.name,
            "validity_status": self.validity.status.name,
            "validity_reason": self.validity.reason,
            "scope_bounded": self.validity.is_bounded(),
            "provenance_method": self.provenance.method if self.provenance else None,
            "requirements": [{"id": r.id, "description": r.description, "satisfied": r.is_satisfied()} for r in self.requirements],
            "validation_count": len(self.validation_history),
            "metadata": self.metadata,
        }


@dataclass
class Evidence(Entity):
    content: Any = None
    raw_signal: Any = field(default=None, repr=False)
    quality_score: float = 0.0
    quality_level: EvidenceQuality = EvidenceQuality.UNASSESSED
    detection_method: Optional[str] = None

    def __post_init__(self):
        if self.type != EntityType.EVIDENCE:
            object.__setattr__(self, "type", EntityType.EVIDENCE)

    def quality_at_least(self, threshold: EvidenceQuality) -> bool:
        quality_order = [EvidenceQuality.UNASSESSED, EvidenceQuality.UNUSABLE, EvidenceQuality.LOW, EvidenceQuality.MEDIUM, EvidenceQuality.HIGH]
        try:
            return quality_order.index(self.quality_level) >= quality_order.index(threshold)
        except ValueError:
            return False

    def is_usable(self) -> bool:
        return self.quality_at_least(EvidenceQuality.LOW)

    def is_high_quality(self) -> bool:
        return self.quality_level == EvidenceQuality.HIGH

    def assess_quality(self, score: float) -> EvidenceQuality:
        if score >= 0.8:
            assessed = EvidenceQuality.HIGH
        elif score >= 0.6:
            assessed = EvidenceQuality.MEDIUM
        elif score >= 0.3:
            assessed = EvidenceQuality.LOW
        else:
            assessed = EvidenceQuality.UNUSABLE
        object.__setattr__(self, "quality_level", assessed)
        object.__setattr__(self, "quality_score", score)
        return assessed


@dataclass
class Claim(Entity):
    metric: Optional[Metric] = None
    probability: Optional[ProbabilityClaim] = None
    justification: str = ""
    evidence: List[EvidenceId] = field(default_factory=list)
    assumptions: List[AssumptionId] = field(default_factory=list)

    def __post_init__(self):
        if self.type != EntityType.CLAIM:
            object.__setattr__(self, "type", EntityType.CLAIM)

    def has_metric(self) -> bool:
        return self.metric is not None

    def has_probability(self) -> bool:
        return self.probability is not None

    def has_justification(self) -> bool:
        return bool(self.justification)

    def is_well_formed(self) -> bool:
        return (self.has_metric() or self.has_probability() or self.has_justification()) and self.is_valid()

    def is_evidence_backed(self) -> bool:
        return len(self.evidence) > 0

    def is_grounded(self) -> bool:
        return self.is_evidence_backed() and self.is_valid()

    def get_value(self) -> Optional[float]:
        if self.metric:
            return self.metric.value
        return None

    def get_unit(self) -> Optional[str]:
        if self.metric:
            return self.metric.unit
        return None

    def merge_with(self, other: "Claim") -> "Claim":
        merged_metric = None
        if self.metric and other.metric and self.metric.is_compatible_unit(other.metric):
            merged_metric = self.metric.add(other.metric).scale(0.5)
        elif self.metric:
            merged_metric = self.metric
        elif other.metric:
            merged_metric = other.metric

        merged_prob = None
        if self.probability and other.probability:
            merged_prob = self.probability.and_then(other.probability)
        elif self.probability:
            merged_prob = self.probability
        elif other.probability:
            merged_prob = other.probability

        merged_validity = self.validity.merge(other.validity)
        merged_evidence = list(set(self.evidence + other.evidence))
        merged_assumptions = list(set(self.assumptions + other.assumptions))

        return Claim(
            id=self.id, type=EntityType.CLAIM, validity=merged_validity,
            metric=merged_metric, probability=merged_prob,
            justification=self.justification or other.justification,
            evidence=merged_evidence, assumptions=merged_assumptions,
        )


@dataclass
class Assumption(Entity):
    statement: str = ""
    assumption_id: Optional[AssumptionId] = None
    held: bool = True
    depends_on: List[AssumptionId] = field(default_factory=list)

    def __post_init__(self):
        if self.type != EntityType.ASSUMPTION:
            object.__setattr__(self, "type", EntityType.ASSUMPTION)

    def is_held(self, context: Optional[str] = None) -> bool:
        return self.held and self.is_valid(context)

    def is_released(self) -> bool:
        return not self.held

    def release(self) -> None:
        object.__setattr__(self, "held", False)

    def reinstate(self) -> None:
        object.__setattr__(self, "held", True)


@dataclass
class FailureMode(Entity):
    failure_id: Optional[FailureModeId] = None
    description: str = ""
    severity: str = "UNKNOWN"
    detected_at: Optional[float] = None
    resolved_at: Optional[float] = None
    affected_entities: List[EntityId] = field(default_factory=list)

    def __post_init__(self):
        if self.type != EntityType.CLAIM:
            object.__setattr__(self, "type", EntityType.CLAIM)

    def is_active(self) -> bool:
        return self.resolved_at is None

    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    def resolve(self, at_timestamp: float) -> None:
        object.__setattr__(self, "resolved_at", at_timestamp)

    def is_severe(self) -> bool:
        return self.severity in {"CRITICAL", "HIGH", "ERROR"}

    def affects(self, entity_id: EntityId) -> bool:
        return entity_id in self.affected_entities


@dataclass
class GranularUnknown:
    category: UnknownCategory
    reason: str
    permitted_inferences: Set[str] = field(default_factory=set)
    forbidden_inferences: Set[str] = field(default_factory=set)
    resolution_requirements: List[Requirement] = field(default_factory=list)
    provenance: Optional[Provenance] = None
    detected_at: Optional[float] = None
    propagated: bool = False

    def can_infer(self, inference_type: str) -> bool:
        if inference_type in self.forbidden_inferences:
            return False
        if self.permitted_inferences:
            return inference_type in self.permitted_inferences
        return True

    def get_inference_status(self, inference_type: str) -> InferenceStatus:
        if inference_type in self.forbidden_inferences:
            return InferenceStatus.BLOCKED
        if self.permitted_inferences and inference_type not in self.permitted_inferences:
            return InferenceStatus.UNCERTAIN
        return InferenceStatus.ALLOWED

    def get_resolution_path(self) -> List[str]:
        return [req.id for req in self.resolution_requirements if not req.is_satisfied()]

    def get_unmet_requirements(self) -> List[Requirement]:
        return [req for req in self.resolution_requirements if req.is_blocking()]

    def is_resolvable(self) -> bool:
        return len(self.get_resolution_path()) > 0

    def is_blocking(self) -> bool:
        return any(r.is_blocking() for r in self.resolution_requirements)

    def blocks_water_level(self) -> bool:
        return "water_level" in self.forbidden_inferences

    def blocks_risk_assessment(self) -> bool:
        return "risk_level" in self.forbidden_inferences

    def with_propagation(self, propagate: bool) -> "GranularUnknown":
        return GranularUnknown(
            category=self.category, reason=self.reason,
            permitted_inferences=self.permitted_inferences.copy(),
            forbidden_inferences=self.forbidden_inferences.copy(),
            resolution_requirements=self.resolution_requirements.copy(),
            provenance=self.provenance, detected_at=self.detected_at, propagated=propagate,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.name, "reason": self.reason,
            "permitted": list(self.permitted_inferences), "forbidden": list(self.forbidden_inferences),
            "resolution_requirements": [r.id for r in self.resolution_requirements],
            "resolvable": self.is_resolvable(), "blocking": self.is_blocking(),
        }

    @staticmethod
    def out_of_scope(reason: str) -> "GranularUnknown":
        return GranularUnknown(
            category=UnknownCategory.OUT_OF_SCOPE, reason=reason,
            permitted_inferences=set(), forbidden_inferences={"water_level", "risk_level", "decision"},
        )

    @staticmethod
    def geometry_violation(reason: str) -> "GranularUnknown":
        return GranularUnknown(
            category=UnknownCategory.GEOMETRY_VIOLATION, reason=reason,
            permitted_inferences={"calibration_needed"}, forbidden_inferences={"water_level", "risk_level"},
            resolution_requirements=[Requirement(id="geometry_calibration", description="Calibrate scene geometry", blocking=True)],
        )

    @staticmethod
    def insufficient_data(reason: str) -> "GranularUnknown":
        return GranularUnknown(
            category=UnknownCategory.INSUFFICIENT_DATA, reason=reason,
            permitted_inferences={"buffer_accumulation"}, forbidden_inferences={"water_level", "risk_level"},
            resolution_requirements=[Requirement(id="more_readings", description="Accumulate temporal buffer", blocking=True)],
        )

    @staticmethod
    def conflicting_evidence(reason: str) -> "GranularUnknown":
        return GranularUnknown(
            category=UnknownCategory.CONFLICTING_EVIDENCE, reason=reason,
            permitted_inferences={"uncertainty_propagation"}, forbidden_inferences={"water_level", "risk_level"},
            resolution_requirements=[Requirement(id="resolve_conflict", description="Reconcile conflicting evidence", blocking=True)],
        )

    @staticmethod
    def unidentifiable(reason: str) -> "GranularUnknown":
        return GranularUnknown(
            category=UnknownCategory.UNIDENTIFIABLE, reason=reason,
            permitted_inferences=set(), forbidden_inferences={"water_level", "risk_level", "decision"},
        )

    @staticmethod
    def model_limitation(reason: str) -> "GranularUnknown":
        return GranularUnknown(
            category=UnknownCategory.MODEL_LIMITATION, reason=reason,
            permitted_inferences={"model_bounds"}, forbidden_inferences={"water_level"},
            resolution_requirements=[Requirement(id="model_improvement", description="Improve model capability", blocking=False)],
        )

    @staticmethod
    def from_category(category: UnknownCategory, reason: str) -> "GranularUnknown":
        factory_map = {
            UnknownCategory.OUT_OF_SCOPE: GranularUnknown.out_of_scope,
            UnknownCategory.GEOMETRY_VIOLATION: GranularUnknown.geometry_violation,
            UnknownCategory.INSUFFICIENT_DATA: GranularUnknown.insufficient_data,
            UnknownCategory.CONFLICTING_EVIDENCE: GranularUnknown.conflicting_evidence,
            UnknownCategory.UNIDENTIFIABLE: GranularUnknown.unidentifiable,
            UnknownCategory.MODEL_LIMITATION: GranularUnknown.model_limitation,
        }
        factory = factory_map.get(category)
        if factory:
            return factory(reason)
        return GranularUnknown(
            category=category, reason=reason,
            permitted_inferences=set(), forbidden_inferences={"water_level", "risk_level", "decision"},
        )
