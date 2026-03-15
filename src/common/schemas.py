from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CreditRiskResult:
    credit_probability: float
    credit_score: float
    credit_category: str
    credit_decision: str
    credit_latency_ms: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FraudRiskResult:
    fraud_probability: float | None
    fraud_score: float | None
    fraud_category: str
    fraud_decision: str
    fraud_latency_ms: float | None
    fraud_status: str = "scored"
    fraud_reason: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FinalDecisionResult:
    credit: dict
    fraud: dict
    final_decision: str
    decision_latency_ms: float

    def to_dict(self):
        return asdict(self)
