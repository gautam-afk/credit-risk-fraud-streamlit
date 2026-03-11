from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CreditRiskResult:
    probability_default: float
    risk_score: float
    risk_category: str
    decision: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FraudRiskResult:
    probability_fraud: float
    fraud_score: float
    fraud_category: str
    fraud_decision: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FinalDecisionResult:
    credit: dict
    fraud: dict
    final_decision: str

    def to_dict(self):
        return asdict(self)

