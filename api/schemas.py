from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=100, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0, examples=[845.50])

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        valid = {"No", "DSL", "Fiber optic"}
        if v not in valid:
            raise ValueError(f"InternetService must be one of {valid}")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment(cls, v):
        valid = {
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        }
        if v not in valid:
            raise ValueError(f"PaymentMethod must be one of {valid}")
        return v


class FactorDetail(BaseModel):
    feature: str
    value: float
    shap_impact: float
    direction: str
    description: str


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_predicted: bool
    threshold_used: float
    explanation_summary: str
    top_factors: list[FactorDetail]


class BatchPredictionItem(BaseModel):
    customer_id: str | None = None
    prediction: PredictionResponse


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool
    metadata: dict[str, Any]
