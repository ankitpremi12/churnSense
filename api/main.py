"""
ChurnSense FastAPI application.

Endpoints:
    POST /predict          — single customer prediction + explanation
    POST /predict/batch    — batch predictions (JSON array or CSV upload)
    GET  /health           — model status
"""

import io
import sys
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse

# Ensure project modules are importable regardless of working directory
_project_root = Path(__file__).parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from churnsense.model import ChurnModel
from api.schemas import (
    CustomerFeatures, GenericPredictionInput, PredictionResponse, BatchPredictionItem,
    FactorDetail, HealthResponse,
)


_model: ChurnModel | None = None
_startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        _model = ChurnModel()
        print(f"Model loaded: {_model.model_name}")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will start but predictions will fail until the model is trained.")
    yield


app = FastAPI(
    title="ChurnSense API",
    description="Customer churn prediction with plain-language explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model() -> ChurnModel:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m churnsense.train` first.",
        )
    return _model


def _customer_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump()])


def _build_prediction_response(result: dict) -> PredictionResponse:
    factors = [FactorDetail(**f) for f in result["top_factors"]]
    return PredictionResponse(
        churn_probability=result["churn_probability"],
        churn_predicted=result["churn_predicted"],
        threshold_used=result["threshold_used"],
        explanation_summary=result["explanation_summary"],
        top_factors=factors,
    )


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    model = _model
    loaded = model is not None
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_name=model.model_name if loaded else "none",
        model_loaded=loaded,
        metadata=model.metadata if loaded else {},
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(
    customer: CustomerFeatures,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Churn probability threshold"),
):
    model = _require_model()
    df = _customer_to_df(customer)
    try:
        results = model.predict_with_explanation(df, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return _build_prediction_response(results[0])


@app.post("/predict/generic", response_model=PredictionResponse, tags=["Predictions"])
def predict_generic(
    input_data: GenericPredictionInput,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Probability threshold"),
):
    model = _require_model()
    df = pd.DataFrame([input_data.features])
    try:
        results = model.predict_with_explanation(df, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generic prediction failed: {e}")
    return _build_prediction_response(results[0])



@app.post("/predict/batch", tags=["Predictions"])
def predict_batch(
    customers: Optional[list[CustomerFeatures]] = None,
    file: Optional[UploadFile] = File(default=None),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    response_format: str = Query(default="json", pattern="^(json|csv)$"),
):
    """
    Accepts either:
    - A JSON array of customer objects in the request body
    - A CSV file upload (multipart/form-data)

    Returns predictions as JSON or CSV based on `response_format`.
    """
    model = _require_model()

    if file is not None:
        content = file.file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
        # TotalCharges might have blank strings
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        customer_ids = df.get("customerID", pd.Series([None] * len(df))).tolist()
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])
    elif customers is not None:
        df = pd.DataFrame([c.model_dump() for c in customers])
        customer_ids = [None] * len(df)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a JSON array in the body or a CSV file upload.",
        )

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Empty input — no customers to predict.")
    if len(df) > 10_000:
        raise HTTPException(status_code=413, detail="Batch too large. Max 10,000 rows.")

    try:
        results = model.predict_with_explanation(df, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    if response_format == "csv":
        rows = []
        for cid, r in zip(customer_ids, results):
            row = {
                "customer_id": cid,
                "churn_probability": r["churn_probability"],
                "churn_predicted": r["churn_predicted"],
                "explanation_summary": r["explanation_summary"],
            }
            for j, factor in enumerate(r["top_factors"], 1):
                row[f"factor_{j}"] = factor["description"]
            rows.append(row)
        out_df = pd.DataFrame(rows)
        csv_buffer = io.StringIO()
        out_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
        )

    # JSON response
    output = []
    for cid, r in zip(customer_ids, results):
        item = BatchPredictionItem(
            customer_id=str(cid) if cid is not None else None,
            prediction=_build_prediction_response(r),
        )
        output.append(item.model_dump())
    return JSONResponse(content=output)
