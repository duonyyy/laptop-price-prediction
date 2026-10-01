import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    LaptopSpecsInput,
    PricePredictionResponse,
    SpecsOptionsResponse,
)
from app.service import get_preset_cases, get_specs_options, predict_price

# Khởi tạo ứng dụng FastAPI với tài liệu Swagger UI tự động
app = FastAPI(
    title="Laptop Price Prediction API",
    description=(
        "REST API phục vụ dự báo giá Laptop dựa trên các đặc trưng phần cứng "
        "bằng mô hình Machine Learning tối ưu (XGBoost Pipeline)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cấu hình CORS để cho phép Streamlit hoặc ứng dụng Web gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    summary="Trạng thái hệ thống & Giới thiệu API",
    tags=["General"],
)
def root():
    return {
        "project": "Laptop Price Prediction Machine Learning API",
        "status": "online",
        "model_architecture": "Scikit-Learn Pipeline (ColumnTransformer + OneHotEncoder + XGBRegressor)",
        "docs_url": "/docs",
        "endpoints": {
            "specs_options": "/specs-options",
            "predict": "/predict",
            "presets": "/presets",
            "health": "/health",
        },
    }


@app.get(
    "/health",
    summary="Kiểm tra tình trạng hoạt động (Health Check)",
    tags=["General"],
)
def health_check():
    return {"status": "healthy", "service": "laptop-price-prediction-api"}


@app.get(
    "/specs-options",
    response_model=SpecsOptionsResponse,
    summary="Lấy danh mục cấu hình hợp lệ",
    tags=["Configuration"],
)
def specs_options():
    try:
        return get_specs_options()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi truy xuất danh mục cấu hình: {str(exc)}",
        )


@app.get(
    "/presets",
    summary="Lấy danh sách các trường hợp cấu hình mẫu",
    tags=["Presets"],
)
def presets():
    try:
        return get_preset_cases()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi truy xuất cấu hình mẫu: {str(exc)}",
        )


@app.post(
    "/predict",
    response_model=PricePredictionResponse,
    summary="Dự báo giá laptop từ thông số cấu hình",
    tags=["Prediction"],
)
def predict(specs: LaptopSpecsInput):
    try:
        # Chuyển đổi model pydantic sang dictionary (bao gồm cả các trường alias)
        data_dict = specs.model_dump(by_alias=True)
        result = predict_price(data_dict)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể xử lý yêu cầu dự đoán: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
