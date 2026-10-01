from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class LaptopSpecsInput(BaseModel):
    """
    Schema dữ liệu cấu hình laptop đầu vào phục vụ dự báo giá.
    Hỗ trợ cả tên trường theo chuẩn Python lẫn tên trường gốc có chứa ký tự đặc biệt.
    """
    Company: str = Field(default="Dell", description="Hãng sản xuất (VD: Dell, Apple, Asus, HP, Lenovo...)")
    TypeName: str = Field(default="Notebook", description="Phân khúc loại máy (Notebook, Ultrabook, Gaming, Workstation...)")
    Inches: float = Field(default=15.6, ge=10.0, le=20.0, description="Kích thước màn hình (inch)")
    Resolution_X: int = Field(default=1920, ge=800, le=4000, description="Độ phân giải ngang (pixels)")
    Resolution_Y: int = Field(default=1080, ge=600, le=3000, description="Độ phân giải dọc (pixels)")
    CPU_Frequency_GHz: float = Field(default=2.5, ge=0.5, le=5.0, alias="CPU_Frequency (GHz)", description="Tần số CPU (GHz)")
    CPU_Tier: str = Field(default="Intel Core i5", description="Phân cấp CPU (Intel Core i3, i5, i7, AMD Processor...)")
    RAM_GB: int = Field(default=8, ge=2, le=64, alias="RAM (GB)", description="Dung lượng RAM (GB)")
    Weight_kg: float = Field(default=2.0, ge=0.5, le=10.0, alias="Weight (kg)", description="Trọng lượng máy (kg)")
    Touchscreen: int = Field(default=0, ge=0, le=1, description="Màn hình cảm ứng (1: Có, 0: Không)")
    IPS_Panel: int = Field(default=0, ge=0, le=1, description="Tấm nền IPS (1: Có, 0: Không)")
    SSD_GB: float = Field(default=256.0, ge=0.0, le=4096.0, description="Dung lượng SSD (GB)")
    HDD_GB: float = Field(default=0.0, ge=0.0, le=4096.0, description="Dung lượng HDD (GB)")
    Flash_Storage_GB: float = Field(default=0.0, ge=0.0, le=512.0, description="Bộ nhớ Flash (GB)")
    Hybrid_GB: float = Field(default=0.0, ge=0.0, le=1024.0, description="Ổ lai Hybrid (GB)")
    GPU_Brand_Cat: str = Field(default="Intel (Integrated)", description="Thương hiệu GPU (Intel (Integrated), Nvidia, AMD (Dedicated)...)")
    GPU_Dedicated: int = Field(default=0, ge=0, le=1, description="GPU rời (1: Có, 0: Tích hợp)")
    OpSys_Group: str = Field(default="Windows", description="Hệ điều hành (Windows, macOS, Linux, Chrome OS, No OS)")
    PPI: Optional[float] = Field(default=None, description="Mật độ điểm ảnh (PPI). Nếu bỏ trống sẽ tự tính từ Resolution và Inches")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Company": "Apple",
                "TypeName": "Ultrabook",
                "Inches": 13.3,
                "Resolution_X": 2560,
                "Resolution_Y": 1600,
                "CPU_Frequency (GHz)": 2.3,
                "CPU_Tier": "Intel Core i5",
                "RAM (GB)": 8,
                "Weight (kg)": 1.37,
                "Touchscreen": 0,
                "IPS_Panel": 1,
                "SSD_GB": 128.0,
                "HDD_GB": 0.0,
                "Flash_Storage_GB": 0.0,
                "Hybrid_GB": 0.0,
                "GPU_Brand_Cat": "Intel (Integrated)",
                "GPU_Dedicated": 0,
                "OpSys_Group": "macOS"
            }
        }
    )


class PricePredictionResponse(BaseModel):
    """
    Schema kết quả trả về từ mô hình định giá laptop.
    """
    success: bool = True
    predicted_price_euro: float = Field(..., description="Giá dự đoán theo đơn vị Euro (€)")
    predicted_price_vnd: str = Field(..., description="Giá dự đoán quy đổi sang VNĐ (chuỗi có định dạng)")
    predicted_price_vnd_num: float = Field(..., description="Giá trị số nguyên của VNĐ")
    predicted_price_usd: str = Field(..., description="Giá dự đoán quy đổi sang USD")
    price_tier: str = Field(..., description="Phân khúc giá thị trường")
    specs_summary: str = Field(..., description="Tóm tắt thông số cấu hình chính")
    calculated_ppi: float = Field(..., description="Chỉ số PPI màn hình")
    total_storage_gb: float = Field(..., description="Tổng dung lượng lưu trữ (GB)")
    has_ssd: int = Field(..., description="Trạng thái có ổ SSD (1: Có, 0: Không)")


class ResolutionOption(BaseModel):
    label: str
    width: int
    height: int


class SpecsOptionsResponse(BaseModel):
    """
    Danh sách các tùy chọn danh mục cấu hình hợp lệ để xây dựng Form giao diện.
    """
    companies: List[str]
    type_names: List[str]
    cpu_tiers: List[str]
    gpu_brands: List[str]
    opsys_groups: List[str]
    common_resolutions: List[ResolutionOption]
    ram_options: List[int]
    ssd_options: List[float]
    hdd_options: List[float]
