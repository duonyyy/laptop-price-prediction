import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

# Đường dẫn đến tệp mô hình đã huấn luyện
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_laptop_price_model.joblib"
DATA_PATH = PROJECT_ROOT / "datas" / "laptop_price_cleaned.csv"

# Danh sách 19 đặc trưng theo đúng thứ tự mà Pipeline mong đợi
FEATURE_COLUMNS = [
    "Inches",
    "CPU_Frequency (GHz)",
    "RAM (GB)",
    "Weight (kg)",
    "Touchscreen",
    "IPS_Panel",
    "PPI",
    "SSD_GB",
    "HDD_GB",
    "Flash_Storage_GB",
    "Hybrid_GB",
    "Total_Storage_GB",
    "Has_SSD",
    "GPU_Dedicated",
    "Company",
    "TypeName",
    "CPU_Tier",
    "GPU_Brand_Cat",
    "OpSys_Group",
]

# Tỷ giá hối đoái tham chiếu
EUR_TO_VND_RATE = 27000.0
EUR_TO_USD_RATE = 1.09

_MODEL_INSTANCE = None


def get_model():
    """
    Tải mô hình tối ưu theo mẫu Singleton để tối ưu hóa bộ nhớ và tốc độ suy luận.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {MODEL_PATH}")
        _MODEL_INSTANCE = joblib.load(MODEL_PATH)
    return _MODEL_INSTANCE


def calculate_ppi(res_x: int, res_y: int, inches: float) -> float:
    """
    Tính mật độ điểm ảnh Pixels Per Inch (PPI).
    """
    if inches <= 0:
        return 0.0
    return round(float(np.sqrt(res_x**2 + res_y**2) / inches), 2)


def determine_price_tier(price_vnd: float) -> str:
    """
    Phân loại phân khúc giá theo tiền Việt.
    """
    if price_vnd < 15_000_000:
        return "Phổ thông (Dưới 15 triệu VNĐ)"
    elif price_vnd < 25_000_000:
        return "Tầm trung (15 - 25 triệu VNĐ)"
    elif price_vnd < 40_000_000:
        return "Cận cao cấp (25 - 40 triệu VNĐ)"
    elif price_vnd < 60_000_000:
        return "Cao cấp (40 - 60 triệu VNĐ)"
    else:
        return "Flagship / Siêu cao cấp (Trên 60 triệu VNĐ)"


def predict_price(specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Thực hiện dự báo giá laptop dựa trên cấu hình phần cứng.
    """
    model = get_model()
    data = specs.copy()

    # 1. Trích xuất hoặc tính toán PPI
    inches = float(data.get("Inches", 15.6))
    if data.get("PPI") is not None and float(data.get("PPI")) > 0:
        ppi = round(float(data["PPI"]), 2)
    else:
        res_x = int(data.get("Resolution_X", 1920))
        res_y = int(data.get("Resolution_Y", 1080))
        ppi = calculate_ppi(res_x, res_y, inches)
    data["PPI"] = ppi

    # 2. Xử lý trường CPU_Frequency (GHz)
    if "CPU_Frequency (GHz)" not in data:
        data["CPU_Frequency (GHz)"] = float(data.get("CPU_Frequency_GHz", 2.5))

    # 3. Xử lý trường RAM (GB)
    if "RAM (GB)" not in data:
        data["RAM (GB)"] = int(data.get("RAM_GB", 8))

    # 4. Xử lý trường Weight (kg)
    if "Weight (kg)" not in data:
        data["Weight (kg)"] = float(data.get("Weight_kg", 2.0))

    # 5. Xử lý lưu trữ
    ssd = float(data.get("SSD_GB", 0.0))
    hdd = float(data.get("HDD_GB", 0.0))
    flash = float(data.get("Flash_Storage_GB", 0.0))
    hybrid = float(data.get("Hybrid_GB", 0.0))
    total_storage = ssd + hdd + flash + hybrid
    data["Total_Storage_GB"] = total_storage
    data["Has_SSD"] = 1 if ssd > 0 else 0

    # 6. Đảm bảo các thuộc tính mặc định
    data["Inches"] = inches
    data["SSD_GB"] = ssd
    data["HDD_GB"] = hdd
    data["Flash_Storage_GB"] = flash
    data["Hybrid_GB"] = hybrid
    data["Touchscreen"] = int(data.get("Touchscreen", 0))
    data["IPS_Panel"] = int(data.get("IPS_Panel", 0))
    data["GPU_Dedicated"] = int(data.get("GPU_Dedicated", 0))

    # 7. Chuẩn bị DataFrame đúng 19 đặc trưng
    row_dict = {col: data.get(col) for col in FEATURE_COLUMNS}
    df_input = pd.DataFrame([row_dict])[FEATURE_COLUMNS]

    # 8. Dự báo (Không gian Log -> Giải mã hàm nghịch đảo np.expm1)
    pred_log = model.predict(df_input)[0]
    pred_euro = float(np.expm1(pred_log))
    pred_vnd_num = round(pred_euro * EUR_TO_VND_RATE)
    pred_usd_num = round(pred_euro * EUR_TO_USD_RATE, 2)

    tier = determine_price_tier(pred_vnd_num)
    summary = (
        f"{data.get('Company')} {data.get('TypeName')} | "
        f"{data.get('CPU_Tier')} | "
        f"{data.get('RAM (GB)')}GB RAM | "
        f"{int(ssd)}GB SSD"
    )

    return {
        "success": True,
        "predicted_price_euro": round(pred_euro, 2),
        "predicted_price_vnd": f"{pred_vnd_num:,.0f} VNĐ",
        "predicted_price_vnd_num": pred_vnd_num,
        "predicted_price_usd": f"${pred_usd_num:,.2f}",
        "price_tier": tier,
        "specs_summary": summary,
        "calculated_ppi": ppi,
        "total_storage_gb": total_storage,
        "has_ssd": data["Has_SSD"],
    }


def get_specs_options() -> Dict[str, Any]:
    """
    Cung cấp danh mục các tùy chọn cấu hình được dùng trên giao diện.
    """
    companies = [
        "Acer", "Apple", "Asus", "Chuwi", "Dell", "Fujitsu", "Google",
        "HP", "Huawei", "LG", "Lenovo", "MSI", "Mediacom", "Microsoft",
        "Razer", "Samsung", "Toshiba", "Vero", "Xiaomi"
    ]
    type_names = [
        "Notebook", "Ultrabook", "Gaming", "2 in 1 Convertible", "Workstation", "Netbook"
    ]
    cpu_tiers = [
        "Intel Core i5", "Intel Core i7", "Intel Core i3", "AMD Processor",
        "Intel Entry-Level", "Intel Core M", "Intel Xeon"
    ]
    gpu_brands = [
        "Intel (Integrated)", "Nvidia", "AMD (Dedicated)", "AMD (Integrated)"
    ]
    opsys_groups = [
        "Windows", "macOS", "Linux", "Chrome OS", "No OS"
    ]
    common_resolutions = [
        {"label": "Full HD (1920 x 1080)", "width": 1920, "height": 1080},
        {"label": "HD Standard (1366 x 768)", "width": 1366, "height": 768},
        {"label": "2K QHD (2560 x 1440)", "width": 2560, "height": 1440},
        {"label": "Retina (2560 x 1600)", "width": 2560, "height": 1600},
        {"label": "Retina 15 inch (2880 x 1800)", "width": 2880, "height": 1800},
        {"label": "4K UHD (3840 x 2160)", "width": 3840, "height": 2160},
        {"label": "WXGA+ (1600 x 900)", "width": 1600, "height": 900},
    ]
    ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
    ssd_options = [0.0, 128.0, 256.0, 512.0, 1024.0]
    hdd_options = [0.0, 500.0, 1024.0, 2048.0]

    return {
        "companies": companies,
        "type_names": type_names,
        "cpu_tiers": cpu_tiers,
        "gpu_brands": gpu_brands,
        "opsys_groups": opsys_groups,
        "common_resolutions": common_resolutions,
        "ram_options": ram_options,
        "ssd_options": ssd_options,
        "hdd_options": hdd_options,
    }


def get_preset_cases() -> Dict[str, Dict[str, Any]]:
    """
    Cung cấp các trường hợp cấu hình mẫu điển hình để thử nghiệm nhanh.
    """
    return {
        "budget": {
            "name": "Laptop Sinh Viên / Văn Phòng Phổ Thông",
            "desc": "Dell Notebook 15.6 inch, Core i3, RAM 4GB, HDD 1TB",
            "data": {
                "Company": "Dell",
                "TypeName": "Notebook",
                "Inches": 15.6,
                "Resolution_X": 1920,
                "Resolution_Y": 1080,
                "CPU_Frequency (GHz)": 2.0,
                "CPU_Tier": "Intel Core i3",
                "RAM (GB)": 4,
                "Weight (kg)": 2.18,
                "Touchscreen": 0,
                "IPS_Panel": 0,
                "SSD_GB": 0.0,
                "HDD_GB": 1024.0,
                "Flash_Storage_GB": 0.0,
                "Hybrid_GB": 0.0,
                "GPU_Brand_Cat": "Intel (Integrated)",
                "GPU_Dedicated": 0,
                "OpSys_Group": "Windows",
            }
        },
        "premium": {
            "name": "Ultrabook Doanh Nhân Mỏng Nhẹ (Apple)",
            "desc": "MacBook Pro Retina 13.3 inch, Core i7, RAM 16GB, SSD 512GB",
            "data": {
                "Company": "Apple",
                "TypeName": "Ultrabook",
                "Inches": 13.3,
                "Resolution_X": 2560,
                "Resolution_Y": 1600,
                "CPU_Frequency (GHz)": 2.3,
                "CPU_Tier": "Intel Core i7",
                "RAM (GB)": 16,
                "Weight (kg)": 1.37,
                "Touchscreen": 0,
                "IPS_Panel": 1,
                "SSD_GB": 512.0,
                "HDD_GB": 0.0,
                "Flash_Storage_GB": 0.0,
                "Hybrid_GB": 0.0,
                "GPU_Brand_Cat": "Intel (Integrated)",
                "GPU_Dedicated": 0,
                "OpSys_Group": "macOS",
            }
        },
        "gaming": {
            "name": "Laptop Gaming Cao Cấp (Asus ROG)",
            "desc": "Asus Gaming 17.3 inch, Core i7, RAM 32GB, SSD 512GB + HDD 1TB, Nvidia GPU",
            "data": {
                "Company": "Asus",
                "TypeName": "Gaming",
                "Inches": 17.3,
                "Resolution_X": 1920,
                "Resolution_Y": 1080,
                "CPU_Frequency (GHz)": 2.8,
                "CPU_Tier": "Intel Core i7",
                "RAM (GB)": 32,
                "Weight (kg)": 4.2,
                "Touchscreen": 0,
                "IPS_Panel": 1,
                "SSD_GB": 512.0,
                "HDD_GB": 1024.0,
                "Flash_Storage_GB": 0.0,
                "Hybrid_GB": 0.0,
                "GPU_Brand_Cat": "Nvidia",
                "GPU_Dedicated": 1,
                "OpSys_Group": "Windows",
            }
        }
    }
