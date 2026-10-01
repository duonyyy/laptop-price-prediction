import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests
import streamlit as st

from app.service import (
    EUR_TO_USD_RATE,
    EUR_TO_VND_RATE,
    get_preset_cases,
    get_specs_options,
    predict_price,
)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ Thống Định Giá Laptop AI | Machine Learning Demo",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS giao diện hiện đại, trực quan cao
st.markdown(
    """
    <style>
    /* Gradient Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .hero-desc {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-bottom: 0;
    }
    /* Price Result Card */
    .price-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 30px -10px rgba(56, 189, 248, 0.25);
        margin: 16px 0;
    }
    .price-main {
        font-size: 2.8rem;
        font-weight: 900;
        color: #38bdf8;
        margin: 8px 0;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .price-sub {
        font-size: 1.25rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .badge-tier {
        display: inline-block;
        background: #0369a1;
        color: #e0f2fe;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 10px;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .status-pill-online {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-pill-offline {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API Endpoint cấu hình
FASTAPI_BASE_URL = "http://127.0.0.1:8000"


@st.cache_data(ttl=5)
def check_fastapi_health() -> bool:
    """Kiểm tra tình trạng FastAPI có đang chạy không"""
    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=1.2)
        return r.status_code == 200
    except Exception:
        return False


# Lấy các tùy chọn danh mục cấu hình
specs_meta = get_specs_options()
preset_cases = get_preset_cases()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/laptop.png", width=70)
    st.markdown("### ⚙️ Điều Khiển & Cấu Hình")

    # Kiểm tra trạng thái FastAPI
    api_online = check_fastapi_health()
    if api_online:
        st.markdown(
            'Chế độ Backend: <span class="status-pill-online">🟢 FastAPI Online (Port 8000)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            'Chế độ Backend: <span class="status-pill-offline">🟡 Direct Python Engine</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### ⚡ Chọn Nhanh Cấu Hình Mẫu:")

    selected_preset = None
    col_p1, col_p2 = st.columns(2)
    if col_p1.button("🎒 Sinh Viên (Dell)", use_container_width=True):
        selected_preset = preset_cases["budget"]["data"]
    if col_p2.button("💼 Doanh Nhân (Apple)", use_container_width=True):
        selected_preset = preset_cases["premium"]["data"]
    if st.button("🔥 Gaming Cấu Hình Khủng (Asus)", use_container_width=True):
        selected_preset = preset_cases["gaming"]["data"]

    # Lưu preset vào session_state nếu người dùng nhấn chọn
    if selected_preset:
        for k, v in selected_preset.items():
            st.session_state[f"field_{k}"] = v
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💱 Tỷ Giá Quy Đổi")
    exchange_rate = st.number_input(
        "Tỷ giá EUR / VNĐ",
        min_value=20000,
        max_value=35000,
        value=int(EUR_TO_VND_RATE),
        step=500,
    )

    st.markdown("---")
    st.markdown(
        """
        **Đồ án Data Science**:
        * Mô hình: `XGBRegressor Pipeline`
        * Features: 19 đặc trưng phần cứng
        * Framework: FastAPI + Streamlit
        """
    )


# ----------------- HERO BANNER -----------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">💻 Hệ Thống Định Giá Laptop AI</div>
        <p class="hero-desc">
            Ứng dụng mô hình học máy tối ưu (XGBoost) kết hợp FastAPI & Streamlit để định giá chính xác theo cấu hình phần cứng thời gian thực.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💡 Định Giá Trực Quan",
        "⚡ Nâng Cấp Cấu Hình (What-If)",
        "📊 Đối Sánh Thị Trường",
        "🔌 Tài Liệu API & Swagger",
    ]
)

# ================= TAB 1: ĐỊNH GIÁ TRỰC QUAN =================
with tab1:
    st.markdown("#### 🛠️ Nhập Thông Số Cấu Hình Laptop Cần Định Giá")

    c1, c2, c3, c4 = st.columns(4)

    # Cột 1: Hãng sản xuất & Loại máy
    with c1:
        st.markdown("**1. Thương Hiệu & Thiết Kế**")
        default_company = st.session_state.get("field_Company", "Dell")
        company_idx = (
            specs_meta["companies"].index(default_company)
            if default_company in specs_meta["companies"]
            else 0
        )
        company = st.selectbox("Hãng sản xuất:", specs_meta["companies"], index=company_idx)

        default_type = st.session_state.get("field_TypeName", "Notebook")
        type_idx = (
            specs_meta["type_names"].index(default_type)
            if default_type in specs_meta["type_names"]
            else 0
        )
        type_name = st.selectbox("Phân khúc loại máy:", specs_meta["type_names"], index=type_idx)

        default_opsys = st.session_state.get("field_OpSys_Group", "Windows")
        opsys_idx = (
            specs_meta["opsys_groups"].index(default_opsys)
            if default_opsys in specs_meta["opsys_groups"]
            else 0
        )
        opsys = st.selectbox("Hệ điều hành:", specs_meta["opsys_groups"], index=opsys_idx)

    # Cột 2: Vi xử lý & Đồ họa
    with c2:
        st.markdown("**2. Vi Xử Lý & Đồ Họa**")
        default_cpu_tier = st.session_state.get("field_CPU_Tier", "Intel Core i5")
        cpu_idx = (
            specs_meta["cpu_tiers"].index(default_cpu_tier)
            if default_cpu_tier in specs_meta["cpu_tiers"]
            else 0
        )
        cpu_tier = st.selectbox("Dòng CPU:", specs_meta["cpu_tiers"], index=cpu_idx)

        default_freq = float(st.session_state.get("field_CPU_Frequency (GHz)", 2.5))
        cpu_freq = st.slider("Xung nhịp CPU (GHz):", min_value=1.0, max_value=4.0, value=default_freq, step=0.1)

        default_gpu_brand = st.session_state.get("field_GPU_Brand_Cat", "Intel (Integrated)")
        gpu_idx = (
            specs_meta["gpu_brands"].index(default_gpu_brand)
            if default_gpu_brand in specs_meta["gpu_brands"]
            else 0
        )
        gpu_brand = st.selectbox("Thương hiệu GPU:", specs_meta["gpu_brands"], index=gpu_idx)

        default_gpu_ded = int(st.session_state.get("field_GPU_Dedicated", 0))
        gpu_dedicated = st.checkbox("Có GPU Card Rời?", value=bool(default_gpu_ded))

    # Cột 3: Màn hình hiển thị
    with c3:
        st.markdown("**3. Màn Hình & Trọng Lượng**")
        default_inches = float(st.session_state.get("field_Inches", 15.6))
        inches = st.slider("Kích thước màn hình (inch):", min_value=11.6, max_value=17.3, value=default_inches, step=0.1)

        # Chọn độ phân giải
        res_labels = [r["label"] for r in specs_meta["common_resolutions"]]
        res_choice = st.selectbox("Độ phân giải màn hình:", res_labels, index=0)
        selected_res = next(r for r in specs_meta["common_resolutions"] if r["label"] == res_choice)
        res_x = selected_res["width"]
        res_y = selected_res["height"]

        default_weight = float(st.session_state.get("field_Weight (kg)", 2.0))
        weight = st.slider("Trọng lượng (kg):", min_value=0.8, max_value=4.5, value=default_weight, step=0.05)

        col_scr1, col_scr2 = st.columns(2)
        default_touch = int(st.session_state.get("field_Touchscreen", 0))
        default_ips = int(st.session_state.get("field_IPS_Panel", 0))
        touchscreen = col_scr1.checkbox("Cảm ứng", value=bool(default_touch))
        ips_panel = col_scr2.checkbox("Tấm nền IPS", value=bool(default_ips))

    # Cột 4: Bộ nhớ RAM & Ổ cứng
    with c4:
        st.markdown("**4. Bộ Nhớ & Ổ Cứng**")
        default_ram = int(st.session_state.get("field_RAM (GB)", 8))
        ram_idx = specs_meta["ram_options"].index(default_ram) if default_ram in specs_meta["ram_options"] else 3
        ram = st.selectbox("Dung lượng RAM (GB):", specs_meta["ram_options"], index=ram_idx)

        default_ssd = float(st.session_state.get("field_SSD_GB", 256.0))
        ssd_idx = specs_meta["ssd_options"].index(default_ssd) if default_ssd in specs_meta["ssd_options"] else 2
        ssd = st.selectbox("Ổ cứng SSD (GB):", specs_meta["ssd_options"], index=ssd_idx)

        default_hdd = float(st.session_state.get("field_HDD_GB", 0.0))
        hdd_idx = specs_meta["hdd_options"].index(default_hdd) if default_hdd in specs_meta["hdd_options"] else 0
        hdd = st.selectbox("Ổ cứng HDD phụ (GB):", specs_meta["hdd_options"], index=hdd_idx)

    st.markdown("---")

    # Đóng gói dữ liệu đầu vào
    input_payload = {
        "Company": company,
        "TypeName": type_name,
        "Inches": inches,
        "Resolution_X": res_x,
        "Resolution_Y": res_y,
        "CPU_Frequency (GHz)": cpu_freq,
        "CPU_Tier": cpu_tier,
        "RAM (GB)": ram,
        "Weight (kg)": weight,
        "Touchscreen": 1 if touchscreen else 0,
        "IPS_Panel": 1 if ips_panel else 0,
        "SSD_GB": ssd,
        "HDD_GB": hdd,
        "Flash_Storage_GB": 0.0,
        "Hybrid_GB": 0.0,
        "GPU_Brand_Cat": gpu_brand,
        "GPU_Dedicated": 1 if gpu_dedicated else 0,
        "OpSys_Group": opsys,
    }

    # Nút bấm tính toán
    col_btn, _ = st.columns([1, 3])
    predict_clicked = col_btn.button("🚀 DỰ BÁO GIÁ NGAY", use_container_width=True, type="primary")

    # Dự báo ngay khi bấm nút hoặc có cấu hình
    if predict_clicked or "last_prediction" in st.session_state:
        prediction_data = None

        # 1. Thử gọi qua FastAPI trước
        if api_online:
            try:
                resp = requests.post(f"{FASTAPI_BASE_URL}/predict", json=input_payload, timeout=2.0)
                if resp.status_code == 200:
                    prediction_data = resp.json()
            except Exception:
                pass

        # 2. Fallback trực tiếp nếu API không chạy
        if prediction_data is None:
            prediction_data = predict_price(input_payload)

        # Tính lại tiền Việt theo tỷ giá điều chỉnh trên sidebar
        custom_vnd = prediction_data["predicted_price_euro"] * exchange_rate
        st.session_state["last_prediction"] = prediction_data

        # Hiển thị Card Kết Quả
        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-sub">💰 GIÁ DỰ BÁO THỊ TRƯỜNG KỲ VỌNG</div>
                <div class="price-main">{custom_vnd:,.0f} VNĐ</div>
                <div class="price-sub">Tương đương: <b>€{prediction_data['predicted_price_euro']:,.2f}</b> | <b>{prediction_data['predicted_price_usd']}</b></div>
                <div><span class="badge-tier">{prediction_data['price_tier']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hiển thị các chỉ số tóm tắt
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mật độ PPI", f"{prediction_data['calculated_ppi']} ppi")
        m2.metric("Tổng lưu trữ", f"{int(prediction_data['total_storage_gb'])} GB")
        m3.metric("Loại ổ cứng", "Có SSD ⚡" if prediction_data['has_ssd'] == 1 else "Chỉ HDD")
        m4.metric("Dung lượng RAM", f"{ram} GB")
        m5.metric("Khối lượng", f"{weight} kg")


# ================= TAB 2: NÂNG CẤP CẤU HÌNH (WHAT-IF) =================
with tab2:
    st.markdown("#### ⚡ Mô Phỏng Biến Động Giá Khi Nâng Cấp Phần Cứng")
    st.info("Mô phỏng giúp khách hàng và người kinh doanh biết chính xác việc nâng thêm RAM, gắn thêm SSD hay chọn CPU cao hơn sẽ làm giá thay đổi bao nhiêu.")

    base_payload = input_payload.copy()
    base_res = predict_price(base_payload)
    base_price_vnd = base_res["predicted_price_euro"] * exchange_rate

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown("##### 📌 Cấu Hình Hiện Tại")
        st.markdown(f"**{base_payload['Company']} {base_payload['TypeName']}**")
        st.markdown(f"- CPU: `{base_payload['CPU_Tier']}` ({base_payload['CPU_Frequency (GHz)']} GHz)")
        st.markdown(f"- RAM: `{base_payload['RAM (GB)']} GB` | SSD: `{int(base_payload['SSD_GB'])} GB`")
        st.markdown(f"- Màn hình: `{base_payload['Inches']} inch` ({base_payload['Resolution_X']}x{base_payload['Resolution_Y']})")
        st.markdown(f"👉 **Giá hiện tại: {base_price_vnd:,.0f} VNĐ**")

    with col_w2:
        st.markdown("##### 🚀 Tùy Chọn Nâng Cấp")
        upgrade_type = st.radio(
            "Chọn phương án nâng cấp muốn thử nghiệm:",
            [
                "Nâng cấp RAM (+8GB)",
                "Nâng cấp SSD (+256GB)",
                "Đổi sang CPU Intel Core i7",
                "Thêm Màn hình Cảm ứng & IPS",
                "Thêm Card Đồ Họa Rời (Dedicated GPU)",
            ],
        )

        upgraded_payload = base_payload.copy()
        if upgrade_type == "Nâng cấp RAM (+8GB)":
            upgraded_payload["RAM (GB)"] = base_payload["RAM (GB)"] + 8
        elif upgrade_type == "Nâng cấp SSD (+256GB)":
            upgraded_payload["SSD_GB"] = base_payload["SSD_GB"] + 256
        elif upgrade_type == "Đổi sang CPU Intel Core i7":
            upgraded_payload["CPU_Tier"] = "Intel Core i7"
            upgraded_payload["CPU_Frequency (GHz)"] = max(base_payload["CPU_Frequency (GHz)"], 2.8)
        elif upgrade_type == "Thêm Màn hình Cảm ứng & IPS":
            upgraded_payload["Touchscreen"] = 1
            upgraded_payload["IPS_Panel"] = 1
        elif upgrade_type == "Thêm Card Đồ Họa Rời (Dedicated GPU)":
            upgraded_payload["GPU_Dedicated"] = 1
            upgraded_payload["GPU_Brand_Cat"] = "Nvidia"

        upgraded_res = predict_price(upgraded_payload)
        upgraded_price_vnd = upgraded_res["predicted_price_euro"] * exchange_rate
        diff_vnd = upgraded_price_vnd - base_price_vnd

        st.metric(
            label=f"Giá sau khi {upgrade_type}",
            value=f"{upgraded_price_vnd:,.0f} VNĐ",
            delta=f"{diff_vnd:+,.0f} VNĐ ({diff_vnd / base_price_vnd * 100:+.1f}%)",
        )


# ================= TAB 3: ĐỐI SÁNH THỊ TRƯỜNG =================
with tab3:
    st.markdown("#### 📊 Đối Sánh Với Dữ Liệu Thị Trường Thực Tế")

    data_file = Path("datas/laptop_price_cleaned.csv")
    if data_file.exists():
        df_clean = pd.read_csv(data_file)

        # Lọc dữ liệu theo Hãng đang chọn
        company_df = df_clean[df_clean["Company"] == company]
        type_df = df_clean[df_clean["TypeName"] == type_name]

        b1, b2, b3 = st.columns(3)
        b1.metric(
            f"Giá TB dòng máy của {company}",
            f"{company_df['Price (Euro)'].mean() * exchange_rate:,.0f} VNĐ",
            f"Số lượng mẫu: {len(company_df)}",
        )
        b2.metric(
            f"Giá TB phân khúc {type_name}",
            f"{type_df['Price (Euro)'].mean() * exchange_rate:,.0f} VNĐ",
            f"Số lượng mẫu: {len(type_df)}",
        )
        b3.metric(
            "Giá cao nhất thị trường trong tập dữ liệu",
            f"{df_clean['Price (Euro)'].max() * exchange_rate:,.0f} VNĐ",
        )

        st.markdown(f"##### 📋 Top 5 Laptop thuộc hãng `{company}` trong bộ dữ liệu gốc:")
        preview_cols = ["Company", "TypeName", "Inches", "RAM (GB)", "SSD_GB", "CPU_Tier", "Price (Euro)"]
        sample_table = company_df[preview_cols].copy().head(5)
        sample_table["Giá (VNĐ)"] = (sample_table["Price (Euro)"] * exchange_rate).apply(lambda x: f"{x:,.0f} VNĐ")
        st.dataframe(sample_table, use_container_width=True)
    else:
        st.warning("Chưa tìm thấy tệp `datas/laptop_price_cleaned.csv` để hiển thị đối sánh thị trường.")


# ================= TAB 4: TÀI LIỆU API & SWAGGER =================
with tab4:
    st.markdown("#### 🔌 Hướng Dẫn Tích Hợp FastAPI Cho Lập Trình Viên")
    st.markdown(
        """
        FastAPI cung cấp giao thức REST API chuẩn giúp bạn dễ dàng tích hợp mô hình Machine Learning vào ứng dụng di động, website bán hàng hoặc hệ thống ERP.
        
        * **Tài liệu Swagger UI tương tác**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
        * **Tài liệu ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
        """
    )

    st.markdown("##### 💻 Ví Dụ Gọi API Bằng Python:")
    sample_code = """import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "Company": "Dell",
    "TypeName": "Notebook",
    "Inches": 15.6,
    "Resolution_X": 1920,
    "Resolution_Y": 1080,
    "CPU_Frequency (GHz)": 2.5,
    "CPU_Tier": "Intel Core i5",
    "RAM (GB)": 8,
    "Weight (kg)": 2.0,
    "Touchscreen": 0,
    "IPS_Panel": 0,
    "SSD_GB": 256.0,
    "HDD_GB": 0.0,
    "Flash_Storage_GB": 0.0,
    "Hybrid_GB": 0.0,
    "GPU_Brand_Cat": "Intel (Integrated)",
    "GPU_Dedicated": 0,
    "OpSys_Group": "Windows"
}

response = requests.post(url, json=payload)
print(response.json())
"""
    st.code(sample_code, language="python")

    st.markdown("##### ⚡ Ví Dụ Gọi Bằng cURL:")
    sample_curl = """curl -X POST "http://127.0.0.1:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"Company": "Apple", "TypeName": "Ultrabook", "Inches": 13.3, "RAM (GB)": 16, "SSD_GB": 512, "CPU_Tier": "Intel Core i7", "GPU_Brand_Cat": "Intel (Integrated)", "OpSys_Group": "macOS"}'
"""
    st.code(sample_curl, language="bash")
