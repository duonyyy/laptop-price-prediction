"""
Script tiện ích để chạy ứng dụng Web Định Giá Laptop (FastAPI + Streamlit).
Sử dụng:
    python run_demo.py          # Chạy đồng thời cả FastAPI và Streamlit
    python run_demo.py --api    # Chỉ chạy FastAPI backend (Port 8000)
    python run_demo.py --ui     # Chỉ chạy Streamlit frontend (Port 8501)
"""

import argparse
import subprocess
import sys
import time


def run_api():
    print("🚀 Đang khởi động FastAPI Backend tại http://127.0.0.1:8000 ...")
    print("📖 Swagger UI sẵn sàng tại: http://127.0.0.1:8000/docs")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"
    ])


def run_ui():
    print("💻 Đang khởi động Streamlit Frontend tại http://localhost:8501 ...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"
    ])


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy hệ thống Demo Dự Báo Giá Laptop")
    parser.add_argument("--api", action="store_true", help="Chỉ chạy FastAPI server")
    parser.add_argument("--ui", action="store_true", help="Chỉ chạy Streamlit frontend")
    args = parser.parse_args()

    procs = []

    try:
        if args.api:
            p = run_api()
            procs.append(p)
            p.wait()
        elif args.ui:
            p = run_ui()
            procs.append(p)
            p.wait()
        else:
            p_api = run_api()
            procs.append(p_api)
            time.sleep(2)  # Đợi 2s để API khởi động
            p_ui = run_ui()
            procs.append(p_ui)
            print("\n" + "=" * 60)
            print("✨ HỆ THỐNG ĐÃ SẴN SÀNG:")
            print("   👉 FastAPI Docs:  http://127.0.0.1:8000/docs")
            print("   👉 Streamlit Web: http://localhost:8501")
            print("   (Nhấn Ctrl+C để dừng toàn bộ dịch vụ)")
            print("=" * 60 + "\n")
            p_ui.wait()
    except KeyboardInterrupt:
        print("\n🛑 Đang dừng toàn bộ dịch vụ...")
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
