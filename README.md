# Dreamweave - Ecommerce Revenue Forecast
## 1. Prerequisites
Trước khi chạy dự án cần đảm bảo các yêu cầu sau:
- Python: >= 3.10
- RAM tối thiểu: 8 GB (khuyến nghị 16GB)
- Hệ điều hành: Window/ Linux/ MacOS

## 2. Installation
Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate # Linux/MacOS
venv\Scripts\activate    # Windows
```
Cài đặt các thư viện cần thiết:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. How to Run
- Bước 1: Tiền xử lý: Chạy file `src/preprocess.py`
```bash
python src/preprocess.py
```
- Bước 2: Feature engineering: chạy file `src/features.py`
```bash
python src/features.py
```
- Bước 3: Train model XGBoost + Catboost: chạy file `src/train.py`
```bash
python src/train.py
```
- Bước 4: Dự đoán Revenue: chạy file `src/predict.py`
```bash
python src/predict.py
```

## 4. Project Structure
```text
DATATHON-2026-ROUND-1/
├── dataset/                # Chứa dữ liệu của cuộc thi
│   ├── raw/                # Dữ liệu gốc chưa qua xử lý (.csv)
│   │   ├── customers.csv
│   │   ├── geography.csv
│   │   ├── inventory.csv
│   │   ├── order_items.csv
│   │   ├── orders.csv
│   │   ├── payments.csv
│   │   ├── products.csv
│   │   ├── promotions.csv
│   │   ├── returns.csv
│   │   ├── reviews.csv
│   │   ├── sales.csv
│   │   ├── sample_submission.csv
│   │   ├── shipments.csv
│   │   └── web_traffic.csv
│   └── processed/          # Dữ liệu đã làm sạch và thực hiện feature engineering
├── models/                 # Lưu trữ các file mô hình đã huấn luyện (.pkl)
├── notebooks/              # Jupyter notebooks phục vụ phân tích và thử nghiệm
│   ├── 01_QA.ipynb         # Kiểm tra chất lượng dữ liệu (Quality Assurance)
│   ├── 02_EDA_V2.ipynb     # Khám phá dữ liệu (Exploratory Data Analysis) phiên bản 2
│   └── EDA_1_7.ipynb       # Các phân tích thăm dò bổ sung
├── src/                    # Mã nguồn chính của dự án
│   ├── __init__.py
│   ├── features.py         # Xử lý biến và đặc trưng (Feature Engineering)
│   ├── preprocess.py       # Tiền xử lý dữ liệu (Cleaning, Scaling, v.v.)
│   ├── train.py            # Huấn luyện mô hình
│   └── predict.py          # Dự đoán revenue
├── venv/                   # Môi trường ảo của Python (Virtual Environment)
├── .gitignore              # Chỉ định các file/thư mục Git không theo dõi
├── README.md               # Tài liệu hướng dẫn dự án
├── requirements.txt        # Danh sách các thư viện cần thiết để chạy dự án
└── submission.csv          # File kết quả dự đoán cuối cùng để nộp bài
```