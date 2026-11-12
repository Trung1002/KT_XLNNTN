LAB 9 - Modern NLP Final Project (SSA-VN)
=========================================
Nội dung gói:
  data/               Bộ dữ liệu SSA-VN (tổng hợp, chỉ dùng cho học tập)
  lab9_tools.py       Tạo thẻ cá nhân + gói dữ liệu riêng theo MSSV
  starter_lab9.py     Mã khởi đầu dạng cell (# %%): baseline + các TODO

Bước 1: python lab9_tools.py pack --mssv <MSSV>
        -> tạo thư mục student_pack_<MSSV>/ và in THẺ CÁ NHÂN (check_code)
Bước 2: mở starter_lab9.py (VS Code/Colab), sửa MSSV ở cell 0, chạy cell 0-4
Bước 3: hoàn thành các cell TODO, ghi results.json bằng save_results()

Yêu cầu: Python 3.9+, pandas, numpy, scikit-learn.
Các phần dense/LoRA cần: sentence-transformers, transformers, peft (khuyến nghị Google Colab).
Lưu ý: mọi dữ liệu là GIẢ LẬP; nội dung "quy định" không phải quy chế của bất kỳ trường nào.
