# Dùng bản Python tối giản
FROM python:3.10-slim

WORKDIR /app

# Copy requirement và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code server vào container
COPY . .

# Mở cổng 8000
EXPOSE 8000

# Lệnh chạy FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]