FROM python:3.11-slim

# Install dependensi sistem yang dibutuhkan OpenCV dan library lainnya
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements dan install library python
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy seluruh kode dan file pkl ke dalam container
COPY . .

# Jalankan Uvicorn di port 7860 (Port standar Hugging Face)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]