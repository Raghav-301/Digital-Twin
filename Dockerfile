FROM python:3.10-slim

# Install system dependencies (Poppler, Tesseract, FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run preprocessing pipeline once during build if needed
RUN python -c "import os; open('.pipeline_initialized', 'w').write('done')"

EXPOSE 7860

CMD ["python", "app.py"]
