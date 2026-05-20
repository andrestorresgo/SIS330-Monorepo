FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline scripts
COPY 01_acquire_data.py .
COPY 02_label_data.py .
COPY 03_chunk_tensors.py .
COPY 04_package_export.py .
COPY run_pipeline.py .

# Run the full pipeline
CMD ["python", "-u", "run_pipeline.py"]
