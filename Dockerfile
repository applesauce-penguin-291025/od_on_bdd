FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY data_analysis/ data_analysis/
COPY model/ model/
COPY utils/ utils/
COPY docs/ docs/
COPY outputs/ outputs/
COPY report_assets/ report_assets/
COPY best.pt .
COPY README.md .
COPY bdd100k.yaml .



CMD ["/bin/bash"]
