FROM python:3.13-slim

LABEL maintainer="Manthan Sagar"
LABEL description="Cross-Sectional Equity Forecasting & Friction-Adjusted L/S Backtest"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY tests/ tests/

# Create data and output directories
RUN mkdir -p data/raw data/processed models outputs

# Default entrypoint: run the full pipeline
ENTRYPOINT ["python", "-m", "src.run_pipeline"]
