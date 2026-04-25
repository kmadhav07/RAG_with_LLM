FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY default_pdfs/ default_pdfs/

# Create uploads directory
RUN mkdir -p uploads

# Expose the HF Spaces port
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--threads", "4", "app:app"]
