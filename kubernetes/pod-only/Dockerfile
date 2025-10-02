FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional, for better compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY download_model.py main.py ./

# Expose application port
EXPOSE 8080

# Use a non-root user for security (optional)
# RUN useradd --create-home appuser
# USER appuser

# Start the application
CMD ["python", "main.py"]