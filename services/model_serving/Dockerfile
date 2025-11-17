# Multi-stage build for HomeCredit Model Serving API

# ============================================
# Stage 1: Base image with dependencies
# ============================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user (security best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ============================================
# Stage 2: Install Python dependencies
# ============================================
FROM base AS dependencies

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: Final runtime image
# ============================================
FROM base AS runtime

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ /app/app/

# Create directories for models
RUN mkdir -p /app/models/tuned && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check (using Python since requests is in requirements)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]