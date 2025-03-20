FROM python:3.12-slim

WORKDIR /app

# Install PostgreSQL dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PATH="/app/venv/bin:$PATH"


RUN mkdir -p /data/user_data


ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV BASE_DIR=/data/user_data

# Expose the port
EXPOSE 443

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:443", "app:app"]