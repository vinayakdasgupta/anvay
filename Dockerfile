# Use a stable, widely supported Python base image
FROM python:3.10-slim

# Ensure consistent Python behaviour
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies required by scipy / scikit-learn / gensim
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Flask's default port
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
