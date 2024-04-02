FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy Metaflow code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir metaflow pandas scikit-learn sentence-transformers matplotlib seaborn numpy

# Expose port
EXPOSE 8080

# Command to run the Metaflow pipeline
CMD ["python", "-m", "FoodRecommendationPipeline"]
