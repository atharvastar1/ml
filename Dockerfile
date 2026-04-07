# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for FAISS and system builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Using --no-cache-dir to keep image small
RUN pip install --no-cache-dir torch torch-geometric numpy==1.26.4 pandas scikit-learn fastapi uvicorn streamlit requests faiss-cpu matplotlib plotly pyyaml

# Copy the rest of the application
COPY . .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000

# Script to run the app (we'll create this next)
CMD ["streamlit", "run", "ui/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
