# Use a lightweight Python base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside container
WORKDIR /app

# Install system deps (build tools, etc. if needed by spacy/skillNer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model inside the image
RUN python -m spacy download en_core_web_sm

# Copy the rest of the project
COPY . /app

# Expose Flask port
EXPOSE 5000

# Default command to run the Flask app
# Adjust if your app uses a different entry point
CMD ["python", "app.py"]
