# Stage 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Install system dependencies (ffmpeg is crucial for your audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Stage 4: Copy the requirements file into the container
COPY requirements.txt .

# Stage 5: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 6: Copy your application code into the container
COPY ./empathetic-ai-therapist/app ./app

# Stage 7: Expose the port the app runs on (This is just documentation, the CMD is what matters)
EXPOSE 8080

# Stage 8: Define the command to run your app when the container starts
# MODIFIED: Use the $PORT environment variable provided by Cloud Run,
# defaulting to 8000 for local development. Note the change from [] to shell form.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}