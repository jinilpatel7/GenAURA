# Stage 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Install system dependencies (ffmpeg is crucial for your audio processing)
# We update the package list, install ffmpeg, then clean up to keep the image small
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Stage 4: Copy the requirements file into the container
COPY requirements.txt .

# Stage 5: Install the Python dependencies
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Stage 6: Copy your application code into the container
# This copies the 'app' directory from your local 'empathetic-ai-therapist' folder
# into the container's '/app' working directory.
COPY ./empathetic-ai-therapist/app ./app

# Stage 7: Expose the port the app runs on
EXPOSE 8000

# Stage 8: Define the command to run your app when the container starts
# The host 0.0.0.0 is essential for the container to be accessible from the outside
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]