# Start from a Playwright base image with Python support.
# This image contains all system dependencies needed for Chromium.
FROM mcr.microsoft.com/playwright/python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files
COPY requirements.txt .
COPY .env .

# Install Python dependencies (from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# This command ensures the Playwright browser is fully downloaded inside the container
# (Often already done by the base image, but it's a good safety measure)
RUN playwright install chromium

# Copy the rest of the application code
COPY main.py .
COPY solver.py .

# Expose the port for FastAPI
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]