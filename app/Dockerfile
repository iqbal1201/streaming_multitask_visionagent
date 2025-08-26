# # Use a lightweight Python image
# FROM python:3.11-slim

# # Set working directory inside the container
# WORKDIR /app

# # Install system dependencies (optional, useful for some Google libs)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     libffi-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies first (caching layer)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the code
# COPY . .


# # Set environment variable so Google SDKs know where to find credentials
# ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

# # Expose port 8080 for Cloud Run
# EXPOSE 8080

# # Start FastAPI with Uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


# FROM python:3.11-slim
# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# RUN adduser --disabled-password --gecos "" myuser && \
#     chown -R myuser:myuser /app

# # Copy code with correct permissions
# COPY --chown=myuser:myuser . .

# USER myuser
# ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

# ENV PATH="/home/myuser/.local/bin:$PATH"

# EXPOSE 8080

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# # CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8080"]



FROM python:3.11-slim

# Set workdir inside container
WORKDIR /app

# Copy everything
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make credentials discoverable
# ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account.json"

# Ensure python can import googleWorkspace package
ENV PYTHONPATH=/app

# Expose Cloud Run port
ENV PORT=8080

# Run ADK web server
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8080"]


