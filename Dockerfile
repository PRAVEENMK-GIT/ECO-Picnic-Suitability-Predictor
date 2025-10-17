# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port for Flask
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=api.py

# Run the Flask API
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
