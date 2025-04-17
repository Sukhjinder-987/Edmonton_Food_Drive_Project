# Base Image
FROM python:3.10-slim

# Set the working directory 
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create a logs directory in the container
RUN mkdir -p /app/logs

# Set an environment variable to define log path
ENV LOG_DIR=/app/logs

# Set the environment variable for the application
ENV PYTHONPATH=/app

# Set the environment variable for Flask
ENV FLASK_APP=predict_api.py

# Make port 5000 available to the world outside this container
EXPOSE 5000

CMD ["python", "src/predict_api.py"]