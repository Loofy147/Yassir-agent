# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY src/ ./src

# Make port 80 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME YassirPricingService

# Run uvicorn when the container launches
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
