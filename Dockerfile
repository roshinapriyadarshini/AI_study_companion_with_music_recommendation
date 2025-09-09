# Use a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose a port (optional)
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
