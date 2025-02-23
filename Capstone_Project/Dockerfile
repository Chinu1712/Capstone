# Use the official Python base image
FROM python:3.9-slim


# Set the working directory
WORKDIR /app

# Copy all files from your local directory to the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (same as Flask)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
