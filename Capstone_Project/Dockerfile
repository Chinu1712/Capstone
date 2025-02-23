# Use Python 3.9 to avoid version issues
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Upgrade pip and other tools
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000
ENV CUDA_VISIBLE_DEVICES=""
EXPOSE 5000

# Command to run your app
CMD ["python", "app.py"]
