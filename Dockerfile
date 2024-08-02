# Use the official Python 3.8 image as a base
FROM python:3.7-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run a command (replace this with your actual entry point)
CMD ["python3", "src/main.py"]

