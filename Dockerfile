# Use the official Python 3.7.16 image as a base
FROM python:3.7.16-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify a user
RUN adduser --disabled-password plex

USER plex
