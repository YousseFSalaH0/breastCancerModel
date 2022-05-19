# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-alpine

# Allow statements and log messages to immediately appear in the Knative logs
EXPOSE 5000/tcp 

# Copy local code to the container image.

WORKDIR /app

COPY requirements.txt .

# Install production dependencies.
RUN pip install -r requirements.txt

Copy app.py .
