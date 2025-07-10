# Dockerfile for Smart Travel Advisor Streamlit App

# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/app.py"]
