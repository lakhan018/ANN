# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY dev_requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r dev_requirements.txt

# Copy the rest of the application's code
COPY . .

# (Optional) Command to run your application, e.g., a streamlit app or API
# CMD ["streamlit", "run", "app/Home.py"]