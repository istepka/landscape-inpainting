# Use an official Python runtime as a parent image
FROM python:3.10.2-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD mlflow /app/mlflow
ADD mlruns /app/mlruns
ADD lightning_logs /app/lightning_logs
ADD requirements.txt /app/requirements.txt
ADD src /app/src
ADD *.py /app/
ADD *.ipynb /app/
ADD *.csv /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run mlflow ui when the container launches
CMD ["mlflow", "ui", "--port", "5000"]