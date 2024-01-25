# Use an official Python runtime as a parent image
FROM python:3.10.2-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD checkpoints/inference.ckpt /app/checkpoints.ckpt
ADD requirements.txt /app/requirements.txt
ADD src /app/src
ADD *.py /app/
ADD public /app/public
ADD templates /app/templates

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run flask app
CMD [ "python", "app.py" ]