# Use an official Python runtime as a parent image
FROM python:3.11.7-slim-bullseye

# Set the working directory in the container to /app
WORKDIR /app

ARG GRADIO_SERVER_PORT=8082
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

# Add the current directory contents into the container at /app
ADD checkpoints/inference.ckpt /app/checkpoints/inference.ckpt
ADD requirements.txt /app/requirements.txt
ADD src /app/src
ADD *.py /app/
ADD images /app/images

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


EXPOSE 8082

# Run gradio_app.py when the container launches
CMD [ "python", "gradio_app.py" ]