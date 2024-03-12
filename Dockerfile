# Use an official Python runtime as a parent image  
FROM python:3.9  
  
# Set the working directory in the container to /app  
WORKDIR /app  
  
# Add the current directory contents into the container at /app  
ADD . /app  
  
# Install any needed packages specified in requirements.txt  
RUN pip install --no-cache-dir uvicorn fastapi llama_index  
  
# Make port 80 available to the world outside this container  
EXPOSE 80  
  
# Run app.py when the container launches  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]  

ENV OPENAI_API_KEY="OPENAI_API_KEY"
