FROM python:3.9  

# Set working directory  
WORKDIR /app  

# Install system dependencies  
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg  

# Copy necessary files  
COPY main.py .  

# Upgrade pip and install Python dependencies  
RUN pip install --upgrade pip && pip install tensorflow-gpu opencv-python mediapipe  

# Set the command to run the script  
CMD ["python", "./main.py"]