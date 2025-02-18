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
RUN pip install --upgrade pip  
RUN pip install --no-cache-dir numpy==1.23.5 moviepy==1.0.3  # Ensure a stable NumPy version  
RUN pip install --no-cache-dir tensorflow-gpu opencv-python mediapipe fer

# Set the command to run the script  
CMD ["python", "./main.py"]