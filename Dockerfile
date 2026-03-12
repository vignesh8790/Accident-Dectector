# Use official Node.js 18 image on Debian Bullseye Slim
# Bullseye-slim provides a lightweight base while including apt-get for system packages
FROM node:18-bullseye-slim

# Set working directory to the app root
WORKDIR /app

# Install Python 3, pip, and required system libraries for OpenCV
# We need libgl1, libglib2.0-0, and ffmpeg for OpenCV to work properly in a container
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink so `python` maps to `python3`
# This ensures spawn('python') in Node works perfectly
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and setuptools before installing YOLO/OpenCV
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# Copy the Python requirements specifically
COPY Engine/requirements.txt /app/Engine/requirements.txt

# Install the Python dependencies (OpenCV, YOLO, etc.)
RUN pip3 install --no-cache-dir -r /app/Engine/requirements.txt

# Now copy the Node.js server dependencies
COPY CrashSenseApp/server/package*.json /app/CrashSenseApp/server/

# Install Node.js dependencies
RUN cd /app/CrashSenseApp/server && npm install --production

# Finally, copy the entire project source code (Engine and Server)
COPY . /app/

# Navigate to the Node.js server directory for the final startup command
WORKDIR /app/CrashSenseApp/server

# Ensure the sample-videos directory exists with correct permissions
RUN mkdir -p ../sample-videos && chmod 777 ../sample-videos

# Set production environment variables
ENV NODE_ENV=production

# Expose the API port
EXPOSE 5000

# Start the Node.js backend server
CMD ["npm", "start"]
