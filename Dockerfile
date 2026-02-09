# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Note: torch and torchvision are already installed in the base image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Release port 8501 (Streamlit's default port)
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
