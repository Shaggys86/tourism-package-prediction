# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Create a non-root user
RUN useradd -m -u 1000 user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch to the non-root user
USER user

# Set the working directory inside the container to the user's home app directory
WORKDIR $HOME/app

# Copy application files into the container with the correct ownership
COPY --chown=user . $HOME/app

# Install Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
