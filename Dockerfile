FROM python:3.10-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" requests

# HF expects this port
EXPOSE 7860

# Start app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
