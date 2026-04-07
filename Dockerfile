FROM public.ecr.aws/docker/library/python:3.10

WORKDIR /app

# Upgrade pip (prevents a bunch of random install issues)
RUN pip install --upgrade pip

# Copy only dependency files first (better build stability)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" requests openai

# Now copy the rest of the app
COPY . .

# HF expects this port
EXPOSE 7860

# Start app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
