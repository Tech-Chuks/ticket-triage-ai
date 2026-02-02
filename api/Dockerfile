FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy API code
COPY api /app/api

# Copy model metadata used by API (labels.json, etc.)
COPY model-repo /app/model-repo

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9000"]
