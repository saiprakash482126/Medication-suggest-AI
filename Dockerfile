FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements_healthcare.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY healthcare_main.py backend.py
COPY cipla_data.json .
COPY healthcare_index.html frontend.html
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]