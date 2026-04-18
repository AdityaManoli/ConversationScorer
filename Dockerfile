FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ollama runs on the host; point to host.docker.internal or override via env
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
ENV OLLAMA_MODEL=llama3
ENV BATCH_SIZE=20

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
