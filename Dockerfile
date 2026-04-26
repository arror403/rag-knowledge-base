FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_engine.py .
COPY main.py .
COPY .streamlit/ .streamlit/

# Copy the embedding model into the image
# This means NO internet needed at runtime
COPY embedding_model/ embedding_model/

RUN mkdir -p data vector_db

# Disable all HuggingFace online access
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1



EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "main.py", \
            "--server.headless", "true", \
            "--server.address", "0.0.0.0", \
            "--server.port", "8501"]
