FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for aspose and generic builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Stanza resources to avoid runtime downloads
RUN python -c "import stanza; stanza.download('vi', processors='tokenize,pos,lemma,depparse')"

COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]