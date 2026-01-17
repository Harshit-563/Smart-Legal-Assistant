FROM python:3.10-slim

WORKDIR /app

# Only install git (needed if some Hugging Face models require it)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (prebuilt wheels → fast install)
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy requirements
COPY requirements.txt .

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# (Optional) Pre-download summarization model at build time
# RUN python -c "from transformers import pipeline; pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
