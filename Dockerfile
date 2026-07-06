# CXSentinel — Hugging Face Space (Docker SDK).
# Docker lets the entrypoint write the OAuth secrets BEFORE Streamlit starts,
# so native Google login (st.login) is configured at server boot — which the
# plain Streamlit SDK can't do with env-var secrets.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# HF Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build), then deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code; make everything writable by the runtime user (secrets, sqlite db)
COPY . .
RUN chmod +x entrypoint.sh && chown -R user:user /app

USER user
EXPOSE 7860
CMD ["./entrypoint.sh"]
