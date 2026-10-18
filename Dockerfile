FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .

RUN mkdir -p atmos_bundle

EXPOSE 7860

ENV PORT=7860

CMD ["python", "serve.py"]
