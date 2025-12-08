
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn pymilvus pypdf python-docx pillow psycopg2-binary openai requests

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8050"]
