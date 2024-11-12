FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir metaflow pandas scikit-learn sentence-transformers fastapi uvicorn groq python-dotenv

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
