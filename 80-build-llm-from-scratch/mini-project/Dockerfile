FROM python:3.12-slim

WORKDIR /app

COPY requirements-batch27-serving.txt /app/
RUN pip install --no-cache-dir -r requirements-batch27-serving.txt

COPY 80-build-llm-from-scratch/src/serve.py /app/serve.py

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
