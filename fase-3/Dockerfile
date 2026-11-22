FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY train.py predict.py api.py cliente.py ./

RUN mkdir -p /app/data /app/models

ENV MODELS_DIR=/app/models
ENV DATA_PATH=/app/data/DataSet_House_Price_Prediction.csv
ENV MODEL_VERSION=v3

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
