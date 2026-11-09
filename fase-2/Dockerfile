FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY train.py predict.py ./
COPY data ./data

RUN mkdir -p models outputs

ENTRYPOINT ["python"]
CMD ["train.py", "--help"]
