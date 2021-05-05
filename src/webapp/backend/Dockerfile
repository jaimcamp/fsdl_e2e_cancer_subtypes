FROM python:3.9-slim

WORKDIR /app

RUN apt-get update

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
