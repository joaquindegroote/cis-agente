﻿FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["sh","-c","streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT}"]

