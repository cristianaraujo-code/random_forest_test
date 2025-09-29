
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y build-essential

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y modelo
COPY server.py .
COPY model.joblib .

EXPOSE 8080

CMD ["python", "server.py"]