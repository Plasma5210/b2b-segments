FROM python:3.10-slim

# Устанавливаем libgomp1 — это и есть та самая библиотека OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем проект и ставим зависимости
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем Streamlit (порт $PORT подставит Layero)
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
