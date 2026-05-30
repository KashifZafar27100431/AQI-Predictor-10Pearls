FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt requirements-ml.txt pyproject.toml setup.py ./
COPY src ./src
COPY app ./app

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-ml.txt \
    && pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn 'app.flask_api:create_app()' --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120"]
