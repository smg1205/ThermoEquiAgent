FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY agent agent
COPY apps apps
COPY database database
COPY knowledge knowledge
COPY schemas schemas
COPY thermo_engine thermo_engine
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
