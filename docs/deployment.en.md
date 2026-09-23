# Deployment

Copy `.env.example` to `.env`, then run `docker compose up --build`. API and web are exposed on 8000
and 3000. For local development, install the Python project and frontend packages, start Uvicorn,
then start Next.js. SQLite is the default; set a PostgreSQL SQLAlchemy URL for migration-ready
deployments. No OpenAI key is required in deterministic mode.
