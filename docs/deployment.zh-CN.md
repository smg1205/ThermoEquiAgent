# 部署

将 `.env.example` 复制为 `.env`，然后执行 `docker compose up --build`。API 与 Web 分别暴露在 8000 和 3000 端口。

本地开发时，先安装 Python 项目与前端依赖，启动 Uvicorn，再启动 Next.js。数据库默认为 SQLite；如需具备迁移能力的部署，设置 PostgreSQL 的 SQLAlchemy URL。在确定性模式下**不需要 OpenAI 密钥**。
