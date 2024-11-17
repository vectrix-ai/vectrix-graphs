FROM python:3.11-slim

# Install build dependencies and poppler-utils
RUN apt-get update && apt-get install -y \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies. --no-dev excludes dev dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache --no-dev

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "/app/src/vectrix_graphs/main.py", "--port", "80", "--host", "0.0.0.0"]