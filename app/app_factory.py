"""FastAPI application factory and entrypoint for the AI Travel Planner API."""

import uvicorn

from fastapi import FastAPI

from app.node import node

APP_NAME = "AI Travel Planner"
APP_VERSION = "0.1.0"


def create_app() -> FastAPI:
    """Build the FastAPI application with the single router (node) included."""
    app = FastAPI(title=APP_NAME, version=APP_VERSION)
    app.include_router(node)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.app_factory:app", host="127.0.0.1", port=8000, reload=True)