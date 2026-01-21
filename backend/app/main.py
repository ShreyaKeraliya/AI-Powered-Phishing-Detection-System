from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api.email_routes import router as email_router
from .api.url_routes import router as url_router

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="AI-Powered Phishing Detection for Emails and URLs using Classical ML and Transformers",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(email_router)
    app.include_router(url_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()


