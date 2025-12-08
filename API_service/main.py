
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, files, logs, config_routes
from app.services.milvus_service import init_milvus

def create_app():
    app = FastAPI(title="Qwen Full RAG API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_headers=["*"],
        allow_methods=["*"],
    )

    app.include_router(chat.router)
    app.include_router(files.router)
    app.include_router(logs.router)
    app.include_router(config_routes.router)

    app.add_event_handler("startup", init_milvus)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
