"""FastAPI entrypoint for the web dashboard backend."""

import sys
from pathlib import Path

import uvicorn


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    uvicorn.run(
        "src.webapi.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

