"""Demo backend configuration.

Loads from environment variables with sensible defaults.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class DemoConfig(BaseSettings):
    """Configuration for the demo backend server."""

    # Index directory (output of build_index.py)
    index_dir: str = Field(default="demo/data/index")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # CORS
    cors_origins: list[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])

    model_config = {"env_prefix": "TALKEX_DEMO_"}
