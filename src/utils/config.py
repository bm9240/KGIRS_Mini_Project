"""Environment-backed application configuration."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class AppSettings:
    root: Path = ROOT
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    dense_model: str = os.getenv(
        "RETRACTTRACE_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    dense_index_path: Path = ROOT / "data" / "processed" / "dense.index"


settings = AppSettings()
