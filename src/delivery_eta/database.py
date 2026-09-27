import os

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://eta:eta@localhost:5432/eta"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def initialize() -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS predictions (
                event_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                predicted_eta_minutes DOUBLE PRECISION NOT NULL,
                model_version TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        )


def save_prediction(
    event_id: str, order_id: str, eta: float, model_version: str
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO predictions (event_id, order_id, predicted_eta_minutes, model_version)
            VALUES (:event_id, :order_id, :eta, :model_version)
            ON CONFLICT (event_id) DO NOTHING
        """),
            {
                "event_id": event_id,
                "order_id": order_id,
                "eta": eta,
                "model_version": model_version,
            },
        )
