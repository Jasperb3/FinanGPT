"""MongoDB persistence adapter used by application use cases."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable

from pymongo import MongoClient
from pymongo.database import Database

__all__ = ["MongoRepository"]


class MongoRepository:
    def __init__(self, uri: str | None = None, client_factory: Callable[[str], MongoClient] | None = None) -> None:
        self._uri = uri or os.getenv("MONGO_URI")
        if not self._uri:
            raise ValueError("Mongo URI is required")
        self._client_factory = client_factory or MongoClient

    @property
    def uri(self) -> str:
        return self._uri

    @contextmanager
    def connect(self) -> Database:
        client = self._client_factory(self._uri)
        try:
            yield client
        finally:
            try:
                client.close()
            except Exception:
                pass

    def get_database(self, client: MongoClient) -> Database:
        try:
            db = client.get_default_database()
            if db is not None:
                return db
        except Exception:
            pass
        path = self._uri.rsplit("/", 1)[-1]
        if not path:
            raise ValueError("Mongo URI must include a database name")
        return client[path]
