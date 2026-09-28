"""Database connect wrappers — single sanctioned path for TimescaleDB + Milvus.

All code that touches Postgres or Milvus MUST import from this package:

    from src.db.timescale import connect as tsdb_connect
    from src.db.milvus import connect as milvus_connect

Direct use of psycopg.connect / pymilvus.MilvusClient outside this package is
forbidden — see PROJECT.md ERR-01 (fail-loud, no silent fallback).
"""
