"""Streaming mini-project: Kafka -> processor -> anomaly models -> PostgreSQL -> Streamlit.

Every layer imports its names, topics, event shapes, ports and table definitions from
``hypersplat.streaming.contract``. This init stays empty so the ACE env (py3.8) can import
the contract alone, the same rule as ``hypersplat.stream``.
"""
