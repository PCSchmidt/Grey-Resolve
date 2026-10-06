"""REST service (Phase 1): FastAPI app with /ingest, /search, /health.

Semantic contract (carried over from IronClad app.py): responses are ranked
candidates with scores. Candidate ranking is not access authorization; the API
never asserts identity. Implementation lands with the Phase 1 code drop.
"""
