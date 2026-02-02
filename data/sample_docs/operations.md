# Operations Guide

Run the API with Docker Compose to bring up FastAPI and Qdrant together. The compose file mounts a persistent volume for Qdrant so vectors survive restarts. Use the /health endpoint for readiness checks in orchestration platforms. Scale the API service horizontally when query traffic spikes.

When operating in production, set resource limits to avoid runaway memory usage during embedding. Consider pinning the embedding model to a local cache to reduce cold start times. Batch ingestion where possible to keep embedding overhead predictable. Monitor disk usage on the Qdrant volume to avoid running out of space.
