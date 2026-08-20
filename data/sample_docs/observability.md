# Observability

Structured JSON logs include a request id so you can trace ingest and query workflows. Metrics are exposed in Prometheus format and count total ingest requests, query requests, errors, abstentions, retrieval strategies, returned citations, and query-latency histogram buckets. The metrics endpoint can be scraped by Prometheus or Grafana Agent. Operators should alert on elevated error rates or sudden drops in ingestion throughput. Adding traces with OpenTelemetry is a natural next step when the deployment grows.
