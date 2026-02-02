# Ticket Triage AI (Triton + FastAPI + Prometheus + Grafana)

A production-style mini project:
- Train a text intent classifier
- Serve the model with NVIDIA Triton (ONNX)
- Call Triton from a FastAPI service
- Expose metrics and visualize with Prometheus + Grafana

## Status
Project runs locally:
- Triton: http://localhost:8000
- API: http://localhost:9000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Quickstart
docker compose up -d --build

curl -s http://localhost:9000/health
curl -s -X POST http://localhost:9000/triage \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"TCK-1","message":"I was charged twice and need a refund ASAP. Email me at test@example.com"}'

