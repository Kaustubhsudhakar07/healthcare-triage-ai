# Cloud & Edge Deployment Guide
## AI-Assisted Pre-Hospital Patient Criticality Prediction System

This guide provides complete, step-by-step production deployment instructions for the **FastAPI Inference Microservice** and **Streamlit Triage Dashboard** across modern cloud platforms (AWS, Google Cloud, Docker, Kubernetes) and edge mobile units.

---

## 1. Local Container Deployment (Docker & Docker Compose)

### 1.1 Building the Docker Image
```bash
# Build the production Docker image
docker build -t prehospital-triage-ai:latest .

# Run the Streamlit Web Application container
docker run -d -p 8501:8501 --name triage-streamlit prehospital-triage-ai:latest

# Or run the FastAPI RESTful Microservice
docker run -d -p 8000:8000 --name triage-api prehospital-triage-ai:latest \
    uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 1.2 Docker Compose Multi-Service Architecture
Create a `docker-compose.yml` to launch both the FastAPI backend and Streamlit frontend concurrently:

```yaml
version: '3.8'

services:
  api:
    build: .
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build: .
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    restart: always
    depends_on:
      - api
```

Run with:
```bash
docker compose up -d
```

---

## 2. Serverless Cloud Deployment (Google Cloud Run)

Google Cloud Run provides auto-scaling serverless container execution ideal for fluctuating emergency call volumes.

### 2.1 Deployment Steps
```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# 2. Build and submit image to Google Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/prehospital-triage-api:latest

# 3. Deploy to Cloud Run with automatic concurrency scaling
gcloud run deploy prehospital-triage-api \
    --image gcr.io/YOUR_GCP_PROJECT_ID/prehospital-triage-api:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 1 \
    --max-instances 10
```

---

## 3. Enterprise AWS Deployment (ECS Fargate + Application Load Balancer)

### 3.1 Architecture Overview
- **AWS Route 53:** DNS resolution with HTTPS certificate termination (AWS Certificate Manager).
- **Application Load Balancer (ALB):** SSL termination and path-based routing (`/api/*` $\rightarrow$ FastAPI target group, `/*` $\rightarrow$ Streamlit target group).
- **AWS ECS Fargate:** Serverless container execution with automated CPU/Memory auto-scaling.
- **AWS CloudWatch:** Real-time latency, error rates, and custom drift metric alarms.

---

## 4. Kubernetes Deployment Manifests (K8s)

### 4.1 Deployment & Service Manifest (`k8s-deployment.yaml`)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triage-api-deployment
  labels:
    app: triage-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: triage-api
  template:
    metadata:
      labels:
        app: triage-api
    spec:
      containers:
      - name: triage-api
        image: prehospital-triage-ai:latest
        command: ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "1000m"
            memory: "1024Mi"
          requests:
            cpu: "250m"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: triage-api-service
spec:
  type: LoadBalancer
  selector:
    app: triage-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

Apply to cluster:
```bash
kubectl apply -f k8s-deployment.yaml
```

---

## 5. Edge Deployment on Mobile Ambulance Data Terminals (MDTs)

In pre-hospital field operations with unreliable satellite/cellular bandwidth:
1. **Lightweight Edge Container:** Pre-install the Docker image locally on the ambulance tablet (Panasonic Toughbook / iPad).
2. **Local Loopback Inference:** The local UI calls `http://localhost:8000/predict` directly, eliminating network latency and connectivity dependencies.
3. **Async Telemetry Sync:** When 4G/5G connection is established, the tablet queues and transmits encrypted records (`docs/SECURITY_PRIVACY.md`) to the hospital receiving dashboard via mutual TLS.
