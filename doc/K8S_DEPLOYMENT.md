# Kubernetes Deployment Notes

These are reference snippets to run MedCHR with an API deployment, a worker deployment, and a purge CronJob.
Adjust resources and secrets to your environment.

## API Deployment (sample)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medchr-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: medchr-api
  template:
    metadata:
      labels:
        app: medchr-api
    spec:
      containers:
        - name: api
          image: your-registry/medchr:latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: medchr-secrets
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
```

## Worker Deployment (sample)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medchr-worker
spec:
  replicas: 1
  selector:
    matchLabels:
      app: medchr-worker
  template:
    metadata:
      labels:
        app: medchr-worker
    spec:
      containers:
        - name: worker
          image: your-registry/medchr:latest
          command: ["python", "-m", "backend.scripts.worker"]
          envFrom:
            - secretRef:
                name: medchr-secrets
```

## Purge CronJob (sample)
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: medchr-purge
spec:
  schedule: "0 3 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: purge
              image: your-registry/medchr:latest
              command: ["python", "-m", "backend.scripts.purge_data", "--execute"]
              envFrom:
                - secretRef:
                    name: medchr-secrets
```

## Ingress / Proxy Headers
- Ensure your ingress sets the `Host` header (for `ALLOWED_HOSTS`).
- Enable proxy headers so rate limiting sees the real client IP.
