# Ollama FastAPI Server (v2.0 - Fly.io Ready)

Self-hosted LLM API with API key authentication.

## Automated Deployment via GitHub Actions

This repository is configured to automatically create volumes and deploy to Fly.io whenever you push to the `main` branch.

### One-Time Setup

To enable automated deployment, you need to add your Fly.io API token to your GitHub repository secrets:

1.  **Get your Fly.io API Token**:
    *   Go to the [Fly.io Dashboard](https://fly.io/dashboard).
    *   Navigate to **Account** > **Access Tokens**.
    *   Create a new token (e.g., named "GitHub Actions").
    *   Copy the token.

2.  **Add the token to GitHub Secrets**:
    *   Go to your repository on GitHub: `https://github.com/Phoenix1185/ollama-fastapi-railway-deployment`.
    *   Click on **Settings** > **Secrets and variables** > **Actions**.
    *   Click **New repository secret**.
    *   Name: `FLY_API_TOKEN`.
    *   Value: Paste your Fly.io API token.
    *   Click **Add secret**.

### How it Works

*   **Volume Creation**: The workflow automatically runs `flyctl volumes create ollama_data --region iad --count 2 --size 10 --yes`. If the volumes already exist, it will safely continue.
*   **Deployment**: The workflow then runs `flyctl deploy --remote-only` to deploy your application.
*   **High Availability**: By default, this setup uses 2 volumes in the `iad` region to support Fly.io's high availability mode.

## Manual Setup (Optional)

If you prefer to manage things manually, you can still use the following commands:

### 1. Install flyctl and login
```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. Launch app
```bash
fly launch --name ollama-fastapi-railway-deployment --region iad --no-deploy
```

### 3. Set secrets
```bash
fly secrets set MASTER_KEY=your-strong-master-key-here
```

### 4. Create Persistence (IMPORTANT)
To avoid losing models on restart, you **must** create the storage volumes before deploying:
```bash
fly volume create ollama_data --region iad --count 2 --size 10
```

### 5. Deploy
```bash
fly deploy
```

## Fly.io Free Tier Limits
- **Memory**: 2GB max (this config is optimized for it)
- **CPU**: 1 shared core (optimized for stability)
- **Model**: Use qwen2.5:0.5b (~300MB) or tinyllama (~600MB)
- **Storage**: Persistent via Volumes (models stay saved across restarts)

## Authentication

### Create API Key (needs MASTER_KEY)
```bash
curl -X POST https://your-app.fly.dev/admin/keys \
  -H "X-Master-Key: your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'
```

### Use API Key
```python
import requests

url = "https://your-app.fly.dev/v1/chat/completions"
headers = {
    "Authorization": "Bearer ollama_xxxxxxxx",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "Hello!"}]
}
res = requests.post(url, json=data, headers=headers)
print(res.json())
```

## Pages
- /ui - Dashboard
- /api-docs - API reference
- /docs - Swagger UI
- /redoc - ReDoc
- /health - Status (no auth)

## Models That Fit in 2GB
| Model | Size | Works? |
|-------|------|--------|
| qwen2.5:0.5b | ~300MB | Yes |
| tinyllama | ~600MB | Yes |
| phi3:mini | ~2GB | Maybe (tight) |
| llama3.2:1b | ~1.3GB | Maybe |
# Trigger Deploy
