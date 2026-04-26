#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "========================================"
echo " Local RAG - Starting"
echo "========================================"
echo ""

# ── Validate model file exists ─────────────────────────────────────────
MODEL_DIR="./models"
GGUF_COUNT=$(find "$MODEL_DIR" -name "*.gguf" 2>/dev/null | wc -l)

if [ "$GGUF_COUNT" -eq 0 ]; then
    echo "ERROR: No .gguf model found in models/ folder"
    echo ""
    echo "   Place your model file in: $MODEL_DIR/"
    echo "   Example: $MODEL_DIR/mistral-7b-instruct.Q4_K_M.gguf"
    echo ""
    exit 1
fi

echo "Found model(s):"
find "$MODEL_DIR" -name "*.gguf" -exec echo "  → {}" \;
echo ""
# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running"
    exit 1
fi

mkdir -p data vector_db

echo "Building and starting..."
docker compose up --build -d

echo ""
echo "Waiting for llama-server to load model..."

MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    printf "."
done
echo ""

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠️  llama-server may still be loading. Check:"
    echo "   docker compose logs llama"
else
    echo "✅ llama-server ready!"
fi

# Get LAN IP
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

echo ""
echo "========================================"
echo ""
echo "  RAG:     http://localhost:8501"
echo "  LLaMA:   http://localhost:8080"
echo ""
echo "  Commands:"
echo "    Stop:      docker compose down"
echo "    Logs:      docker compose logs -f"
echo "    Restart:   docker compose restart"
echo ""
echo "========================================"
