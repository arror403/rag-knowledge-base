# 🛠️ Local RAG Knowledge Base

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system.
Ask questions about your documents — everything runs on your own machine.
No data ever leaves your computer.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![llama.cpp](https://img.shields.io/badge/llama.cpp-LLM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [GPU Setup](#gpu-setup-recommended)
- [Usage Guide](#usage-guide)
- [Model Recommendations](#model-recommendations)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## How It Works

```text
Your Documents                  Your Question
(PDF, TXT, MD)                       │
      │                              │
      ▼                              ▼
┌─────────────┐            ┌─────────────────┐
│  Ingestion  │            │    Retrieval    │
│             │            │                 │
│ Split into  │──────────► │ Find the most   │
│ chunks and  │  Indexed   │ relevant chunks │
│ embed them  │            │ for your query  │
└─────────────┘            └────────┬────────┘
      │                             │
      ▼                             ▼
┌─────────────┐            ┌─────────────────┐
│  FAISS      │            │   LLM Answer    │
│  Vector DB  │            │                 │
│  (local)    │            │ Context + Query │
└─────────────┘            │ → Local LLM     │
                           │ → Your Answer   │
                           └─────────────────┘

```
### How It All Connects

```text
┌──────────────────────────────────────────────────────────────┐
│                     Docker Network                           │
│                                                              │
│  ┌─────────────────────────┐   ┌──────────────────────────┐  │
│  │ llama                   │   │ rag                      │  │
│  │ (official Docker image) │   │ (your Python app)        │  │
│  │                         │   │                          │  │
│  │ ghcr.io/ggml-org/       │   │ Streamlit UI             │  │
│  │   llama.cpp:server      │◄──│ rag_engine.py            │  │
│  │                         │   │                          │  │
│  │ /models/your_model.gguf │   │ Calls:                   │  │
│  │ (mounted from host)     │   │ http://llama:8080/v1/... │  │
│  │                         │   │ ▲                        │  │
│  │ Port 8080               │   │ │ "llama" resolves to    │  │
│  └──────────┬──────────────┘   │ │ the other container    │  │
│             │                  └─┼────────────────────────┘  │
│             │                    │                           │
└─────────────┼────────────────────┼───────────────────────────┘
              │                    │
         Host:8080             Host:8501
              │                    │
        You can also         You open this
        test directly        in your browser
```
1. **Ingest** — Your documents are split into chunks and converted into
   numerical vectors (embeddings) using a local sentence-transformer model.
   These are stored in a FAISS vector database on your disk.

2. **Retrieve** — When you ask a question, the same embedding model converts
   your question into a vector and finds the most similar chunks in the database.

3. **Generate** — The retrieved chunks are sent as context to a local LLM
   (running via llama.cpp), which generates a grounded answer.

---

## Features

- 📄 **Multi-format support** — Ingest PDF, TXT, and Markdown files
- 🔒 **Fully local** — No API keys, no cloud, no data sharing
- 🖥️ **GPU accelerated** — NVIDIA CUDA support via official llama.cpp image
- 🐳 **Docker-based** — One command setup, works on any machine with Docker
- 📤 **In-app upload** — Drag and drop files directly in the browser UI
- 🗑️ **File management** — View and delete documents from the UI
- 💾 **Persistent storage** — Documents and vector DB survive container restarts

---

## Requirements

### Hardware

| Component | Minimum         | Recommended          |
|-----------|-----------------|----------------------|
| RAM       | 8 GB            | 16 GB+               |
| VRAM      | None (CPU mode) | 6 GB+ (NVIDIA GPU)   |
| Storage   | 10 GB free      | 20 GB+ free          |
| CPU       | 4 cores         | 8 cores+             |

### Software

| Software            | Version   | Download |
|---------------------|-----------|----------|
| Docker Desktop / Engine | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Docker Compose      | V2+       | Included with Docker Desktop |
| NVIDIA Driver (GPU) | 550+      | [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) *(Windows users)* |
| NVIDIA Container Toolkit | Latest | [Guide below](#gpu-setup-recommended) |

> **WSL Users (Windows):** Install the NVIDIA driver on **Windows**,
> not inside WSL. WSL shares the Windows driver automatically.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/arror403/rag-knowledge-base.git
cd rag-knowledge-base
```

### 2. Add Your Model

Download a GGUF model and place it in the `/models` folder.

> Not sure which model to use? See [Model Recommendations](#model-recommendations).

```
Example: Download  gemma-4-E4B-it-Q4_K_M.gguf (4.98 GB)
from https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=true
then place it in the models folder.
```

### 3. Update the Model Filename

Edit `docker-compose.yml` and change:

```yaml
# From
    command: >
      -m /models/gemma-4-E4B-it-Q4_K_M.gguf

# to:
    command: >
      -m /models/{your_model}.gguf
```

### 4. Launch

```bash
chmod +x start.sh
./start.sh
```
*The command chmod (change mode) is to change permission of start.sh being able to execute.*

First run takes longer because Docker pulls the llama.cpp image (~2-3 GB).

### 5. Open in browser

Open **localhost:8501** in your browser.

You can also open **localhost:8080** to use the WebUI of llama.cpp. It's like using ChatGPT, but run on your device locally.

---
### **How to setup Docker in WSL?**

You can follow the [official guide](https://docs.docker.com/engine/install/ubuntu/).

---

## GPU Setup (Recommended)

Skip this section if you want to run on CPU only.
CPU mode works but is significantly slower.

### Step 1: Verify Your GPU Is Visible

```bash
nvidia-smi
```
You should see something like:
```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 555.xx       Driver Version: 555.xx       CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8     5W /  80W |    0MiB /  6144MiB   |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

If this fails, update your NVIDIA driver on Windows and restart WSL.

### Step 2: Install NVIDIA Container Toolkit

**It's recommended to follow the [Official Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**

1. Install the prerequisites for the instructions below:
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
 ca-certificates \
 curl \
 gnupg2
```
2. Configure the production repository:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
3. Update the packages list from the repository:
```bash
sudo apt-get update
```
4. Install the NVIDIA Container Toolkit packages:
```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```
### Step 3: Configuring Docker

1. Configure the container runtime by using the nvidia-ctk command:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```
The nvidia-ctk command modifies the /etc/docker/daemon.json file on the host. The file is updated so that Docker can use the NVIDIA Container Runtime.

2. Restart the Docker daemon:
```bash
sudo systemctl restart docker
```
### Step 4: Check docker-compose.yml

```yaml
# CPU:
image: ghcr.io/ggml-org/llama.cpp:server

# GPU (CUDA):
image: ghcr.io/ggml-org/llama.cpp:server-cuda13
```
Here is the official Docker documentation of llama.cpp:
https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md


---

## Usage Guide

### Upload Documents

1. Open the sidebar → **Documents** section
2. Drag and drop files or click to browse
3. Supported: `.pdf`, `.txt`, `.md`
4. Click **Save Uploaded Files**

### Ingest Documents

After uploading, click **📥 Ingest** in the sidebar.
This processes all documents and builds the vector database.
You only need to do this when documents change.

> Ingestion time depends on document size.
> Expect ~30 seconds per 100 pages.

### Ask Questions

Type your question in the main input field.
The app will:
1. Find the 3 most relevant passages from your documents
2. Send these 3 and your question as context to llama-server by API call
3. Display the generated answer

Click **View Retrieved Context** to see which passages were used.

### Manage Documents

In the sidebar under **Documents**, you can see all files currently
in the `data/` folder and delete individual files using the 🗑️ button.
After deleting files, click **Ingest** again to rebuild the index.


---

## Model Recommendations

Models must be in **GGUF format**.
Download from [Hugging Face](https://huggingface.co/models?library=gguf).


> **Quantization guide:**
> `Q4_K_M` is the sweet spot — good quality, reasonable size.
> `Q3_K_M` is smaller but lower quality.
> `Q8_0` is higher quality but much larger.

---

## Project Structure

```text
rag-knowledge-base/
├── .streamlit/
│   └── config.toml         # Streamlit settings
├── data/                   # Your documents live here, persists between container restarts
│
├── embedding_model/
│   └── all-MiniLM-L6-v2/
│       ├── config.json
│       ├── model.safetensors
│       └── ...
|                           
├── models/
│   └── your_model.gguf     # GGUF model files, not included in repo (too large)
│                           
├── vector_db/              # FAISS index — auto-created on first ingest, persists between container restarts
│                            
│
├── main.py                 # Streamlit web interface
├── rag_engine.py           # Core logic: ingestion, retrieval, LLM calls
├── requirements.txt        # Python dependencies
│
├── Dockerfile              # Container definition for the RAG app
├── docker-compose.yml      # Orchestrates RAG app + llama-server
├── .dockerignore           # Files excluded from Docker builds
│
├── start.sh                # Convenience launch script (Linux/WSL/Mac)
└── README.md               # This file
```

---

## Configuration

All key settings are in `docker-compose.yml`.

### Change the Model

For all config: [LLaMA.cpp HTTP Server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)

```yaml
command: >
  -m /models/YOUR_MODEL_FILENAME_HERE.gguf
  --host 0.0.0.0
  --port 8080
  -ngl 99
  --ctx-size 2048
  -n 512
```

### Key llama-server Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-ngl` | `99` | GPU layers. Set to `0` for CPU only. Reduce if VRAM runs out. |
| `--ctx-size` | `2048` | Context window. Larger = more VRAM. Max recommended: `4096` for 6GB. |
| `-n` | `512` | Max tokens per response. |
| `--threads` | `4` | CPU threads. Increase if running CPU-only. |


### Embedding Model

The embedding model is set in `rag_engine.py` 
embedding_model\all-MiniLM-L6-v2\model.safetensors (~90 MB)
`all-MiniLM-L6-v2` is fast and works well for most use cases.

---

## Troubleshooting

### App does not start

```bash
# Check all containers are running
docker compose ps

# Check logs for errors
docker compose logs rag
docker compose logs llama
```

### LLM server not responding

```bash
# Check if llama-server is healthy
curl http://localhost:8080/health
# Expected: {"status":"ok"}

# Use the Probe Server button in the sidebar
# to auto-detect the correct endpoint
```

### GPU not being used

```bash
# Check GPU visibility
nvidia-smi
```


### Out of VRAM

```bash
# Reduce GPU layers in docker-compose.yml
# Start at half your model's total layers
-ngl 20       # instead of -ngl 99

# Or reduce context size
--ctx-size 1024   # instead of 2048

# Or use a smaller/more quantized model
# Q3_K_M instead of Q4_K_M
```

### Slow responses on CPU

This is expected. Options:
- Enable GPU (see [GPU Setup](#gpu-setup-recommended))
- Use a smaller model (1B or 3B instead of 7B)
- Reduce `-n 512` to `-n 256` for shorter responses


### Ingest fails with "No documents found"

- Make sure you clicked **Save Uploaded Files** before clicking **Ingest**
- Check the `data/` folder actually contains your files
- Verify file extensions are `.pdf`, `.txt`, or `.md` (lowercase)

### Rebuilding after code changes

```bash
docker compose up --build -d
```

### Full reset

```bash
# Stop and remove containers
docker compose down

# Remove vector database (forces re-ingestion)
rm -rf vector_db/

# Start fresh
docker compose up --build -d
```


## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Local LLM inference
- [LangChain](https://github.com/langchain-ai/langchain) — Document processing pipeline
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Streamlit](https://streamlit.io) — Web interface
- [Sentence Transformers](https://www.sbert.net) — Document embeddings

---

<p align="center">
  Built for local use · No cloud · No tracking · Your data stays yours
</p>
