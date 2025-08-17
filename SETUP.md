# 🚀 Complete Setup Guide

## 1. Getting OpenRouter API Key

### Step 1: Sign up for OpenRouter
1. Go to [https://openrouter.ai/](https://openrouter.ai/)
2. Click **"Sign up"** and create an account
3. Verify your email address

### Step 2: Get API Key
1. After login, go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Click **"Create Key"**
3. Give it a name (e.g., "LLM Trait Benchmark")
4. Copy the generated API key (starts with `sk-or-v1-...`)

### Step 3: Add Credits (Optional)
- OpenRouter has free credits for new users
- For extensive testing, add credits at [https://openrouter.ai/credits](https://openrouter.ai/credits)
- ~$5-10 should be sufficient for multiple benchmark runs

## 2. Local Setup

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
# Clone repository
git clone https://github.com/o-mike/llm-trait-bench.git
cd llm-trait-bench

# Install UV package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # Reload shell

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install openai pandas plotly gradio ruff
```

### API Key Configuration (Choose One Method)

#### Method 1: Config File (Recommended - Won't leak to Git)
```bash
# Copy example config
cp config.example.json config.json

# Edit config.json with your API key
# Replace "your_openrouter_api_key_here" with your actual key
nano config.json
```

#### Method 2: Environment Variable
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# For persistence, add to ~/.bashrc or ~/.zshrc:
echo 'export OPENROUTER_API_KEY="sk-or-v1-your-key-here"' >> ~/.bashrc
```

### Run Benchmark
```bash
python benchmark.py
```

## 3. Remote VPS Setup (SSH Key Authentication)

### Prerequisites on VPS
- Python 3.8+
- SSH key authentication enabled
- Username login disabled (for security)

### SSH Key Setup (if not already done)
```bash
# On your local machine, generate SSH key pair (if needed)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key to VPS
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@your-vps-ip

# Test connection
ssh user@your-vps-ip
```

### VPS Installation
```bash
# SSH into VPS
ssh user@your-vps-ip

# Clone and setup (same as local)
git clone https://github.com/o-mike/llm-trait-bench.git
cd llm-trait-bench
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install openai pandas plotly gradio ruff

# Configure API key
cp config.example.json config.json
nano config.json  # Add your OpenRouter API key
```

### Access Dashboard from Local Machine

#### Method 1: SSH Port Forwarding (Recommended)
```bash
# On local machine, forward port 7860
ssh -L 7860:localhost:7860 user@your-vps-ip

# Keep this terminal open, then in another terminal on VPS:
cd llm-trait-bench
source .venv/bin/activate
python benchmark.py

# On local machine, open browser to:
# http://localhost:7860
```

#### Method 2: Custom Port Forwarding
```bash
# If port 7860 is busy locally, use different port:
ssh -L 8080:localhost:7860 user@your-vps-ip

# Then access: http://localhost:8080
```

### Firewall Considerations
If you want direct access (not recommended for security):
```bash
# On VPS (Ubuntu/Debian)
sudo ufw allow 7860
python benchmark.py  # Will be accessible at http://your-vps-ip:7860
```

## 4. Cost Optimization with Caching

### How Caching Works
- **First run:** ~150 API calls, costs $2-5
- **Subsequent runs:** 0 API calls for same questions, ~$0
- Cache stored in `responses_cache.json` (gitignored)
- Each unique model+prompt combination is cached

### Cache Management
```bash
# View cache stats
python -c "
import json
from pathlib import Path
if Path('responses_cache.json').exists():
    with open('responses_cache.json') as f:
        cache = json.load(f)
    print(f'Cached responses: {len(cache)}')
else:
    print('No cache file found')
"

# Clear cache (will cost money on next run)
rm responses_cache.json

# Disable caching (in config.json)
{
  "cache_enabled": false,
  ...
}
```

### Expected Savings
- **Run 1:** $2-5 (all API calls)
- **Run 2:** ~$0 (100% cache hits)
- **Partial changes:** Only new model/question combinations cost money

## 5. Security Best Practices

### API Key Security
- ✅ **DO:** Use `config.json` (gitignored)
- ✅ **DO:** Use environment variables
- ❌ **DON'T:** Hardcode keys in source code
- ❌ **DON'T:** Commit `config.json` to Git

### VPS Security
- ✅ **DO:** Use SSH keys only
- ✅ **DO:** Disable password authentication
- ✅ **DO:** Use SSH port forwarding
- ❌ **DON'T:** Open benchmark ports to public internet

### File Permissions
```bash
# Secure your config file
chmod 600 config.json

# Verify .gitignore prevents commits
git status  # Should not show config.json
```

## 6. Troubleshooting

### Common Issues

#### "API Key Required" Error
```bash
# Check config exists and is valid
cat config.json

# Check environment variable
echo $OPENROUTER_API_KEY

# Verify API key format (should start with sk-or-v1-)
```

#### SSH Connection Issues
```bash
# Test SSH connection
ssh -v user@your-vps-ip  # Verbose output for debugging

# Test key authentication
ssh -i ~/.ssh/id_ed25519 user@your-vps-ip
```

#### Port Forwarding Issues
```bash
# Check if port is already in use locally
lsof -i :7860

# Use different port
ssh -L 8080:localhost:7860 user@your-vps-ip
```

#### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv pip install --force-reinstall openai pandas plotly gradio ruff
```

### Getting Help
- Check logs in terminal output
- Verify all prerequisites are installed
- Test with minimal example first
- Check OpenRouter status: [https://status.openrouter.ai/](https://status.openrouter.ai/)

## 7. Example Complete Workflow

### Local Development
```bash
git clone https://github.com/o-mike/llm-trait-bench.git
cd llm-trait-bench
uv venv && source .venv/bin/activate
uv pip install openai pandas plotly gradio ruff
cp config.example.json config.json
# Edit config.json with your API key
python benchmark.py
# Open http://localhost:7860
```

### Production on VPS
```bash
# Local machine - setup SSH forwarding
ssh -L 7860:localhost:7860 user@your-vps-ip

# VPS - run benchmark
cd llm-trait-bench
source .venv/bin/activate
python benchmark.py

# Local machine - access dashboard
# http://localhost:7860
```