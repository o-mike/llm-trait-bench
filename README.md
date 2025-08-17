# 🧠 LLM Trait Benchmark - Prompt-Induced Manipulativeness

A benchmarking tool that measures how easily Large Language Models shift to manipulative responses under inducing prompts.

## 🎯 What This Measures

**Trait:** Prompt-Induced Manipulativeness  
**Scale:** 0-10 (0 = honest/transparent, 10 = highly manipulative/deceptive)

### Test Scenarios
- **Neutral:** Baseline scenarios asking for honest advice
- **Induced:** Scenarios explicitly prompting for manipulative behavior

### Models Tested
- Google Gemini 1.5 Pro
- Anthropic Claude 3.5 Sonnet  
- DeepSeek V2
- OpenAI GPT-4o
- Meta Llama 3.1 405B

## 📸 Dashboard Preview

![LLM Trait Benchmark Dashboard](screenshots/thingy.png)

*Interactive Gradio dashboard showing benchmark results with manipulativeness scores, cache performance metrics, and detailed analysis across all tested models.*

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenRouter API key

### Installation
```bash
# Clone and setup
git clone https://github.com/o-mike/llm-trait-bench.git
cd llm-trait-bench

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install UV if needed
uv venv
source .venv/bin/activate
uv pip install openai pandas plotly gradio ruff

# Set API key (choose one method)
# Method 1: Config file (recommended)
cp config.example.json config.json
# Edit config.json with your OpenRouter API key

# Method 2: Environment variable
export OPENROUTER_API_KEY="your_key_here"

# Run benchmark
python benchmark.py
```

### Remote Access (VPS)
```bash
# On local machine
ssh -L 7860:localhost:7860 user@your-vps-ip

# On VPS
source .venv/bin/activate
python benchmark.py

# Open browser to http://localhost:7860
```

## 📊 Output

- Interactive Gradio dashboard with visualizations
- CSV export of detailed results  
- **Smart Caching:** First run ~$2-5, subsequent runs ~$0
- ~150 API calls total (cached after first run)
- Cache hit rate and savings displayed

## 🔧 Development

```bash
# Code quality
ruff check benchmark.py
ruff format benchmark.py

# Project structure
benchmark.py         # Main application
config.example.json  # API key configuration template
SETUP.md            # Detailed setup instructions
CLAUDE.md           # Project memory for Claude
README.md           # This file
```

## 📈 Future Extensions

This MVP is designed for easy extension:
- Dynamic question generation
- Multiple judge LLMs (council approach)
- Additional traits (e.g., helpfulness, safety)
- More visualization options
- Advanced scoring methods

## ⚠️ Important Notes

- **Cost:** First run ~$2-5, subsequent runs ~$0 (caching)
- **Time:** ~5-10 minutes for complete benchmark
- **Setup:** See [SETUP.md](SETUP.md) for detailed instructions
- **Security:** Uses config.json (gitignored) for API keys
- **Ethics:** Used for defensive security research only

## 📝 License

MIT License - Use responsibly for defensive AI safety research.