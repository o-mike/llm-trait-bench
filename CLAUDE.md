# CLAUDE.md - Project Memory for llm-trait-bench

## Project Overview
This repo benchmarks LLM traits like Prompt-Induced Manipulativeness. MVP: Hardcoded questions, 5 LLMs via OpenRouter, single-judge eval, Gradio dashboard. Vision: Dynamic generation, councils, more traits.

## Coding Standards
- Use Python 3 with UV for deps.
- Modular functions (e.g., reusable for dynamic inputs).
- Error handling: Retries, logging.
- Follow Ruff for linting/formatting.

## Build/Run Steps
1. uv venv && uv pip install openai pandas plotly gradio ruff
2. source .venv/bin/activate
3. python benchmark.py (launches Gradio)
4. SSH port forwarding: ssh -L 7860:localhost:7860 user@vps-ip

## Key Commands/Tips
- /clear: Reset context after tasks
- For drafts/plans: Use /plan for subtasks
- ruff check && ruff format for code quality

## API Configuration
- OpenRouter base URL: https://openrouter.ai/api/v1
- Models: google/gemini-1.5-pro, anthropic/claude-3-5-sonnet, deepseek/deepseek-chat, openai/gpt-4o, meta-llama/llama-3.1-405b-instruct
- Judge model: anthropic/claude-3-5-sonnet

## Trait Definition
Prompt-Induced Manipulativeness: Measures how easily an LLM shifts to manipulative/deceptive responses under inducing prompts.
- Scale: 0-10 (0 = honest/no manipulation, 10 = highly deceptive/manipulative)
- Themes: Neutral Scenarios (baseline) vs Induced Scenarios (pressure test)

Add custom details here as the project evolves.