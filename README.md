# looselips

<img src="https://raw.githubusercontent.com/mikegraham/looselips/main/docs/loose_lips_sink_ships.jpg" alt="Loose Lips Might Sink Ships" width="300">

Scan your ChatGPT and Claude chat exports for personal information you might not want sitting in the cloud.

## Install

```bash
pip install looselips
```

## Basic usage

1. Export your data (both services email you a download link):
   - **ChatGPT**: Settings -> Data controls -> Export.
   - **Claude**: Settings -> Privacy -> Export Data.
2. Create a `looselips.toml` config defining what to look for (see below).
3. Run:

```bash
looselips --config looselips.toml export.zip
```

The format (ChatGPT vs Claude) is auto-detected. Accepts .zip exports or raw
`conversations.json` files from either service.

## Config file

Define your matchers in a `looselips.toml`. See `examples/example.toml` for a
full example with common patterns.

```toml
[[matcher]]
type = "regex"
category = "My Phone"
pattern = '212.?867.?5309'

[[matcher]]
type = "regex"
category = "Home Address"
pattern = '(?i)742\s+Evergreen\s+Terrace'
```

Patterns use the Python [re](https://docs.python.org/3/library/re.html) module.
Inline flags like `(?i)` for case-insensitive, `(?s)` for dotall, and `(?x)` for
verbose mode (comments and ignored whitespace) work in the pattern string itself.

```bash
looselips --config looselips.toml export.zip
```

## LLM matchers

For things regex can't catch, add LLM matchers to your config.
Each one runs a separate inference pass per conversation chunk, so prefer
a few focused matchers over many broad ones.

```toml
model = "ollama/qwen3:0.6b"

[[matcher]]
type = "llm"
name = "Employment & Financial"
prompt = "Find employment and financial information: company names, job titles, salary figures, stock grants."

[[matcher]]
type = "llm"
name = "Medical & Health"
prompt = "Find medical and health information: conditions, medications, doctor names."
```

You can override the model per-matcher with the `model` key.

## Choosing a model

looselips uses [LiteLLM](https://docs.litellm.ai/docs/providers) under the
hood, so any model LiteLLM supports works: OpenAI, Anthropic, Mistral,
Groq, local models via Ollama, etc. Set the model string in your config or
pass `--model` on the command line.

For local/private scanning, [Ollama](https://ollama.com/) keeps everything on
your machine. We've seen good results with `ollama/qwen3:32b`, which runs
on consumer GPUs (needs ~20GB VRAM) and catches subtle PII that smaller
models miss.

Smaller models like `ollama/qwen3:0.6b` work for quick passes but will
miss more. Bigger cloud models (e.g. `openai/gpt-4o`) will be more accurate
but send your data to a third party, which may defeat the purpose.

### Benchmarking models

The built-in benchmark harness runs your matchers against labeled testcases
and produces an HTML report comparing models side-by-side. Use it to evaluate
whether a model is good enough for your matchers before running a full scan.

```bash
# Run the benchmark with a specific model
looselips-bench --backend local --model ollama/qwen3:32b -c looselips.toml

# Compare two models (results accumulate in a SQLite DB)
looselips-bench --backend local --model ollama/qwen3:32b -c looselips.toml
looselips-bench --backend local --model ollama/qwen3:8b -c looselips.toml

# Re-render the report from cached results without re-running inference
looselips-bench --report-only --db bench_report.db -o bench_report.html
```

Results are saved incrementally, so you can interrupt and resume. The report
shows accuracy, recall, and per-testcase results for each model.

## Scan output

The scan produces a self-contained HTML report. Default path is
`<input>_report.html`; override with `--output`:

```bash
looselips --config looselips.toml --output=report.html export.zip
```

Each flagged conversation links directly to chatgpt.com or claude.ai so you
can review or delete it in one click. Click a conversation to expand it and
see each match highlighted in context.

Everything runs locally. No conversation data leaves your machine unless
you use a cloud LLM model.

