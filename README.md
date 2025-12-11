[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pplyz?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/projects/pplyz)

# pplyz

Source & issues: https://github.com/masaki39/pplyz

Minimal CSV→LLM→CSV transformer powered by LiteLLM and uv.

## Requirements

- [uv](https://github.com/astral-sh/uv)
  - macOS/Linux: `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `scoop install uv`
- At least one LiteLLM-compatible API key (OpenAI, Gemini, Anthropic, Groq, etc.)

`uvx` downloads the right Python runtime automatically, so no global Python is needed once uv is installed.

## Quick run (uvx)

```bash
uvx pplyz \
  data/sample.csv \
  --input question,answer \
  --output 'score:int,notes:str'
```

- `--preview` dry-runs a handful of rows (set `[pplyz].preview_rows` to change how many rows are shown).
- `--model provider/name` overrides the LiteLLM model (e.g., `groq/llama-3.1-8b-instant`).
- Prompts are entered interactively at runtime (history is stored under `~/.config/pplyz/`). For non-interactive runs, provide the prompt when the CLI asks for it.

_pplyz overwrites the input CSV; copy it first if you need to keep the original file._

Run `uvx pplyz --help` for every flag.

## Common options

| Flag | Description | Required |
| --- | --- | --- |
| `INPUT` (positional) | Input CSV path. | Yes |
| `-i, --input title,abstract` | Comma-separated source columns passed to the LLM. | Yes (unless `[pplyz].default_input` is set) |
| `-o, --output 'score:int,notes:str'` | Output column schema. Types: `bool`, `int`, `float`, `str` (missing `:type` defaults to `str`). | Yes (unless `[pplyz].default_output` is set) |
| `-p, --preview` | Process a few rows and show would-be output without writing (row count configured via `[pplyz].preview_rows`). | No |
| `-m, --model provider/name` | LiteLLM model (default `gemini/gemini-2.5-flash-lite`). | No |
| `-f, --force` | Disable resume mode; always recompute rows and overwrite existing output. | No |

## Configuration

1. Create the user config once:

   ```bash
   mkdir -p ~/.config/pplyz
   $EDITOR ~/.config/pplyz/config.toml
   ```

2. Add only the providers you actually use:

   ```toml
   [env]
   OPENAI_API_KEY = "sk-..."
   GROQ_API_KEY = "gsk-..."

   [pplyz]
   default_model = "gpt-4o-mini"
   default_input = "title,abstract"
   default_output = "relevant:bool,summary:str"
   ```

3. At runtime pplyz loads settings in this order: environment variables → config file. The default path is `~/.config/pplyz/config.toml` (or `%APPDATA%\\pplyz\\config.toml` on Windows; if `XDG_CONFIG_HOME` is set, it uses that). To keep configs elsewhere, set `PPLYZ_CONFIG_DIR=/path/to/dir` and place `config.toml` there.

Tip: `pplyz data/papers.csv --input title,abstract --output 'summary:str'` uses the positional `data/papers.csv` as the CSV input.

### Settings reference

**[pplyz] table**

| key | description | default |
| --- | --- | --- |
| `default_model` | Sets the fallback LiteLLM model when `--model` is omitted. | `gemini/gemini-2.5-flash-lite` |
| `default_input` | Comma-separated columns used when `-i/--input` is omitted. | unset |
| `default_output` | Output schema used when `-o/--output` is omitted. | unset |
| `preview_rows` | Number of rows used when `--preview` is set (can also be overridden via `PPLYZ_PREVIEW_ROWS`). | `3` |

### Provider API keys

Set these inside the `[env]` table of your `config.toml`:

| Provider | Keys (checked in order) |
| --- | --- |
| Gemini | `GEMINI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic / Claude | `ANTHROPIC_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Replicate | `REPLICATE_API_KEY` |
| Hugging Face | `HUGGINGFACE_API_KEY` |
| Together AI | `TOGETHERAI_API_KEY`, `TOGETHER_AI_TOKEN` |
| Perplexity | `PERPLEXITY_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| xAI | `XAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_API_KEY` |
| AWS (Bedrock/SageMaker) | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` |

## Supported models

For the latest list of supported models, see the LiteLLM provider docs: https://docs.litellm.ai/docs/providers

## Examples

Sentiment pass with a preview first (`preview_rows` set to 5 in your config):

```toml
[pplyz]
preview_rows = 5
```

```bash
uvx pplyz \
  data/reviews.csv \
  --input review_text \
  --output 'sentiment:str,confidence:float' \
  --preview
```

Boolean classifier that writes back into the same CSV:

```bash
uvx pplyz \
  data/articles.csv \
  --input title,abstract \
  --output 'is_relevant:bool,summary:str'
```

Model override with Anthropic:

```bash
uvx pplyz \
  data/papers.csv \
  --input title,abstract \
  --output 'findings:str' \
  --model claude-3-5-sonnet-20241022
```

## Tips

- Boolean output columns keep binary classifiers deterministic (`true`/`false`).
- Some models do not support JSON mode; pplyz only sends `response_format` to models that advertise support. Explicitly state “return valid JSON only” in your prompt to keep outputs consistent.
- Keep prompts short and explicit about the JSON schema you expect to avoid parsing errors.
- Use `--preview` before long or expensive CSV batches to validate prompts and model choice.
- Resume mode is on by default; rows with existing output columns are skipped. Use `--force` to recompute everything.
- Dynamic (schema-less) mode is not supported; always provide `--output` (or set `[pplyz].default_output`).
- CSV encoding is UTF-8 only; convert input files beforehand if they use another encoding.
