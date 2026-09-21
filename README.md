# pplyz

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pplyz?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/projects/pplyz)

**p**ython + **p**rompt + ana**lyz**e

Add LLM-generated columns to a CSV with one command.

> [!important]
> **0.2.0:** the default model is now `openrouter/google/gemini-2.5-flash-lite`, which needs
> `OPENROUTER_API_KEY`. To keep using another provider, pass `--model`
> (e.g. `gemini/gemini-2.5-flash-lite`) or set `default_model` in `config.toml`.

## Requirements

- [uv](https://github.com/astral-sh/uv) (Recommended)
- An [OpenRouter](https://openrouter.ai/) API key (recommended — one key covers the default model and `pplyz judge`), any other LiteLLM-compatible API key (OpenAI, Gemini, Anthropic, etc.), **or** a local [Ollama](https://ollama.com/) server (no API key needed)

uv is the easiest way to run the CLI.

## Usage

### Install

```bash
uv tool install pplyz
```

### Example

```bash
pplyz test.csv \
  --input title,abstract \
  --output "relevant:bool,summary:str" \
  --model openai/gpt-4o-mini
```

This command sends the `title` and `abstract` columns to the LLM,
adds `relevant` and `summary` columns to `test.csv`,
and uses the `openai/gpt-4o-mini` model.

> [!note]
> The CLI prompts you for a task description before processing unless `--prompt` or `--prompt-file` is provided.
> Output is written back to the input CSV file (overwrite).

### Command arguments

 Use `-h` or `--help` to list arguments.

```bash
pplyz -h
```

| Flag | Description | Required |
| --- | --- | --- |
| `INPUT` (positional) | Input CSV path. | Yes |
| `-i, --input` | Comma-separated input column names (e.g., `title,abstract`). | Yes (unless default is set) |
| `-o, --output` | Output schema (e.g., `score:int,notes:str`). Types: `bool`, `int`, `float`, `str`. | Yes (unless default is set) |
| `-p, --preview` | Process a few rows and show would-be output without writing. | No |
| `-m, --model` | LiteLLM model name. | No |
| `--api-base` | Base URL for the LLM API (for Ollama or other OpenAI-compatible local servers). | No |
| `-f, --force` | Reprocess all rows (resume is default). | No |
| `--prompt` | Inline prompt text (skips interactive prompt entry). | No |
| `--prompt-file` | Path to a prompt text file (skips interactive prompt entry). | No |

## Yes/no, choice and scale decisions with Jev (`pplyz judge`)

`pplyz judge` asks [Jev](https://openrouter.ai/typesafe) — TypeSafe's decision-only model —
the same questions about every row. Jev never writes text, so answers are always one of
the allowed values, and it is much faster and cheaper than a chat LLM (rows run 8 at a
time by default).

It uses your `OPENROUTER_API_KEY` (create one at <https://openrouter.ai/keys>; see
[Keeping API keys out of config files](#keeping-api-keys-out-of-config-files-macos-keychain)).
Jev is served from OpenRouter's *alpha* decisions endpoint, so details may still change.

```bash
pplyz judge papers.csv
```

The first run walks you through the questions — no special syntax needed:

```
Columns in the CSV (3):
  title, abstract, year
Which columns should Jev read? (comma-separated, Tab to complete): title,abstract

Question 1: What should be judged for each row?
  Jev reads the selected columns of each row and answers this.
  e.g. "Is this paper about cancer research?"
> Is this paper about cancer research?
How should it be answered?
  1) Yes / No (saved as the probability of yes, 0-1)
  2) Pick one of several options
  3) Rate on a scale (lowest -> highest)
Choose [1]: 1
Output column name [paper_about_cancer_research]: is_cancer
Add another question? [y/N]: n

Questions:
  1. is_cancer (yes/no probability): Is this paper about cancer research?
Use these questions? (n = start over) [Y/n]:

→ Saved questions to papers.judge.toml (edit it or rerun to reuse)
```

- When picking columns, press Tab to complete names — typing any part of a name finds it,
  which helps with wide CSVs.
- pplyz then previews a few rows and asks before processing the whole file.
- **Results are added as new columns to the input CSV (in place).** Pass `-o out.csv` to
  keep the input unchanged. Existing cells are kept as text (e.g. `001` stays `001`).
- Later runs reuse `papers.judge.toml` and skip rows that already have answers. If you
  edit the questions, pplyz asks whether to recompute those rows (or pass `-f`).
- For a scale, list levels from lowest to highest (e.g. `low|medium|high`).
- Use one questions file per CSV: it remembers which questions produced the answers
  in that CSV.

| Answer type | Column value |
| --- | --- |
| Yes / No | probability of "yes" (0–1), e.g. `0.93` |
| Pick one of several options | the chosen option |
| Rate on a scale | the most likely level |

Yes/no columns hold the probability itself so you can pick any threshold later
(e.g. `df[df.is_cancer >= 0.8]`). `--with-probability` adds `<column>_p` — the
probability of the chosen option or level — for choice and scale questions.

| Flag | Description |
| --- | --- |
| `-i, --input` | Columns shown to Jev (asked interactively if omitted). |
| `-o, --output` | Write results to another CSV instead of updating the input in place (recomputing rebuilds it from the input). |
| `-q, --questions` | Questions file (default: `<csv name>.judge.toml` next to the CSV). |
| `-p, --preview` | Show results for a few rows without writing. |
| `-f, --force` | Recompute every row (resume is default). |
| `-w, --workers` | Rows processed in parallel (default: `8`). |
| `-m, --model` | Jev model on OpenRouter (default: `typesafe/jev-1.13`, or `jev_model` in config). |
| `--with-probability` | Also write `<column>_p` for choice/scale questions. |
| `--endpoint` | Decisions endpoint URL (default: OpenRouter's alpha endpoint). |

## Configuration

1. Create the user config once:

```bash
mkdir -p ~/.config/pplyz
$EDITOR ~/.config/pplyz/config.toml
```

On Windows, use `%APPDATA%\\pplyz\\config.toml`.

2. Add only the providers you actually use:

```toml
[env]
OPENROUTER_API_KEY = "keychain:openrouter"   # read from the macOS Keychain
OPENAI_API_KEY = "sk-..."                    # or plain text (not recommended)

[pplyz]
default_model = "openrouter/openai/gpt-4o-mini"
default_input = "title,abstract"
default_output = "relevant:bool,summary:str"
```

### Keeping API keys out of config files (macOS Keychain)

Store the key once — the input is hidden and never lands in shell history:

```bash
pplyz auth set openrouter
```

Then reference it instead of pasting the key:

```toml
[env]
OPENROUTER_API_KEY = "keychain:openrouter"
```

`keychain:<name>` reads the generic password with service `pplyz` and account `<name>`;
`keychain:<service>/<account>` reads any other Keychain item. The same syntax works in
environment variables. Keys are only read for the provider actually used.

### Settings priority

pplyz loads settings in this order (earlier wins):

1. Existing environment variables
2. `pplyz.local.toml` in the project root (optional)
3. User config: `~/.config/pplyz/config.toml`
   (or `%APPDATA%\\pplyz\\config.toml` on Windows; if `XDG_CONFIG_HOME` is set, it uses that)

To keep configs elsewhere, set `PPLYZ_CONFIG_DIR=/path/to/dir` and place `config.toml` there.

### [env] table (API keys)

Set these inside the `[env]` table of your `config.toml`
(or export them as environment variables):

| Provider | Keys (checked in order) |
| --- | --- |
| Gemini | `GEMINI_API_KEY` |
| OpenRouter (also `pplyz judge`) | `OPENROUTER_API_KEY` |
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
| Ollama (local) | *none* — set `OLLAMA_API_BASE` if not `http://localhost:11434` |

### [pplyz] table (Default settings)

| key | description | default |
| --- | --- | --- |
| `default_model` | Sets the fallback LiteLLM model when `--model` is omitted. | `openrouter/google/gemini-2.5-flash-lite` |
| `jev_model` | Jev model used by `pplyz judge` (same as `PPLYZ_JEV_MODEL`). | `typesafe/jev-1.13` |
| `default_input` | Comma-separated columns used when `-i/--input` is omitted. | unset |
| `default_output` | Output schema used when `-o/--output` is omitted. | unset |
| `default_api_base` | Base URL passed to the provider (same as `--api-base` / `PPLYZ_API_BASE`). | unset |
| `preview_rows` | Number of rows used when `--preview` is set (can also be overridden via `PPLYZ_PREVIEW_ROWS`). | `3` |

## Local models with Ollama

pplyz can run entirely offline against a local [Ollama](https://ollama.com/) server — no API key required.

```bash
# 1. Start Ollama and pull a model
ollama serve            # usually already running
ollama pull llama3.1

# 2. Run pplyz against it (use the ollama_chat/ prefix)
pplyz test.csv \
  --input title,abstract \
  --output "relevant:bool,summary:str" \
  --model ollama_chat/llama3.1
```

Notes:

- Use the `ollama_chat/` prefix (chat API); it handles system prompts better than `ollama/`.
- The default endpoint is `http://localhost:11434`. Override it with `--api-base`,
  the `OLLAMA_API_BASE` / `PPLYZ_API_BASE` env var, or `default_api_base` in `config.toml`.
- JSON output is enforced via Ollama's JSON mode. Local models follow schemas less
  reliably than hosted ones — prefer `bool` fields for decisions and test with `--preview`.
- `--api-base` also works for other OpenAI-compatible local servers (LM Studio, vLLM, …).

## Supported models

For the latest list of supported models, see the LiteLLM provider docs: https://docs.litellm.ai/docs/providers
