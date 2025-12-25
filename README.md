# pplyz

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pplyz?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/projects/pplyz)

**p**ython + **p**rompt + ana**lyz**e

Add LLM-generated columns to a CSV with one command.

## Requirements

- [uv](https://github.com/astral-sh/uv) (Recommended)
- At least one LiteLLM-compatible API key (OpenAI, Gemini, Anthropic, etc.)

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
| `-f, --force` | Reprocess all rows (resume is default). | No |
| `--prompt` | Inline prompt text (skips interactive prompt entry). | No |
| `--prompt-file` | Path to a prompt text file (skips interactive prompt entry). | No |

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
OPENAI_API_KEY = "sk-..."
GROQ_API_KEY = "gsk-..."

[pplyz]
default_model = "openai/gpt-4o-mini"
default_input = "title,abstract"
default_output = "relevant:bool,summary:str"
```

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

### [pplyz] table (Default settings)

| key | description | default |
| --- | --- | --- |
| `default_model` | Sets the fallback LiteLLM model when `--model` is omitted. | `gemini/gemini-2.5-flash-lite` |
| `default_input` | Comma-separated columns used when `-i/--input` is omitted. | unset |
| `default_output` | Output schema used when `-o/--output` is omitted. | unset |
| `preview_rows` | Number of rows used when `--preview` is set (can also be overridden via `PPLYZ_PREVIEW_ROWS`). | `3` |

## Supported models

For the latest list of supported models, see the LiteLLM provider docs: https://docs.litellm.ai/docs/providers
