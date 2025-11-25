# pplyz（日本語）

GitHub: https://github.com/masaki39/pplyz

English → [README.md](README.md)

## 必要なもの

- [uv](https://github.com/astral-sh/uv)
  - macOS/Linux: `brew install uv` または `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `scoop install uv`
- LiteLLM に対応した API キー（OpenAI / Gemini / Anthropic / Groq など）

uvx が Python ランタイムを自動取得するため、uv を入れれば他の準備は不要です。

## かんたん実行（uvx）

```bash
uvx pplyz \
  data/sample.csv \
  --input question,answer \
  --output 'score:int,notes:str'
```

- `--preview` で一部だけプレビュー（行数は `[pplyz].preview_rows` で設定）。
- `--list` は同梱テンプレートを表示して終了。
- `--model provider/name` で LiteLLM モデルを上書き（例: `groq/llama-3.1-8b-instant`）。

_pplyz は入力 CSV をそのまま上書きします。元データを残したい場合は事前にコピーしてください。_

詳細なフラグは `uvx pplyz --help` を参照。

## 主なオプション

| フラグ | 説明 | 必須 |
| --- | --- | --- |
| `INPUT`（位置引数） | 入力 CSV。 | はい |
| `-i, --input title,abstract` | LLM に渡す列名。カンマ区切り。 | はい（`[pplyz].default_input` 設定時は不要） |
| `-o, --output 'score:int,notes:str'` | 出力スキーマ。型は `bool/int/float/str/list[...] / dict`。省略時は `str`。 | はい（`[pplyz].default_output` 設定時は不要） |
| `-p, --preview` | 数行だけ処理して結果を表示（行数は `[pplyz].preview_rows` で設定）。 | いいえ |
| `-m, --model provider/name` | LiteLLM モデル指定。初期値 `gemini/gemini-2.5-flash-lite`。 | いいえ |
| `-f, --force` | 既存出力があっても毎回最初から処理し上書き。 | いいえ |
| `-l, --list` | 利用可能テンプレート/モデルを表示して終了。 | いいえ |

## 設定

1. 一度だけユーザー設定ファイルを作成します。

   ```bash
   mkdir -p ~/.config/pplyz
   $EDITOR ~/.config/pplyz/config.toml
   ```

2. 使うプロバイダだけキーを入れます。

   ```toml
   [env]
   OPENAI_API_KEY = "sk-..."
   GROQ_API_KEY = "gsk-..."

   [pplyz]
   default_model = "gpt-4o-mini"
   default_input = "title,abstract"
   default_output = "is_relevant:bool,summary:str"
   ```

3. 読み込み順は「環境変数 → 設定ファイル」。既定は `~/.config/pplyz/config.toml`（Windows は `%APPDATA%\\pplyz\\config.toml`、`XDG_CONFIG_HOME` があればそちらを使用）。別の場所を使いたい場合は `PPLYZ_CONFIG_DIR=/path/to/dir` を設定して、そのディレクトリに `config.toml` を置きます。

ワンポイント: `pplyz data/papers.csv --input title,abstract --output 'summary:str'` のように、最初の引数をそのまま入力 CSV として扱えます。

### 設定値リファレンス

**[pplyz] テーブル**

| キー | 説明 | 既定値 |
| --- | --- | --- |
| `default_model` | `--model` を省略した際の LiteLLM モデル。 | `gemini/gemini-2.5-flash-lite` |
| `default_input` | `-i/--input` を省略したときに使う列リスト。 | 未設定 |
| `default_output` | `-o/--output` を省略したときに使う出力スキーマ。 | 未設定 |
| `preview_rows` | `--preview` 使用時に処理する行数（`PPLYZ_PREVIEW_ROWS` でも指定可）。 | `3` |

### プロバイダ別 API キー

すべて `[env]` テーブル内に記述してください。

| プロバイダ | キー |
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

## モデル一覧

`pplyz/config.py` に含まれる代表的なモデルです（LiteLLM ではさらに多数を利用できます）。

| モデル ID | メモ |
| --- | --- |
| `gemini/gemini-2.5-flash-lite` | 既定。高速 + 低コスト。 |
| `gemini/gemini-1.5-pro` | 高品質 Gemini。 |
| `gpt-4o` | OpenAI フラッグシップ。 |
| `gpt-4o-mini` | コスト重視の GPT-4o Mini。 |
| `claude-3-5-sonnet-20241022` | バランス型 Anthropic。 |
| `claude-3-haiku-20240307` | 高速 Anthropic Haiku。 |
| `groq/llama-3.1-8b-instant` | Groq 上の低レイテンシ。 |
| `mistral/mistral-large-latest` | エンタープライズ向け Mistral。 |
| `cohere/command-r-plus` | ツール利用に強い Cohere。 |
| `replicate/meta/meta-llama-3-8b-instruct` | Replicate 版 Llama 3 8B。 |
| `huggingface/meta-llama/Meta-Llama-3-8B-Instruct` | Hugging Face Endpoint。 |
| `xai/grok-beta` | xAI Grok Beta。 |
| `together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | Together AI 集約 API。 |
| `perplexity/llama-3.1-sonar-small-128k-online` | Web 参照付き Perplexity Sonar。 |
| `deepseek/deepseek-chat` | DeepSeek Chat。 |
| `azure/gpt-4o` | Azure OpenAI 版。 |
| `databricks/mixtral-8x7b-instruct` | Databricks MosaicML エンドポイント。 |
| `sagemaker/meta-textgeneration-llama-3-8b` | AWS SageMaker エンドポイント。 |

`uvx pplyz --list` で手元のバージョンが持つリストを確認できます。

## 例

プレビューで感情分析（設定で `preview_rows = 5` とした例）：

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

同じ CSV に分類結果を書き戻す例：

```bash
uvx pplyz \
  data/articles.csv \
  --input title,abstract \
  --output 'is_relevant:bool,summary:str'
```

Anthropic モデルを指定：

```bash
uvx pplyz \
  data/papers.csv \
  --input title,abstract \
  --output 'findings:str' \
  --model claude-3-5-sonnet-20241022
```

## ヒント

- Boolean 出力列（`--output 'is_relevant:bool'`）は二値分類を安定させます。
- 欲しい JSON スキーマはプロンプトに明記するとパースエラーを減らせます。
- 高コストの CSV を回す前に `--preview` で挙動とモデルを確認しましょう。

## サポート / ライセンス

Issue / PR を歓迎します。MIT ライセンス（`LICENSE` を参照）。
