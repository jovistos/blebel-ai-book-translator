# Model & Provider Configuration

`book-translator` features a flexible, multi-provider configuration system. Every AI agent or stage in the pipeline can be bound to a different model or provider depending on cost, quality, and latency requirements.

Configuration is centralized in [book_translator.toml](file:///home/jovis/Documents/WORK/BleBelTranslateBook/book_translator.toml) and can be overridden via environment variables or CLI flags.

---

## 1. Supported Providers

| Provider | Type | Default Model | Base URL | Setup / Key Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **`deepseek`** (Default) | Cloud | `deepseek-flash` | `https://api.deepseek.com/v1` | Set `DEEPSEEK_API_KEY` in `.env` |
| **`lmstudio`** | Local / Offline | Auto-detected | `http://127.0.0.1:1234/v1` | **No key needed**. Run LM Studio locally |
| **`openai`** | Cloud | `gpt-4.1-mini` | `https://api.openai.com/v1` | Set `OPENAI_API_KEY` in `.env` |
| **`gemini`** | Cloud | `gemini-2.5-flash` | `https://generativelanguage.googleapis.com/v1beta/openai` | Set `GEMINI_API_KEY` in `.env` |
| **`claude`** | Cloud | `anthropic/claude-3.5-haiku` | `https://openrouter.ai/api/v1` | Set `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` |
| **`ollama`** | Local / Offline | `llama3.1` | `http://127.0.0.1:11434/v1` | Local Ollama instance |

> [!TIP]
> **Zero Configuration for LM Studio:** When using LM Studio (`provider = "lmstudio"`), you do **not** need to specify a model name! The pipeline queries LM Studio's `/v1/models` endpoint and automatically uses whichever model you currently have loaded.

---

## 2. Configuration Hierarchy & Precedence

When a stage requests model settings, the resolver walks an explicit precedence ladder (first match wins):

```mermaid
flowchart TD
    P1["1. CLI Specific Overrides<br><code>--llm &lt;stage&gt;.&lt;field&gt;=VALUE</code>"] --> P2["2. Stage CLI Flag<br><code>--translate-model</code>, <code>--model</code>"]
    P2 --> P3["3. Stage Environment Variable<br><code>SPLITTER_&lt;STAGE&gt;_MODEL</code>, <code>TRANSLATION_*</code>"]
    P3 --> P4["4. Global Pipeline CLI Flags<br><code>--provider</code>, <code>--model</code>, <code>--base-url</code>"]
    P4 --> P5["5. TOML Stage Table<br><code>[llm.&lt;stage&gt;]</code> in book_translator.toml"]
    P5 --> P6["6. TOML Global Table<br><code>[llm]</code> in book_translator.toml"]
    P6 --> P7["7. Provider Built-in Default"]
```

---

## 3. Configuration File: `book_translator.toml`

A standard `book_translator.toml` looks like this:

```toml
# Global defaults inherited by all stages
[llm]
provider = "deepseek"
model = "deepseek-flash"
base_url = "https://api.deepseek.com/v1"
temperature = 0.0
timeout = 180.0
max_retries = 3

# Granular per-stage overrides:
# Available stage sections: [llm.split], [llm.preprocess], [llm.dictionary],
#                          [llm.matter], [llm.translate], [llm.postprocess]

[llm.split]
model = "deepseek-flash"

[llm.preprocess]
model = "deepseek-flash"

[llm.dictionary]
model = "deepseek-flash"

# Stage 7: Matter localization and multimodal matter image recreation
# Note: Recreating images with text (covers, title plates) requires a multimodal model!
[llm.matter]
model = "deepseek-flash"
# For multimodal matter image generation/recreation, set GEMINI_API_KEY in .env,
# or set provider = "gemini" and model = "gemini-3.1-flash-image" here.

# Translate stage using local LM Studio for zero-cost offline body translation:
[llm.translate]
provider = "lmstudio"
base_url = "http://127.0.0.1:1234/v1"
# model is omitted: auto-detects loaded LM Studio model

[llm.postprocess]
model = "deepseek-flash"
```

> [!IMPORTANT]
> **Multimodal Model Required for Matter Image Translation (Stage 7):**  
> Stage 7 not only formats front and back matter text, but also includes an end-to-end **multimodal image translation and recreation** phase. When images containing text (such as original book covers, illustrated title plates, or publisher frontispieces) are detected in matter chunks, they are sent to a multimodal vision + image generation model. The model reconstructs the full illustration from scratch with translated typography seamlessly integrated into the original artwork style (no bounding boxes or overlays).  
> 
> A **multimodal model with image generation capabilities is required for this step** (default: `gemini-3.1-flash-image` via Google Gemini). Ensure `GEMINI_API_KEY` is defined in your `.env` file, or configure `[llm.matter]` to use a multimodal provider and model (e.g. `provider = "gemini"`). If a text-only model is specified for matter text translation, the stage automatically resolves the multimodal image model from Gemini if `GEMINI_API_KEY` is present. Pure illustrations without text are recognized and kept untouched without calling the image generation model.

---

## 4. Environment Variables & `.env`

Credentials are **never** stored in TOML files or git commits. Store your API keys in a `.env` file at the repository root:

```bash
# .env file
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxx
```

### Environment Variable Overrides
You can also override stage settings globally or per stage via environment variables:
* `SPLITTER_PROVIDER=openai`
* `SPLITTER_MODEL=gpt-4.1-mini`
* `SPLITTER_TRANSLATE_PROVIDER=lmstudio`
* `BOOK_TRANSLATOR_CONFIG=/path/to/custom_config.toml`

---

## 5. Runtime CLI Overrides

You can override any parameter on the fly without modifying files:

```bash
# Override the translation stage provider on the CLI
book-translator-pipeline "Book.md" --llm translate.provider=lmstudio

# Run split with Google Gemini and translation with DeepSeek
book-translator-pipeline "Book.md" \
  --llm split.provider=gemini \
  --llm split.model=gemini-2.5-flash \
  --llm translate.provider=deepseek

# Point to an alternate configuration profile
book-translator-pipeline "Book.md" --llm-config config/cheap-profile.toml
```
