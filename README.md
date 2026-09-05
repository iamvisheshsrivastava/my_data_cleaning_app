# Smart CSV Toolkit — LLM‑Assisted CSV Cleaning & Metadata Inference

**Live demo → https://smart-csv-toolkit.onrender.com/** (free-tier Render instance, so the first load after idle can take 30-60s to spin up)

A Streamlit app for cleaning messy CSVs, figuring out what's actually in each column, merging multiple files, and generating plots without writing pandas by hand every time. I built this mostly to stop rewriting the same `df.dropna()` / dtype-coercion boilerplate for every one-off dataset — an LLM proposes the cleaning code, you read it before it runs, and every action gets logged to SQLite so you can see what happened after the fact.

> **TL;DR**
> • **Tab 1 – CSV Cleaner:** choose built‑in steps from `table_steps.json`, get **LLM suggestions**, optionally apply **LLM‑generated Python code** (shown before execution), and download the cleaned CSV.
> • **Tab 2 – Metadata Inspector:** upload/merge up to 5 CSVs, **infer column types**, preview media/links, and **auto‑generate plots** (with or without LLM help).
> • **Advanced:** interactive **decision‑tree UI** for guided cleaning; optional **custom LLM API**; session/file/event **audit logging** with timestamped CSVs.

---

## Features

* **Guided CSV Cleaning**

  * Configurable steps defined in `table_steps.json` (sliders, selects, text inputs rendered dynamically)
  * LLM proposes **5 non‑redundant** cleaning suggestions (excludes already‑selected steps)
  * Any natural‑language instruction → **executable Python code** that mutates `df` in place
    *(code is displayed before execution for review)*

* **Metadata Inference & Usability**

  * Infers semantic types like: *Categorical, Text, Numerical, Datetime, GPS Coordinates, Email, Phone, Currency, Percentage, Color Code, Image/Video/Document/General URL, Identifier/ID, Null‑heavy, Constant/Low Variance*, etc.
  * Context‑aware visualizations:

    * Categorical → top‑k bar chart
    * Text → word cloud
    * Numerical → box plot + correlation heatmap
    * Datetime → time‑series line plot
    * GPS → map from `lat,lon`
    * Color codes → swatches
    * Email/Phone → frequency bars
    * URLs → previews (images, videos) + **webpage summarization** via LLM for general links

* **Multi‑CSV Merge UI**

  * Upload up to **5** CSVs, configure pairwise joins (keys + type), then merge CSVs with one click

* **Interactive Cleaning Graph**

  * AGraph‑based decision tree per column; **click leaf nodes** to apply LLM‑generated cleaning for the chosen path; executed code is surfaced and actions are tracked

* **Exploration via D‑Tale**

  * One‑click link to open D‑Tale and explore the current DataFrame

* **Audit Logging**

  * Sessions, files, and events logged to SQLite via `DB/log_to_db.py`
  * Uploaded/merged CSVs saved into an **audit folder** with timestamp + session id

* **Optional Custom LLM API**

  * `LLM/config.py` can point the type-inference fallback at your own model endpoint instead of OpenRouter, if you're running something locally (e.g. DeepSeek Coder behind a small FastAPI wrapper)

---

## Repository Layout

```
my_data_cleaning_app/
├─ app.py                      # Streamlit UI (tabs, LLM helpers, decision tree, logging)
├─ pipeline_logic.py           # Executes selected cleaning steps on df
├─ metadata_inference.py       # Column type inference + LLM‑assisted helpers
├─ cleaningDecisionTree.py     # AGraph/PyVis decision tree + click‑to‑clean
├─ table_steps.json            # Declarative config driving the Cleaner UI
├─ DB/
│  ├─ log_to_db.py            # log_session, log_file, log_event (SQLite)
│  └─ auditCSVFiles/          # audit folder for saved CSVs (created at runtime)
├─ LLM/
│  └─ config.py               # custom LLM endpoint (fallback type-inference path only)
├─ .streamlit/                # Streamlit settings
├─ .devcontainer/             # VS Code Dev Container setup
├─ requirements.txt           # Python dependencies
└─ README.md                  # (this file)
```

---

## How It Works (High Level)

1. **Config‑Driven UI** — `table_steps.json` defines sections, step names, descriptions, and typed options. `app.py` renders controls automatically and builds a `steps` list.
2. **Pipeline Execution** — `pipeline_logic.run_pipeline(df, steps)` executes selected steps in order. If an LLM instruction was accepted, its generated code is appended as a step and executed safely within the pipeline wrapper.
3. **LLM Helpers**

   * `call_llm()` uses the OpenRouter API (OpenAI-compatible) with model `z-ai/glm-4.6` to:

     * generate *cleaning suggestions* (`fetch_llm_suggestions`)
     * translate a *natural instruction* → **raw Python code** (`get_cleaning_code_from_llm`)
   * `fallback_infer_type_with_llm()` posts to the endpoint in `LLM.config.API_URL` as a secondary type-inference path when the heuristics are unsure.
4. **Metadata Inference & Visuals** — `metadata_inference.analyze_dataframe(df)` infers types and suggests basic visualizations; URL columns can be summarized via an LLM.
5. **Decision Tree** — `cleaningDecisionTree.render_agraph_tree()` builds a compact action tree (max branching/leaf count). Clicking a **leaf** triggers `custom_cleaning_via_llm()` with contextual instruction; code and results are shown.
6. **Auditability** — All key actions are logged via `log_session`, `log_file`, and `log_event`. CSVs are saved to an audit directory with timestamp and session id.

---

## Quick Start

### 1) Clone & Create Environment

```bash
git clone https://github.com/iamvisheshsrivastava/Smart_CSV_Toolkit.git
cd Smart_CSV_Toolkit

# (recommended) Python 3.10+ virtual env
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Configure LLMs (Optional but recommended)

**OpenRouter API (used by `call_llm`)**

* Create an account at [OpenRouter](https://openrouter.ai/keys) and get an API key.
* Copy [`.env.example`](.env.example) to `.env` and fill in the local values for development.
* Add the API key to your `.streamlit/secrets.toml` file:

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key-here"
```

Alternatively, set the environment variable before running Streamlit:

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "YOUR_KEY"

# macOS/Linux
export OPENROUTER_API_KEY="YOUR_KEY"
```

The model defaults to `z-ai/glm-4.6` and can be overridden with the `OPENROUTER_MODEL` environment variable.

**Custom LLM endpoint (used only as a fallback for type inference, `LLM/config.py`)**

```python
# LLM/config.py
API_URL = os.getenv("CUSTOM_LLM_API_URL", "http://localhost:9000/generate")
```

This was originally meant to power a separate "Custom Trained LLM via API" panel too, but that panel was quietly pointed at `localhost:9000` in production and just failed every time — it's since been switched over to the same OpenRouter client as everything else (`call_llm()`). The only thing that still hits `CUSTOM_LLM_API_URL` is the type-inference fallback in `metadata_inference.py`, and it degrades to `"Text"` if that request fails, so leaving it unset is harmless.

### 3) Audit Folder Path

Uploaded and merged CSVs are saved automatically under the repo-local audit folder:

```text
DB/auditCSVFiles/
```

The app now resolves this path relative to `app.py`, so the default setup is portable across Windows, macOS, and Linux.

### 4) Run the App

```bash
streamlit run app.py
```

### 5) Browser E2E Tests

This repo now includes a browser smoke test for the main Streamlit cleaning flow.

```bash
pip install -r requirements.txt
python -m playwright install chromium
pytest tests/e2e -v
```

The browser test launches Streamlit locally, uploads a sample CSV, enables a built-in cleaning step, and verifies that the cleaned output renders in the UI.

---

## Usage Guide

### Tab 1 — **CSV Cleaner**

1. **Upload CSV** → preview top rows.
2. **Select processing steps** (forms are generated from `table_steps.json`).
3. **Get Smart LLM Suggestions** → returns 5 new, non‑redundant suggestions.
4. **(Optional) Custom instruction** → enter natural language.
5. **Run Cleaning Pipeline**

   * If a custom/selected LLM instruction exists, the app will:

     * Call the LLM to produce **raw Python code** for `df`
     * Show the code (for your review)
     * Append it as a step and run the full pipeline
6. **Preview & Download** the cleaned CSV.

### Tab 2 — **Metadata Inference & Usability**

1. **Upload up to 5 CSVs** (or a single CSV). If multiple, configure **joins** (left/right keys + join type) and merge.
2. **Run Inference** → get a table with inferred types.
3. **Explore**

   * **D‑Tale** link to inspect data interactively.
   * **Visualizations by type** (bar/word cloud/box+heatmap/line/map/swatch/etc.).
   * **General URL** columns: choose a link, add an instruction (e.g., “summarize key points”), and the app fetches the page and asks the LLM to summarize.

### Advanced — **Interactive Decision Tree**

* Click **Show Interactive Graph** → pick a column → generate action tree.
* Click a **leaf node** to apply that cleaning action sequence via LLM.
* The executed code is shown; the resulting DataFrame updates in place; repeated clicks on the same leaf are ignored.

### "Custom Trained LLM via API" (footer section)

* Free‑form prompt box, always visible below the tabs. Despite the name it now runs through the same OpenRouter/GLM‑4.6 client as the rest of the app rather than a separately hosted model — the original self‑hosted-endpoint version never worked in production, so this got consolidated.
* Response is shown in a text area with timing info.

---

## `table_steps.json` (UI Config)

A minimal example to illustrate the shape (your file may be richer):

```json
{
  "processing": {
    "missing_values": [
      {
        "name": "Drop Nulls",
        "description": "Drop rows with too many missing values",
        "options": [
          { "name": "threshold", "data_type": "float", "value": 0.5 }
        ]
      }
    ],
    "encoding": [
      {
        "name": "One-Hot Encode",
        "description": "Encode low-cardinality categoricals",
        "options": [
          { "name": "max_unique", "data_type": "int", "value": 20 }
        ]
      }
    ]
  }
}
```

Each `option` supports `data_type` (`int`, `float`, `str`, `select`) and optional `options` (for dropdowns). The UI will render appropriate widgets and collect parameters into the `steps` list for `pipeline_logic.run_pipeline`.

---

## Audit Logging

* **Session**: A unique `session_id` is created (`uuid4`).
* **CSV Save**: After upload/merge, the app writes to the audit folder:
  `uploaded_<UTC_YYYYMMDD-HHMMSS>_<session8>.csv`
* **DB Logging**: `log_session(session_id)`, `log_file(session_id, filename, path)`, `log_event(session_id, event_type, event_detail)` write to SQLite (`audit.db`).

> **Note:** The exact SQLite schema is defined in `DB/log_to_db.py`. Typical events include `file_upload`, `inference_triggered`, `column_visualized`, `custom_viz_success/error`, `custom_cleaning_success/error`, `agraph_tree_generated`, `agraph_node_cleaning_success/error`, and `feedback`.

---

## Security & Safety

* **Review before execution**: LLM‑generated code is shown in the UI; execute only if you trust it. There's no real sandbox around it yet — see [issue #4](https://github.com/iamvisheshsrivastava/Smart_CSV_Toolkit/issues/4) if you want the details, it's a known gap and the fix is more involved than it looks.
* **Network requests**: General URL analysis fetches webpages; outbound URLs are checked against a public-IP allowlist to block SSRF against internal/link-local addresses, but that's not the same as trusting arbitrary domains — avoid feeding it links you don't control.
* **Secrets**: Keep API keys in environment variables or in `.streamlit/secrets.toml` (don't commit them). The OpenRouter client reads `OPENROUTER_API_KEY` from secrets.toml or environment.
* **CSV export**: Cells that could be interpreted as spreadsheet formulas (leading `=`, `+`, `-`, `@`) get neutralized on download so a "cleaned" CSV can't turn into a formula-injection payload when someone opens it in Excel/Sheets.
* **Session isolation**: the vector-memory store used for "recall past cleaning instructions" is scoped per session, not shared globally — an earlier version leaked one user's cleaning code to every other visitor ([#9](https://github.com/iamvisheshsrivastava/Smart_CSV_Toolkit/issues/9)).

---

## Requirements

* Python **3.10+**
* See `requirements.txt` for the full list (notably: `streamlit`, `pandas`, `plotly`, `matplotlib`, `seaborn`, `wordcloud`, `beautifulsoup4`, `dtale`, `streamlit-agraph`, `pyvis`, `openai`, `requests`). The prod deploy uses the slimmer `requirements-prod.txt` — no `torch`/`transformers`, which trims the Docker image down considerably.

---

## Development Notes

* **VS Code Dev Container**: Open the repo in VS Code → "Reopen in Container" to develop in a preconfigured environment (see `.devcontainer`).
* **Styling/UX**: Streamlit components, Plotly charts, Matplotlib/Seaborn for custom visuals, AGraph for the interactive tree, and D‑Tale for data exploration.
* **Paths**: the audit directory is resolved relative to `app.py` with `pathlib`, so it works the same on Windows/macOS/Linux — no more hardcoded `C:\...` paths.

---

## Docker & CI/CD

* This app can run as **one container** because the current codebase is a Streamlit app only.
* **Live deployment:** hosted on [Render](https://render.com)'s free tier via `render.yaml` + `Dockerfile` — Render auto-redeploys on every push to `main`. The app previously SSH-deployed to a self-hosted DigitalOcean droplet; that droplet has been decommissioned.
* For local Docker use, `docker-compose.yml` still works:

```bash
docker compose up --build -d
```

* The GitHub Actions workflow in `.github/workflows/ci-cd.yml` runs syntax checks on every PR/push. Its SSH-deploy step targets the retired droplet and is no longer functional — actual deployment now happens via Render's own auto-deploy, not this workflow.
* If you later add a FastAPI backend, then splitting into **two containers** makes sense. For now, one container is enough.

---

## Troubleshooting

* *D‑Tale link not opening*: Ensure your browser can reach the host/port D‑Tale binds to; check firewall and proxy; try opening the printed URL directly.
* *LLM suggestions/code empty or errors*: Confirm `OPENROUTER_API_KEY` is set in `.streamlit/secrets.toml`; ensure your OpenRouter account has credit/quota available; retry with a simpler instruction.
* *Type-inference fallback errors*: only relevant if you've set `CUSTOM_LLM_API_URL` — make sure that server is running and returns `{ "response": "..." }` JSON. Left unset, it just silently falls back to `"Text"`.
* *Large CSVs*: If memory is tight, run with a smaller sample or increase system RAM; consider chunked processing in future extensions.
* *Visualization errors*: Some plots assume valid numeric/datetime parsing; ensure columns are cast or adjust instructions accordingly.

---

## Roadmap (Ideas)

* Real sandboxing for LLM‑generated code (`__builtins__ = {}` alone doesn't cut it — see #4)
* Non‑LLM heuristics for the type-inference fallback, so it degrades better without an API key
* Undo/history for Tab 2's in‑place cleaning steps (#13)
* Built‑in CSV join diagnostics and key suggestions
* Executable cleaning **playback** (export steps as a Python script)
* Multi‑page layout (Cleaner / Inspector / Recipes / Logs)

---

## Contributing

Issues and PRs are welcome. Please include a clear description, steps to reproduce, and screenshots/logs where helpful.

---

