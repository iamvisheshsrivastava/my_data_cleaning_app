# Smart CSV Toolkit — LLM‑Assisted CSV Cleaning & Metadata Inference

A Streamlit app that helps you **clean CSVs, infer column semantics, merge multiple files, and generate visualizations** — with optional help from an LLM. It’s designed to be practical and auditable: you always see the code an LLM proposes, you can accept/reject steps, and your actions are **logged to SQLite**.

> **TL;DR**
> • **Tab 1 – CSV Cleaner:** choose built‑in steps from `table_steps.json`, get **LLM suggestions**, optionally apply **LLM‑generated Python code** (shown before execution), and download the cleaned CSV.
> • **Tab 2 – Metadata Inspector:** upload/merge up to 5 CSVs, **infer column types**, preview media/links, and **auto‑generate plots** (with or without LLM help).
> • **Advanced:** interactive **decision‑tree UI** for guided cleaning; optional **custom LLM API**; session/file/event **audit logging** with timestamped CSVs.

---

## ✨ Features

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

  * Upload up to **5** CSVs, configure pairwise joins (keys + type), then merge with one click

* **Interactive Cleaning Graph**

  * AGraph‑based decision tree per column; **click leaf nodes** to apply LLM‑generated cleaning for the chosen path; executed code is surfaced and actions are tracked

* **Exploration via D‑Tale**

  * One‑click link to open D‑Tale and explore the current DataFrame

* **Audit Logging**

  * Sessions, files, and events logged to SQLite via `DB/log_to_db.py`
  * Uploaded/merged CSVs saved into an **audit folder** with timestamp + session id

* **Optional Custom LLM API**

  * Query your own model endpoint (e.g., DeepSeek Coder) via `LLM/config.py`

---

## 🗂️ Repository Layout

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
│  └─ config.py               # your custom LLM API endpoint config
├─ .streamlit/                # Streamlit settings
├─ .devcontainer/             # VS Code Dev Container setup
├─ requirements.txt           # Python dependencies
└─ README.md                  # (this file)
```

---

## ⚙️ How It Works (High Level)

1. **Config‑Driven UI** — `table_steps.json` defines sections, step names, descriptions, and typed options. `app.py` renders controls automatically and builds a `steps` list.
2. **Pipeline Execution** — `pipeline_logic.run_pipeline(df, steps)` executes selected steps in order. If an LLM instruction was accepted, its generated code is appended as a step and executed safely within the pipeline wrapper.
3. **LLM Helpers**

   * `call_llm()` uses the Together API with model `moonshotai/Kimi-K2-Instruct` to:

     * generate *cleaning suggestions* (`fetch_llm_suggestions`)
     * translate a *natural instruction* → **raw Python code** (`get_cleaning_code_from_llm`)
   * A separate **custom LLM API** block posts to `LLM.config.API_URL`.
4. **Metadata Inference & Visuals** — `metadata_inference.analyze_dataframe(df)` infers types and suggests basic visualizations; URL columns can be summarized via an LLM.
5. **Decision Tree** — `cleaningDecisionTree.render_agraph_tree()` builds a compact action tree (max branching/leaf count). Clicking a **leaf** triggers `custom_cleaning_via_llm()` with contextual instruction; code and results are shown.
6. **Auditability** — All key actions are logged via `log_session`, `log_file`, and `log_event`. CSVs are saved to an audit directory with timestamp and session id.

---

## 🚀 Quick Start

### 1) Clone & Create Environment

```bash
git clone https://github.com/iamvisheshsrivastava/my_data_cleaning_app.git
cd my_data_cleaning_app

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

**Together API (used by `call_llm`)**

* Create an account and get an API key.
* Set the environment variable before running Streamlit:

```bash
# Windows PowerShell
$env:TOGETHER_API_KEY = "YOUR_KEY"

# macOS/Linux
export TOGETHER_API_KEY="YOUR_KEY"
```

**Custom LLM endpoint (used by the bottom "Custom Trained LLM via API" section)**

Create/adjust `LLM/config.py` (already present in the repo). Example:

```python
# LLM/config.py
API_URL = "http://localhost:9000/generate"  # your FastAPI/Flask inference endpoint
# If you need headers/auth, modify app.py where requests.post is called.
```

> The app posts `{ "prompt": "..." }` to `API_URL` and expects `{ "response": "..." }` in return. SSL verification is disabled in that call by default (`verify=False`).

### 3) Audit Folder Path

In `app.py`, the audit directory defaults to a **Windows path**:

```python
AUDIT_DIR = r"C:\\Users\\sriva\\Desktop\\AICUFLow\\my_data_cleaning_app\\DB\\auditCSVFiles"
```

Change this to a portable relative path if you’re on macOS/Linux:

```python
from pathlib import Path
AUDIT_DIR = Path("DB/auditCSVFiles")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
```

### 4) Run the App

```bash
streamlit run app.py
```

---

## 🧩 Usage Guide

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

* Free‑form prompt UI that posts to `LLM.config.API_URL`.
* Response is shown in a text area with timing info.

---

## 🧱 `table_steps.json` (UI Config)

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

## 🧾 Audit Logging

* **Session**: A unique `session_id` is created (`uuid4`).
* **CSV Save**: After upload/merge, the app writes to the audit folder:
  `uploaded_<UTC_YYYYMMDD-HHMMSS>_<session8>.csv`
* **DB Logging**: `log_session(session_id)`, `log_file(session_id, filename, path)`, `log_event(session_id, event_type, event_detail)` write to SQLite (`audit.db`).

> **Note:** The exact SQLite schema is defined in `DB/log_to_db.py`. Typical events include `file_upload`, `inference_triggered`, `column_visualized`, `custom_viz_success/error`, `custom_cleaning_success/error`, `agraph_tree_generated`, `agraph_node_cleaning_success/error`, and `feedback`.

---

## 🔐 Security & Safety

* **Review before execution**: LLM‑generated code is shown in the UI; execute only if you trust it.
* **Network requests**: General URL analysis fetches webpages; avoid unknown or untrusted domains.
* **Secrets**: Keep API keys in environment variables (don’t commit them). The Together client reads `TOGETHER_API_KEY` from env.
* **SSL**: The custom LLM request uses `verify=False` by default; enable verification for production.

---

## 🧰 Requirements

* Python **3.10+**
* See `requirements.txt` for the full list (notably: `streamlit`, `pandas`, `plotly`, `matplotlib`, `seaborn`, `wordcloud`, `beautifulsoup4`, `dtale`, `streamlit-agraph`, `pyvis`, `together`, `requests`).

---

## 🛠️ Development Notes

* **VS Code Dev Container**: Open the repo in VS Code → "Reopen in Container" to develop in a preconfigured environment (see `.devcontainer`).
* **Styling/UX**: Streamlit components, Plotly charts, Matplotlib/Seaborn for custom visuals, AGraph for the interactive tree, and D‑Tale for data exploration.
* **Windows paths**: The default audit path is Windows‑specific; switch to `pathlib.Path` for portability as shown above.

---

## 🧩 Troubleshooting

* *D‑Tale link not opening*: Ensure your browser can reach the host/port D‑Tale binds to; check firewall and proxy; try opening the printed URL directly.
* *LLM suggestions/code empty or errors*: Confirm `TOGETHER_API_KEY` is set; the Together service reachable; retry with a simpler instruction.
* *Custom LLM API errors*: Ensure your server at `LLM.config.API_URL` is running and returns `{ "response": "..." }` JSON.
* *Large CSVs*: If memory is tight, run with a smaller sample or increase system RAM; consider chunked processing in future extensions.
* *Visualization errors*: Some plots assume valid numeric/datetime parsing; ensure columns are cast or adjust instructions accordingly.

---

## 🗺️ Roadmap (Ideas)

* More robust, non‑LLM type inference heuristics
* Built‑in CSV join diagnostics and key suggestions
* Executable cleaning **playback** (export steps as a Python script)
* Switch to portable audit paths by default; add env‑configurable audit dir
* Optional sandboxing for LLM‑generated code
* Multi‑page layout (Cleaner / Inspector / Recipes / Logs)

---

## 🤝 Contributing

Issues and PRs are welcome! Please include a clear description, steps to reproduce, and screenshots/logs where helpful.

---


---

## 🙌 Acknowledgements

* **Streamlit**, **pandas**, **Plotly**, **Matplotlib**, **Seaborn**, **WordCloud**, **BeautifulSoup**
* **D‑Tale**, **streamlit‑agraph**, **PyVis**
* **Together** (LLM API)

---

## 🧑‍💻 Author

**Vishesh Srivastava**
Portfolio: visheshsrivastava.com
GitHub: github.com/iamvisheshsrivastava
LinkedIn: linkedin.com/in/iamvisheshsrivastava
