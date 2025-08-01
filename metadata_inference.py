import pandas as pd
import re
import json
from datetime import datetime
import numpy as np
from together import Together
import json
from typing import Tuple
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import nltk
from geopy.distance import geodesic
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tldextract
from llm_utils import load_deepseek_model, generate_response

nltk.download("stopwords")
english_stops = set(stopwords.words("english"))

ml_friendly_types = [
    "Numerical",
    "Boolean",
    "Categorical",
    "Ordinal",
    "Text",
    "Datetime",
    "GPS Coordinates",
    "Percentage",
    "Currency",
    "Duration / Timedelta",
    "Image URL"
]

def infer_column_type(col: pd.Series) -> str:
    col_str = col.dropna().astype(str).str.strip()
    unique_ratio = col.nunique(dropna=True) / max(len(col), 1)
    null_ratio = col.isna().sum() / max(len(col), 1)
    sample = col_str.sample(min(100, len(col_str)), random_state=42)

    # 1. Null-heavy
    if null_ratio > 0.8:
        return "Null-heavy"

    # 2. Constant / Low Variance
    if col.nunique(dropna=True) <= 1:
        return "Constant / Low Variance"

    # 3. Image URL
    if sample.str.contains(r'https?://.*\.(jpg|jpeg|png|gif)$', case=False).mean() > 0.5:
        return "Image URL"

    # 4. Video URL (YouTube or direct links)
    youtube_pattern = r"(youtu\.be/|youtube\.com/(watch\?v=|embed/))"
    video_file_pattern = r"\.(mp4|mov|avi|webm)$"

    if sample.str.contains(youtube_pattern, case=False).mean() > 0.5 or \
    sample.str.contains(video_file_pattern, case=False).mean() > 0.5:
        return "Video URL"

    # 5. Document URL
    if sample.str.contains(r'https?://.*\.(pdf|docx?|xlsx?|csv)$', case=False).mean() > 0.5:
        return "Document URL"

    # 6. General URL
    if sample.str.contains(r'https?://', case=False).mean() > 0.5:
        return "General URL"

    # 7. File Path
    if sample.str.contains(r'([a-zA-Z]:\\|/).*\.\w+$').mean() > 0.5:
        return "File Path"

    # 8. GPS Coordinates
    if sample.str.contains(r'^-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+$').mean() > 0.5:
        return "GPS Coordinates"

    # 9. Email Address
    if sample.str.contains(r'^[\w\.-]+@[\w\.-]+\.\w+$').mean() > 0.5:
        return "Email Address"

    # 10. Phone Number
    if sample.str.contains(r'^\+?\d[\d\s\-]{7,}$').mean() > 0.5:
        return "Phone Number"

    # 11. Currency
    if sample.str.contains(r'^[$€₹]\s?\d+', case=False).mean() > 0.5:
        return "Currency"

    # 12. Percentage
    if sample.str.contains(r'^\d+(\.\d+)?%$').mean() > 0.5:
        return "Percentage"

    # 13. Color Code
    if sample.str.contains(r'^(#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})|rgb\()').mean() > 0.5:
        return "Color Code"

    # 14. JSON / Nested
    if sample.str.startswith("{").mean() > 0.5:
        return "JSON / Nested"
    
    # 15. Numerical
    if pd.api.types.is_numeric_dtype(col):
        return "Numerical"

    # 16. Datetime
    dt_valid = pd.to_datetime(col, errors='coerce')
    if dt_valid.notna().sum() / max(len(col), 1) > 0.7:
        return "Datetime"

    # 17. Boolean
    if sample.isin(["True", "False", "true", "false", "0", "1"]).mean() > 0.8:
        return "Boolean"

    # 18. Identifier / ID
    if col.nunique(dropna=True) == len(col.dropna()):
        return "Identifier / ID"

    # 19. Categorical
    if unique_ratio < 0.05 and col.dtype == object:
        return "Categorical"

    # 20. Ordinal (simple heuristic)
    known_ordinals = ["low", "medium", "high", "rare", "common", "excellent", "poor"]
    if sample.str.lower().isin(known_ordinals).mean() > 0.5:
        return "Ordinal"

    # 21. Duration
    if sample.str.contains(r'^\d+:\d{2}(:\d{2})?$').mean() > 0.5:
        return "Duration / Timedelta"

    # 22. Mixed / Ambiguous
    type_set = set(type(v).__name__ for v in col.dropna().sample(min(20, len(col))))
    if len(type_set) > 1:
        return "Mixed / Ambiguous"

    # 23. Text (fallback)
    return "Text"

def get_cleaning_and_enrichment_suggestions(df: pd.DataFrame) -> dict:
    column_details = []

    for col in df.columns:
        col_data = df[col]
        col_type = infer_column_type(col_data)
        sample_values = col_data.dropna().astype(str).tolist()[:3]

        column_details.append({
            "column_name": col,
            "column_type": col_type,
            "sample_values": sample_values
        })

    prompt = (
        "You are a data analysis assistant.\n"
        "For each of the following columns, suggest:\n"
        "1. A practical cleaning/transformation suggestion (for ML pipelines).\n"
        "2. A useful enrichment/derived feature suggestion.\n\n"
        "Respond in valid JSON ONLY with this format:\n"
        "{\n"
        "  \"cleaning\": {\n"
        "    \"column1\": \"...\",\n"
        "    \"column2\": \"...\"\n"
        "  },\n"
        "  \"enrichment\": {\n"
        "    \"column1\": \"...\",\n"
        "    \"column2\": \"...\"\n"
        "  }\n"
        "}\n\n"
        "Here are the columns:\n"
    )

    for detail in column_details:
        prompt += (
            f"- Column Name: {detail['column_name']}\n"
            f"  Type: {detail['column_type']}\n"
            f"  Sample Values: {detail['sample_values']}\n\n"
        )

    response = call_llm(prompt)

    response = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(response)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        print("⚠️ Failed to parse LLM response:", e)
        print("🔍 Raw response was:", response[:500])
        return {}


def get_pipeline_score(series, col_type):
    total = len(series)
    missing_pct = series.isna().mean() * 100
    n_unique = series.nunique()
    unique_pct = (n_unique / total) * 100 if total > 0 else 0
    description = f"{missing_pct:.1f}% missing | {unique_pct:.1f}% unique"

    if col_type in ["Numerical", "Boolean", "Percentage", "Currency"]:
        min_val = pd.to_numeric(series, errors="coerce").min()
        max_val = pd.to_numeric(series, errors="coerce").max()
        if missing_pct < 5:
            return f"⭐⭐⭐⭐⭐ — Clean numeric ({description} | Min: {min_val}, Max: {max_val})"
        else:
            return f"⭐⭐⭐ — Numeric column with moderate issues ({description})"

    if col_type in ["Categorical", "Ordinal"]:
        top_vals = series.value_counts().head(2).to_dict()
        example = ', '.join([f"{k} ({v})" for k, v in top_vals.items()])
        if n_unique <= 20 and missing_pct < 10:
            return f"⭐⭐⭐⭐ — Low-cardinality categorical ({description} | Top: {example})"
        elif n_unique <= 50 and missing_pct < 20:
            return f"⭐⭐⭐ — Medium-cardinality categorical ({description} | Top: {example})"
        else:
            return f"⭐⭐ — High-cardinality ({description} | Top: {example})"

    if col_type == "Text":
        lengths = series.dropna().astype(str).apply(len)
        avg_len = lengths.mean()
        return f"⭐⭐⭐ — Text column ({description} | Avg length: {avg_len:.1f} chars)"

    if col_type == "Datetime":
        dt_valid = pd.to_datetime(series, errors="coerce")
        if dt_valid.notna().sum() > 0:
            range_info = f" | Range: {dt_valid.min().date()} to {dt_valid.max().date()}"
        else:
            range_info = ""
        return f"⭐⭐⭐ — Datetime column ({description}{range_info})"

    if col_type == "Duration / Timedelta":
        time_series = pd.to_timedelta(series, errors="coerce")
        if time_series.notna().sum() > 0:
            range_info = f" | Range: {time_series.min()} to {time_series.max()}"
        else:
            range_info = ""
        return f"⭐⭐⭐ — Duration data ({description}{range_info})"

    if col_type == "GPS Coordinates":
        return f"⭐⭐⭐ — Location data ({description})"

    if col_type == "Image URL":
        pct_valid = series.dropna().str.contains(r'\.(jpg|jpeg|png|gif)$', case=False).mean() * 100
        return f"⭐⭐⭐ — Image references ({description} | {pct_valid:.1f}% valid image URLs)"

    if missing_pct > 70:
        return f"⭐ — Too many missing values ({missing_pct:.1f}%)"

    return f"⭐ — Unsupported or unclear type ({description})"

def get_viz_capability(col_type: str, col_name: str = "") -> str:
    """
    Returns the visualization usability indicator with chart type if supported.
    """
    plot_map = {
        "Numerical": "Box Plot",
        "Categorical": "Bar Chart",
        "Boolean": "Bar Chart",
        "Ordinal": "Histogram",
        "Text": "Word Cloud",
        "Datetime": "Time Series Line Chart",
        "GPS Coordinates": "Map",
        "Percentage": "Histogram",
        "Currency": "Histogram",
        "Color Code": "Swatches",
        "Email Address": "Top Values",
        "Phone Number": "Top Values",
        "Image URL": "Image Viewer",
        "Video URL": "Video Preview",
        "General URL": "LLM Summary",
        "Document URL": "Download Viewer",
        "File Path": "Download Viewer",
    }

    if col_type == "General URL" and col_name.lower() in ["website", "site", "webpage"]:
        return "✅ (LLM Summary)"

    if col_type in plot_map:
        return f"✅ ({plot_map[col_type]})"

    return "❌"

def analyze_dataframe(df):
    metadata = []
    response = "—"
    try:
        tokenizer, model = load_deepseek_model()
        response = generate_response("Print hello world in Python", tokenizer, model)        
        #suggestions = get_cleaning_and_enrichment_suggestions(df)
        
        #cleaning_map = suggestions.get("cleaning", {})
        #enrichment_map = suggestions.get("enrichment", {})
    except Exception as e:
        print("Error parsing LLM JSON response:", e)
        cleaning_map = {}
        enrichment_map = {}

    for col in df.columns:
        series = df[col]
        col_type = infer_column_type(series)
        usable_for_ml = col_type in ml_friendly_types
        usable_for_viz = get_viz_capability(col_type, col)
        ml_readiness = get_pipeline_score(series, col_type) if usable_for_ml else "—"

        metadata.append({
            "Column": col,
            "Inferred Type": col_type,
            "Usable for ML": "✅" if usable_for_ml else "❌",
            "ML Readiness": ml_readiness,
            "Usable for Visualization": usable_for_viz,
            #"Suggested Improvements": cleaning_map.get(col, "—"),
            "Cleaning Suggestion": response,
            #"Enrichment Suggestion": enrichment_map.get(col, "—")       
        })

    return pd.DataFrame(metadata)





def custom_cleaning_via_llm(user_instruction: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Calls the LLM with a strict prompt, parses returned code from JSON, executes it on df.

    Args:
        user_instruction (str): User's natural language cleaning instruction.
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        Tuple[pd.DataFrame, str]: Cleaned DataFrame and executed code.
    """
    formatted_df = df.to_csv(index=False)
    prompt = f"""
You are a Python data cleaning assistant.

Your task is to generate Python code that modifies the `df` DataFrame *in-place* based on the user's instruction. You have access to the full dataset below in CSV format. 

Make sure the code:
- Assumes that a pandas DataFrame named `df` already exists.
- Can handle all rows of the dataset, not just a sample.
- Does not include any explanations, comments, or markdown.
- Only returns a JSON object of the form: {{ "code": "<your_python_code>" }}

USER INSTRUCTION:
{user_instruction}

FULL DATAFRAME (CSV FORMAT):
{formatted_df}

PROMPT LENGTH (characters): {len(user_instruction) + len(formatted_df)}
"""

    try:
        llm_response = call_llm(prompt)
        code_data = json.loads(llm_response)
        code_str = code_data.get("code", "")

        if not code_str:
            raise ValueError("No 'code' key found in LLM response.")

        global_vars = {
            "pd": pd,
            "np": np,
            "re": re,
            "datetime": datetime,
            "SimpleImputer": SimpleImputer,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "Binarizer": Binarizer,
            "SMOTE": SMOTE,
            "RandomOverSampler": RandomOverSampler,
            "RandomUnderSampler": RandomUnderSampler,
            "nltk": nltk,
            "geopy": __import__("geopy"),  
            "geodesic": geodesic,
            "stop_words": english_stops,
            "stops": set(stopwords.words("english")),
            "word_tokenize": word_tokenize,
            "transformers": __import__("transformers"),
            "tldextract": tldextract
        }

        local_vars = {"df": df.copy()}

        exec(code_str, global_vars, local_vars)

        return local_vars["df"], code_str

    except Exception as e:
        raise RuntimeError(f"Failed to apply LLM cleaning: {e}")

def call_llm(prompt: str, temperature=0.3, max_tokens=700) -> str:
    client = Together()
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        #model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib
import re
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def execute_plot_code(code: str, df: pd.DataFrame):
    """
    Executes LLM-generated plot code on a given DataFrame using a safe, scoped environment.
    """
    global_vars = {
        "pd": pd,
        "np": np,
        "re": re,
        "plt": plt,
        "sns": sns,
        "datetime": datetime,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "Binarizer": Binarizer,
        "SMOTE": SMOTE,
        "RandomOverSampler": RandomOverSampler,
        "RandomUnderSampler": RandomUnderSampler
    }

    local_vars = {"df": df}

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, global_vars, local_vars)

        fig = plt.gcf()
        fig.set_size_inches(6, 4)  
        return fig

    except Exception as e:
        raise RuntimeError(f"Error executing visualization code: {e}")
