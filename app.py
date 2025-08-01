import streamlit as st
import pandas as pd
import json
from pipeline_logic import run_pipeline
from metadata_inference import analyze_dataframe, custom_cleaning_via_llm, execute_plot_code  
import requests
import replicate
import os
from together import Together
import uuid
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
#from custom_components.interactive_table import interactive_table
from cleaningDecisionTree import render_pyvis_tree, render_agraph_tree
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
from PIL import Image
from io import BytesIO
import re
from streamlit_agraph import agraph, Config
from uuid import uuid4
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib

########################################################################################
############################CSV Cleaning with AI Suggestions############################
########################################################################################

def call_llm(prompt: str, temperature=0.3, max_tokens=700) -> str:
    client = Together()
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def get_cleaning_code_from_llm(instruction: str, df: pd.DataFrame) -> str:
            example_data = df.head(2).to_dict(orient="records")
            prompt = f"""
            You are a Python data cleaning assistant.

            INSTRUCTION: Convert the following user instruction into Python code that modifies the `df` DataFrame in-place.

            RULES:
            - DO NOT wrap your answer in markdown or backticks
            - DO NOT return explanation or comments
            - ONLY return raw Python code

            Instruction:
            \"\"\"{instruction}\"\"\"

            Sample Data:
            {json.dumps(example_data, indent=2)}
            """
            return call_llm(prompt, temperature=0.2, max_tokens=500)

def fetch_llm_suggestions(df, config):
    excluded_keywords = []
    for section, step_list in config.get("processing", {}).items():
        for step in step_list:
            excluded_keywords.append(step["name"].lower())
            excluded_keywords.append(step["description"].lower())

    example_data = df.head(2).to_dict(orient="records")

    prompt = f"""
        You are a smart data cleaning assistant.

        Your job is to suggest 5 **new**, **non-redundant**, and **directly executable** data cleaning actions based on the uploaded CSV file.

        IMPORTANT:
        - ONLY suggest actions that can be applied using the available data — no vague, manual, or unverifiable suggestions.
        - Do NOT suggest anything already covered in: {excluded_keywords}
        - Do NOT include ideas like “flag suspicious data” or “verify with external sources”

        Format:
        - Return a JSON array of 5 short, actionable suggestion **strings**
        - Do NOT include IDs, explanations, or markdown — just plain text

        Example output:
        [
        "Convert 'DOB' column to datetime format",
        "Drop columns with more than 50% missing values",
        ...
        ]

        Here is a preview of the data (first 2 rows):
        {json.dumps(example_data, indent=2)}

        Return ONLY the JSON array of 5 suggestion strings.
    """

    llm_output = call_llm(prompt)
    return json.loads(llm_output)


with open("table_steps.json", "r") as f:
    config = json.load(f)

st.set_page_config(page_title="Smart CSV Toolkit", layout="wide")
st.title("Smart CSV Toolkit")
tab1, tab2 = st.tabs(["🧼 CSV Cleaner", "🧠 Metadata Inspector"])

with tab1:
    st.header("CSV Cleaning with AI Suggestions")
    st.header("Step 1: Upload CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        num_rows = st.slider("Rows to display", min_value=5, max_value=len(df), value=10)
        st.dataframe(df.head(num_rows), use_container_width=True)

        st.header("Step 2: Select and Configure Processing Steps")
        steps = []
        sections = config.get("processing", {})

        if "df" in locals() or "df" in globals():
            categorical_columns = [col for col in df.columns if df[col].dtype == "object" or df[col].nunique() < 20]
        else:
            categorical_columns = []

        for section, step_list in sections.items():
            st.markdown(f"###  {section.replace('_', ' ').title()}")
            for step in step_list:
                step_key = f"step_{step['name']}"
                if st.checkbox(f"{step['name']} – {step['description']}", key=step_key):
                    params = {}
                    for opt in step.get("options", []):
                        param_key = f"{step['name']}_{opt['name']}"
                        dtype = opt.get("data_type")
                        default = opt.get("value")
                        choices = opt.get("options", [])

                        if opt["name"] == "target_column" and dtype == "select":
                            if categorical_columns:
                                params[opt["name"]] = st.selectbox(
                                    label=f"Select target column (categorical) for {step['name']}",
                                    options=categorical_columns,
                                    key=param_key
                                )
                            else:
                                st.warning("No categorical columns found in the dataset to use as target.")
                                params[opt["name"]] = None

                        elif dtype == "int":
                            if choices:
                                params[opt["name"]] = st.selectbox(param_key, choices, index=choices.index(default), key=param_key)
                            else:
                                params[opt["name"]] = st.number_input(param_key, value=int(default), key=param_key)

                        elif dtype == "float":
                            if choices:
                                params[opt["name"]] = st.selectbox(param_key, choices, index=choices.index(default), key=param_key)
                            else:
                                params[opt["name"]] = st.number_input(param_key, value=float(default), key=param_key)

                        elif dtype == "str":
                            if choices:
                                params[opt["name"]] = st.selectbox(param_key, choices, index=choices.index(default), key=param_key)
                            else:
                                params[opt["name"]] = st.text_input(param_key, value=default, key=param_key)

                    steps.append({"name": step["name"], "params": params})


        st.header("AI Cleaning Suggestions using LLM")

        if "llm_suggestions" not in st.session_state:
            st.session_state.llm_suggestions = []
        if "suggested_once" not in st.session_state:
            st.session_state.suggested_once = False
        if "clicked_llm" not in st.session_state:
            st.session_state.clicked_llm = False
        if "selected_suggestion" not in st.session_state:
            st.session_state.selected_suggestion = ""
        if "custom_suggestion" not in st.session_state:
            st.session_state.custom_suggestion = ""


        button_label = "Regenerate Suggestions" if st.session_state.suggested_once else "Get Smart LLM Suggestions"
        if st.button(button_label) and uploaded_file:
            try:
                st.session_state.llm_suggestions = fetch_llm_suggestions(df, config)
                st.session_state.clicked_llm = True
                st.session_state.suggested_once = True
                st.session_state.selected_suggestion = ""
                st.session_state.custom_suggestion = ""
            except Exception as e:
                st.error(f"Failed to fetch suggestions: {e}")

        selected_radio = None
        if st.session_state.clicked_llm and st.session_state.llm_suggestions:
            st.subheader("Suggested Improvements")
            suggestion_texts = st.session_state.llm_suggestions

            if suggestion_texts:
                selected_radio = st.radio(
                    "Select one suggestion to apply:",
                    suggestion_texts,
                    key="radio_selection"
                )

        if selected_radio and not st.session_state.custom_suggestion:
            st.session_state.selected_suggestion = selected_radio

        st.markdown("### Custom LLM Operation")
        custom_input = st.text_input(
            "Describe your own cleaning step (e.g., 'Drop column City')",
            value=st.session_state.custom_suggestion,
            key="custom_input"
        )

        if custom_input.strip() and custom_input != st.session_state.custom_suggestion:
            st.session_state.custom_suggestion = custom_input
            st.session_state.selected_suggestion = ""

        final_llm_operation = st.session_state.custom_suggestion or st.session_state.selected_suggestion

        st.markdown(f"**Final Cleaning Instruction:** `{final_llm_operation}`")


        with st.container():
            st.header("Step 3: Run and Download")

            if st.button("Run Cleaning Pipeline"):
                final_instruction = st.session_state.custom_suggestion or st.session_state.selected_suggestion
                if final_instruction:
                    try:
                        code = get_cleaning_code_from_llm(final_instruction, df)
                        st.warning("⚠️ This is the Python code automatically generated and executed by the LLM to perform the data cleaning. Please review it carefully to ensure correctness.")
                        st.code(code, language="python")
                        steps.append({
                            "name": "LLM Data Cleaning",
                            "params": {
                                "code": code
                            }
                        })
                    except Exception as e:
                        st.error(f"Failed to get code from LLM: {e}")
                        st.stop()

                try:
                    cleaned_df = run_pipeline(df.copy(), steps)
                    st.session_state.cleaned_df = cleaned_df 
                    st.success("✅ Cleaning complete!")

                except (ValueError, RuntimeError) as e:
                    st.warning(f"Some steps were skipped or failed:\n\n{e}")
                    st.stop()

            if "cleaned_df" in st.session_state:
                cleaned_df = st.session_state.cleaned_df

                with st.expander("Preview of Cleaned Data", expanded=True):
                    num_rows_output = st.slider(
                        "Rows to display (cleaned data)",
                        min_value=5,
                        max_value=len(cleaned_df),
                        value=10,
                        key="cleaned_rows"
                    )
                    visible_rows = cleaned_df.head(num_rows_output)
                    row_height = 35
                    min_height = 150
                    max_height = 300
                    st.dataframe(visible_rows, use_container_width=True, height=min(max(len(visible_rows) * row_height, min_height), max_height))

                csv = cleaned_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Cleaned CSV", data=csv, file_name="cleaned_output.csv", mime="text/csv")

    

##################################################################################################
##################################Metadata Inference & Usability##################################
##################################################################################################

def render_image_urls(urls):
    st.markdown("### 🖼️ Image Previews")
    for url in urls:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=url, use_column_width=True)
        except:
            st.warning(f"Could not load image: {url}")

def render_file_links(urls):
    st.markdown("### 📎 File Links")
    for url in urls:
        st.markdown(f"[📥 Download File]({url})", unsafe_allow_html=True)

def render_video_urls(urls):
    st.markdown("### 🎬 Video Previews")
    for url in urls:
        embed_url = None

        if "youtube.com/embed/" in url:
            embed_url = url

        elif "youtube.com/watch?v=" in url:
            video_id = url.split("watch?v=")[-1].split("&")[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

        elif url.lower().endswith((".mp4", ".webm", ".mov")):
            embed_url = url

        if embed_url:
            st.video(embed_url)
            st.caption(f"[🔗 Watch on YouTube]({url})")
        else:
            st.warning(f"⚠️ Unsupported video format: {url}")

def fetch_and_summarize_url(url: str, instruction: str, llm_func) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)[:1500]

        full_prompt = (
            f"You are a helpful assistant. The user wants to perform the following task on this web page:\n\n"
            f"URL: {url}\n\n"
            f"Extracted Content:\n{text}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Please analyze or summarize accordingly."
        )

        return llm_func(full_prompt)
    except Exception as e:
        return f"❌ Failed to fetch or process the URL: {e}"


def render_column_visualization(df: pd.DataFrame, col: str, inferred_type: str):
    with st.expander(f"Visualization for `{col}` ({inferred_type})", expanded=True):
        try:
            if inferred_type == "Categorical":
                top_k = df[col].value_counts().nlargest(10).reset_index()
                top_k.columns = [col, 'count']
                fig = px.bar(top_k, x=col, y='count')
                st.plotly_chart(fig)

            elif inferred_type == "Text":
                text_data = " ".join(df[col].dropna().astype(str)).lower()
                if text_data.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning("Text column is empty or invalid for wordcloud.")

            elif inferred_type == "Numerical":
                fig = px.box(df, y=col)
                st.plotly_chart(fig)
                if df.select_dtypes(include="number").shape[1] > 1:
                    st.markdown("**Correlation Heatmap**")
                    corr = df.select_dtypes(include="number").corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)

            elif inferred_type == "Datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
                timeline = df.groupby(df[col].dt.date).size().reset_index(name='Count')
                fig = px.line(timeline, x=col, y='Count')
                st.plotly_chart(fig)

            elif inferred_type == "GPS Coordinates":
                df[['lat', 'lon']] = df[col].str.split(",", expand=True).astype(float)
                st.map(df[['lat', 'lon']])

            elif inferred_type in ["Boolean", "Ordinal", "Percentage", "Currency"]:
                data = df[col].dropna()
                if inferred_type in ["Currency", "Percentage"]:
                    data = data.replace(r'[^\d\.]', '', regex=True).astype(float)
                fig = px.histogram(data, x=col)
                st.plotly_chart(fig)

            elif inferred_type == "Color Code":
                unique_colors = df[col].dropna().unique()[:20]
                st.markdown("**Color Swatches**")
                st.write("".join([
                    f"<div style='display:inline-block;width:50px;height:20px;background-color:{color};margin:2px;border:1px solid #000'></div>"
                    for color in unique_colors]), unsafe_allow_html=True)

            elif inferred_type in ["Email Address", "Phone Number"]:
                top_vals = df[col].value_counts().nlargest(10).reset_index()
                top_vals.columns = [col, 'count']
                fig = px.bar(top_vals, x=col, y='count')
                st.plotly_chart(fig)

            elif inferred_type == "Identifier / ID":
                st.info("Column identified as unique ID – typically not useful for visualization.")

            elif inferred_type in ["Null-heavy", "Constant / Low Variance"]:
                st.warning(f"Column classified as `{inferred_type}` – not suitable for meaningful visualization.")

            elif inferred_type in ["Image URL", "Video URL", "Document URL", "General URL", "File Path"]:
                urls = df[col].dropna().unique().tolist()

                if inferred_type == "Image URL":
                    with st.container():
                        st.markdown("<div style='max-height:400px; overflow-y:auto;'>", unsafe_allow_html=True)
                        render_image_urls(urls)
                        st.markdown("</div>", unsafe_allow_html=True)

                elif inferred_type == "Video URL":
                    urls = df[col].dropna().unique().tolist()
                    render_video_urls(urls)

                elif inferred_type == "General URL":
                    st.markdown("### 🌐 General URL Processor")

                    selected_url = st.selectbox("Select a URL to summarize", urls, key=f"{col}_url_select")
                    user_prompt = st.text_area("💬 What would you like to know or extract?", placeholder="e.g., Summarize the article, extract key points...", key=f"{col}_prompt_input")

                    if st.button("Analyze Webpage", key=f"{col}_analyze_button"):
                        if selected_url and user_prompt.strip():
                            with st.spinner("⏳ Fetching and analyzing..."):
                                result = fetch_and_summarize_url(selected_url, user_prompt, call_llm)

                            st.success("Done")
                            st.markdown("### Result:")
                            st.write(result)
                        else:
                            st.warning("⚠️ Please select a URL and provide an instruction.")


                elif inferred_type in ["Document URL", "File Path"]:
                    with st.container():
                        st.markdown("<div style='max-height:400px; overflow-y:auto;'>", unsafe_allow_html=True)
                        render_file_links(urls)
                        st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.write(urls) 

        except Exception as e:
            st.error(f"Error visualizing `{col}`: {str(e)}")


if "expanded_columns" not in st.session_state:
    st.session_state.expanded_columns = set()

with tab2:
    st.header("Metadata Inference & Usability")
    uploaded_file = st.file_uploader("Upload your CSV for Metadata Analysis", type=["csv"], key="meta")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        num_rows = st.slider("Rows to display", min_value=5, max_value=len(df), value=10)
        st.dataframe(df.head(num_rows), use_container_width=True)

        if st.button("🔍 Run Inference on Columns"):
            with st.spinner("Inferring column types and suggestions..."):
                st.session_state.metadata_df = analyze_dataframe(df)

        if "metadata_df" in st.session_state:
            metadata_df = st.session_state.metadata_df

            st.subheader("Column Type Inference")
            st.dataframe(metadata_df)

            st.subheader("Basic Suggested Visualizations")
            st.markdown("### Choose columns to visualize")
            selected_cols = st.multiselect(
                "Select columns to show visualizations for",
                options=list(metadata_df["Column"]),
                default=list(metadata_df["Column"])[:0],  
                key="selected_columns"
            )

            for col in selected_cols:
                inferred_type = metadata_df[metadata_df["Column"] == col]["Inferred Type"].values[0]
                render_column_visualization(df, col, inferred_type)

################################### Custom Visualization Via LLM ###################################
            st.subheader("🧪 Custom Visualization via LLM")

            user_viz_prompt = st.text_area(
                "Describe your custom visualization (e.g., 'Plot a bar chart of category counts')",
                height=100,
                key="custom_viz_prompt"
            )

            if st.button("✨ Generate and Render Plot with LLM"):
                if not user_viz_prompt.strip():
                    st.warning("Please enter a visualization instruction.")
                else:
                    prompt = f"""
            You are a Python data visualization assistant.

            TASK: Convert the following user instruction into Python code that uses matplotlib or seaborn to generate a plot from a pandas DataFrame named `df`.

            RULES:
            - DO NOT wrap the code in markdown or backticks.
            - DO NOT include explanations or comments.
            - The code should work directly with `df` and create the plot.
            - Use seaborn or matplotlib as needed.
            - The DataFrame has the following columns: {', '.join(df.columns)}.

            USER INSTRUCTION: {user_viz_prompt}
            """
                    with st.spinner("Calling LLM to generate plot..."):
                        try:
                            response = call_llm(prompt)
                            fig = execute_plot_code(response, df)
                            with st.container():
                                st.markdown("### 📊 Generated Plot")
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(str(e))



################################### Custom Cleaning Via LLM ###################################
            if "df" not in st.session_state:
                st.session_state.df = df.copy()  # Initial load

            st.markdown("## 🛠️ Custom Cleaning via LLM")

            with st.form(key="custom_cleaning_form"):
                user_instruction = st.text_area("Enter your custom cleaning instruction")
                submitted = st.form_submit_button("Submit to LLM")

                if submitted and user_instruction:
                    with st.spinner("Calling LLM and applying changes..."):
                        try:
                            cleaned_df, executed_code = custom_cleaning_via_llm(user_instruction, st.session_state.df)
                            
                            st.session_state.df = cleaned_df

                            st.success("✅ Cleaning applied successfully!")
                            st.markdown("### Executed Code")
                            st.code(executed_code, language="python")

                        except Exception as err:
                            st.error("❌ Failed to apply cleaning.")
                            st.markdown("#### Error Details")
                            st.error(str(err))

            st.subheader("📄 Current Working CSV")
            st.dataframe(st.session_state.df)



################################### Decision Tree Visualization ##################################

            # if "show_tree" not in st.session_state:
            #     st.session_state["show_tree"] = False

            # if "executed_actions" not in st.session_state:
            #     st.session_state["executed_actions"] = set()

            # if st.button(" Show Interactive Decision Tree for Advanced Cleaning"):
            #     st.session_state["show_tree"] = not st.session_state["show_tree"]
            #     st.write(" Toggled tree visibility:", st.session_state["show_tree"])

            # if st.session_state["show_tree"]:
            #     st.subheader(" Interactive Pyvis Decision Tree")

            #     working_df = st.session_state.df
            #     col = st.selectbox("Select column for decision tree", working_df.columns)

            #     if col:
            #         st.write(" Selected column:", col)

            #         response = render_pyvis_tree(working_df, call_llm, col)

            #         if response["status"] == "success":
            #             components.html(response["html"], height=600, scrolling=True)

            #             st.subheader(" Execute Suggested Cleaning Actions")

            #             # Extract leaves from response
            #             leaves = response.get("leaves", [])
            #             st.write(" Leaf actions found:", leaves)

            #             for leaf in leaves:
            #                 is_executed = leaf in st.session_state["executed_actions"]
            #                 button_label = f"{'✅' if is_executed else '🔷'} {leaf}"

            #                 st.write(f" Rendering button for leaf: {leaf} | Executed: {is_executed}")

            #                 if st.button(button_label, key=f"{col}_{leaf}"):
            #                     st.write(f" Button clicked for: {leaf}")

            #                     try:
            #                         instruction = f"Apply the following action on the column '{col}': {leaf}"
            #                         cleaned_df, code_str = custom_cleaning_via_llm(instruction, st.session_state.df)

            #                         # Update state with cleaned data
            #                         st.session_state.df = cleaned_df
            #                         st.session_state["executed_actions"].add(leaf)

            #                         st.success(f" Action performed: `{leaf}`")
            #                         st.code(code_str, language="python")
            #                         st.dataframe(cleaned_df)

            #                     except Exception as e:
            #                         st.error(f"❌ Error while applying action: {e}")
            #         else:
            #             st.error(f"❌ {response['message']}")


            if "show_tree" not in st.session_state:
                st.session_state["show_tree"] = False
            if "executed_actions" not in st.session_state:
                st.session_state["executed_actions"] = set()
            if "agraph_tree_data" not in st.session_state:
                st.session_state["agraph_tree_data"] = None
            if "agraph_leaf_nodes" not in st.session_state:
                st.session_state["agraph_leaf_nodes"] = []
            if "agraph_col" not in st.session_state:
                st.session_state["agraph_col"] = ""
            if "last_executed_click" not in st.session_state:
                st.session_state["last_executed_click"] = ""
            if "last_executed_code" not in st.session_state:
                st.session_state["last_executed_code"] = ""

            if st.button("Show Interactive Graph for Advanced Cleaning"):
                st.session_state["show_tree"] = not st.session_state["show_tree"]
                #st.write("Toggled tree visibility:", st.session_state["show_tree"])

            if st.session_state["show_tree"]:
                st.subheader("Interactive AGraph")

                working_df = st.session_state.df
                selected_col = st.selectbox("Select column for decision tree", working_df.columns)

                if selected_col:
                    st.write("Selected column:", selected_col)

                    if (
                        st.session_state["agraph_tree_data"] is None
                        or st.session_state["agraph_col"] != selected_col
                    ):
                        response = render_agraph_tree(working_df, call_llm, selected_col)

                        if response["status"] == "success":
                            st.session_state["agraph_tree_data"] = {
                                "nodes": response["nodes"],
                                "edges": response["edges"]
                            }
                            st.session_state["agraph_leaf_nodes"] = response["leaves"]
                            st.session_state["agraph_col"] = selected_col
                        else:
                            st.error(f"❌ {response['message']}")
                            st.stop()

                    config = Config(
                        width=800,
                        height=400,
                        directed=True,
                        nodeHighlightBehavior=True,
                        highlightColor="#F7A7A6",
                        collapsible=True,
                        physics=True,
                    )

                    tree_data = st.session_state["agraph_tree_data"]
                    result = agraph(
                        nodes=tree_data["nodes"],
                        edges=tree_data["edges"],
                        config=config
                    )

                    if isinstance(result, str) and result != st.session_state["last_executed_click"]:
                        clicked_node = result
                        st.session_state["last_executed_click"] = clicked_node

                        st.write(f"Clicked node: `{clicked_node}`")

                        if clicked_node in st.session_state["agraph_leaf_nodes"]:
                            if clicked_node in st.session_state["executed_actions"]:
                                st.info(f"ℹ️ `{clicked_node}` already executed.")
                            else:
                                try:
                                    path = [clicked_node]
                                    current = clicked_node
                                    parent_map = st.session_state.get("agraph_parent_map", {})
                                    while True:
                                        parent = parent_map.get(current)
                                        if parent is None or parent.startswith("Column: "):
                                            break
                                        path.insert(0, parent)
                                        current = parent

                                    context_path = " → ".join(path)
                                    instruction = f"For the column '{selected_col}', apply the following action sequence:\n{context_path}"

                                    cleaned_df, code_str = custom_cleaning_via_llm(instruction, st.session_state.df)

                                    st.session_state.df = cleaned_df
                                    st.session_state["executed_actions"].add(clicked_node)
                                    st.session_state["last_executed_code"] = code_str  

                                    st.success(f"Cleaning action applied for `{clicked_node}`")
                                    st.code(code_str or st.session_state["last_executed_code"], language="python")

                                    st.subheader(" Updated Working DataFrame")

                                except Exception as e:
                                    st.error(f"❌ Error while applying cleaning: {e}")

                    num_rows = st.slider(
                        "Rows to display",
                        min_value=5,
                        max_value=len(st.session_state.df),
                        value=10,
                        key="updated_df_row_slider"
                    )
                    st.dataframe(st.session_state.df.head(num_rows), use_container_width=True)
                    