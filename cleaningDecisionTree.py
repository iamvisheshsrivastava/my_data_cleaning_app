import re
import os
import tempfile
from pyvis.network import Network
import pandas as pd
import metadata_inference
from streamlit_agraph import Node, Edge

def generate_tree_prompt(df: pd.DataFrame) -> str:
    """
    Generates a decision tree prompt with direct, data-specific preprocessing steps
    (not conditional rules), based on summary stats and sample values
    from a single-column DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with exactly one column

    Returns:
        str: Prompt string to send to the LLM
    """
    assert df.shape[1] == 1, "DataFrame must contain exactly one column."

    col = df.columns[0]
    series = df[col]

    total = len(series)
    missing = series.isna().sum()
    unique = series.nunique()
    inferred_type = metadata_inference.infer_column_type(series)

    samples = series.dropna().astype(str).sample(min(10, len(series.dropna())), random_state=42).tolist()
    sample_text = "\n".join(samples)

    if len(sample_text) > 2000:
        sample_text = sample_text[:2000] + "..."

    metadata_summary = f"""Type: {inferred_type}
Missing Values: {missing} / {total}
Unique Values: {unique}
Total Rows: {total}
"""

    return f"""
You are a data preprocessing assistant.

Below is the metadata and sample values for a specific column. Based on this concrete data, suggest the most appropriate cleaning or transformation actions using a branching tree format.

Important:
- DO NOT use conditional logic (no 'if X then do Y')
- DO NOT generate generalized rules
- DO NOT break down identical operations into multiple nodes
- DO NOT use separate branches to show different values for the same transformation
- Group similar operations into one node if they share the same logic
- Use only 1 example per step if necessary (avoid listing multiple mappings)
- Assume the metadata is already known and fixed
- Only list exact preprocessing actions applicable to this column
- Use MAX 2 branches from the Start node
- Use MAX 4 final leaves in total
- Format it as a tree with indentation and branching lines
- DO NOT include explanations or comments
- DO NOT use markdown or backticks

Column Name: {col}

Metadata Summary:
{metadata_summary}

Sample Values (1 per line):
{sample_text}

Follow this format:

Start
├── Preprocessing Step A
└── Preprocessing Step B
    ├── Sub-step B1
    └── Sub-step B2
"""


def parse_tree_text(tree_text: str):
    """
    Parses an indented tree text (LLM-generated) into nodes and edges for graph rendering.
    Returns:
        nodes (set): Unique node labels
        edges (list): List of (parent, child) tuples
    """
    lines = tree_text.strip().splitlines()
    nodes = set()
    edges = []
    stack = []

    for line in lines:
        label = re.sub(r'[│├└─]+', '', line).strip()
        if not label:
            continue

        depth = len(line) - len(line.lstrip(" │├└─"))

        unique_label = f"{label}_{depth}_{len(nodes)}"
        nodes.add(label)

        while stack and stack[-1][0] >= depth:
            stack.pop()

        if stack:
            parent = stack[-1][1]
            edges.append((parent, label))

        stack.append((depth, label))

    return nodes, edges


def render_pyvis_tree(df: pd.DataFrame, call_llm_func, col: str) -> dict:
    """
    Generates an interactive decision tree HTML using PyVis and LLM-generated structure
    for the selected column in the dataframe.

    Args:
        df (pd.DataFrame): Full dataframe
        call_llm_func (function): Function to query the LLM
        col (str): Name of the column to generate tree for

    Returns:
        dict: {
            "status": "success",
            "html": <rendered_html>,
            "leaves": <list_of_leaf_nodes>
        } or {
            "status": "error",
            "message": <error_message>
        }
    """
    try:
        if col not in df.columns:
            return {"status": "error", "message": f"Column '{col}' not found in dataframe."}

        single_col_df = df[[col]]  

        prompt = generate_tree_prompt(single_col_df)
        tree_text = call_llm_func(prompt)
        if not tree_text.strip():
            return {"status": "error", "message": "LLM returned empty response."}

        nodes, edges = parse_tree_text(tree_text)

        all_sources = set(src for src, _ in edges)
        leaf_nodes = [n for n in nodes if n not in all_sources and n.lower() != "start"]

        net = Network(height="550px", width="100%", directed=True, notebook=False)
        net.barnes_hut()

        for node in nodes:
            shape = "box" if node.lower() in ["start", "final"] else "ellipse"
            color = "#f39c12" if node.lower() == "start" else (
                "#8e44ad" if node.lower() == "final" else "#27ae60"
            )
            net.add_node(node, label=node, color=color, shape=shape)

        for src, dst in edges:
            net.add_edge(src, dst)

        net.set_options('''
        const options = {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        ''')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            tmp_path = tmp_file.name
            net.write_html(tmp_path)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            html = f.read()

        os.remove(tmp_path)

        return {
            "status": "success",
            "html": html,
            "leaves": leaf_nodes
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


#################################################
#### New Logic via streamlit_agraph for Decision Trees ####
##################################################

from streamlit_agraph import Node, Edge
import pandas as pd
import metadata_inference
import re

def generate_tree_to_graph_prompt(df: pd.DataFrame) -> str:
    assert df.shape[1] == 1, "DataFrame must contain exactly one column."
    col = df.columns[0]
    series = df[col]

    total = len(series)
    missing = series.isna().sum()
    unique = series.nunique()
    inferred_type = metadata_inference.infer_column_type(series)

    samples = series.dropna().astype(str).sample(min(10, len(series.dropna())), random_state=42).tolist()
    sample_text = "\n".join(samples)
    if len(sample_text) > 2000:
        sample_text = sample_text[:2000] + "..."

    metadata_summary = f"""Type: {inferred_type}
Missing Values: {missing} / {total}
Unique Values: {unique}
Total Rows: {total}
"""

    return f"""
You are a data preprocessing assistant.

Below is the metadata and sample values for a specific column. Based on this concrete data, suggest the most appropriate cleaning or transformation actions using a branching tree format.

Important:
- DO NOT use conditional logic
- DO NOT use explanations or general rules
- Use indentation and branch symbols to define a clear tree

Column Name: {col}

Metadata Summary:
{metadata_summary}

Sample Values (1 per line):
{sample_text}

Follow this format:

Start
├── Preprocessing Step A
└── Preprocessing Step B
    ├── Sub-step B1
    └── Sub-step B2
"""

def parse_tree_to_graph(tree_text: str):
    lines = tree_text.strip().splitlines()
    edges = []
    nodes = []
    seen = set()
    stack = []

    for line in lines:
        label = re.sub(r'[│├└─]+', '', line).strip()
        if not label:
            continue
        depth = len(line) - len(line.lstrip(" │├└─"))

        node_id = f"{label}_{depth}_{len(nodes)}"
        if label not in seen:
            nodes.append(Node(id=label, label=label, size=25))
            seen.add(label)

        while stack and stack[-1][0] >= depth:
            stack.pop()

        if stack:
            parent_label = stack[-1][1]
            edges.append(Edge(source=parent_label, target=label))

        stack.append((depth, label))

    leaf_nodes = [n.label for n in nodes if all(e.source != n.label for e in edges)]
    return nodes, edges, leaf_nodes

def render_agraph_tree(df: pd.DataFrame, call_llm_func, col: str) -> dict:
    try:
        if col not in df.columns:
            return {"status": "error", "message": f"Column '{col}' not found."}

        prompt = generate_tree_to_graph_prompt(df[[col]])
        tree_text = call_llm_func(prompt)
        if not tree_text.strip():
            return {"status": "error", "message": "LLM returned empty response."}

        nodes, edges, leaves = parse_tree_to_graph(tree_text)
        return {
            "status": "success",
            "nodes": nodes,
            "edges": edges,
            "leaves": leaves
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
