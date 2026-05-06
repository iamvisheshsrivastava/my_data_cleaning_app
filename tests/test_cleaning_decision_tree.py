"""
Unit tests for cleaningDecisionTree.py
========================================
Covers tree-parsing and prompt-generation logic.  LLM/HTTP calls and
Streamlit-agraph imports that depend on the browser are patched out so all
tests run fully offline.

Run with:
    pytest tests/test_cleaning_decision_tree.py -v
"""

import sys
import os

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch metadata_inference.infer_column_type before importing the module
# so no real LLM/HTTP call is triggered during tests.
with patch("metadata_inference.infer_column_type", return_value="Numerical"):
    from cleaningDecisionTree import (
        parse_tree_text,
        parse_tree_to_graph,
        generate_tree_prompt,
        generate_tree_to_graph_prompt,
    )


# ---------------------------------------------------------------------------
# Shared sample tree text
# ---------------------------------------------------------------------------

SAMPLE_TREE = """\
Start
├── Impute missing values with mean
└── Normalize to 0-1 range
    ├── Apply MinMaxScaler
    └── Clip values above 1
"""

FLAT_TREE = """\
Start
├── Step A
└── Step B
"""


# ---------------------------------------------------------------------------
# parse_tree_text
# ---------------------------------------------------------------------------

class TestParseTreeText:
    """Tests for parse_tree_text() — converts indented tree text into sets of
    nodes and (parent, child) edge pairs."""

    def test_returns_nodes_and_edges(self):
        nodes, edges = parse_tree_text(SAMPLE_TREE)
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_start_node_is_present(self):
        nodes, _ = parse_tree_text(SAMPLE_TREE)
        assert "Start" in nodes

    def test_start_is_a_parent_node(self):
        _, edges = parse_tree_text(SAMPLE_TREE)
        parents = {src for src, _ in edges}
        assert "Start" in parents

    def test_leaf_nodes_are_not_parents(self):
        nodes, edges = parse_tree_text(SAMPLE_TREE)
        parents = {src for src, _ in edges}
        # "Apply MinMaxScaler" and "Clip values above 1" should be leaves
        assert "Apply MinMaxScaler" not in parents
        assert "Clip values above 1" not in parents

    def test_empty_string_returns_empty_collections(self):
        nodes, edges = parse_tree_text("")
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_single_node_no_edges(self):
        nodes, edges = parse_tree_text("Start")
        assert "Start" in nodes
        assert len(edges) == 0

    def test_flat_two_branch_tree(self):
        nodes, edges = parse_tree_text(FLAT_TREE)
        assert "Start" in nodes
        assert ("Start", "Step A") in edges
        assert ("Start", "Step B") in edges

    def test_all_edge_nodes_exist_in_node_set(self):
        nodes, edges = parse_tree_text(SAMPLE_TREE)
        for src, dst in edges:
            assert src in nodes, f"Source '{src}' missing from nodes"
            assert dst in nodes, f"Target '{dst}' missing from nodes"


# ---------------------------------------------------------------------------
# parse_tree_to_graph
# ---------------------------------------------------------------------------

class TestParseTreeToGraph:
    """Tests for parse_tree_to_graph() — same parsing but returns streamlit-agraph
    Node/Edge objects instead of plain strings."""

    def test_returns_nodes_edges_leaves(self):
        nodes, edges, leaves = parse_tree_to_graph(SAMPLE_TREE)
        assert len(nodes) > 0
        assert len(edges) > 0
        assert len(leaves) > 0

    def test_leaf_nodes_are_not_edge_sources(self):
        nodes, edges, leaves = parse_tree_to_graph(SAMPLE_TREE)
        edge_sources = {e.source for e in edges}
        for leaf in leaves:
            assert leaf not in edge_sources, f"Leaf '{leaf}' is also an edge source"

    def test_no_duplicate_node_ids(self):
        nodes, _, _ = parse_tree_to_graph(SAMPLE_TREE)
        ids = [n.id for n in nodes]
        assert len(ids) == len(set(ids)), "Duplicate node IDs detected"

    def test_all_edge_endpoints_reference_known_nodes(self):
        nodes, edges, _ = parse_tree_to_graph(SAMPLE_TREE)
        known_labels = {n.label for n in nodes}
        for edge in edges:
            assert edge.source in known_labels, f"Edge source '{edge.source}' not in nodes"
            assert edge.target in known_labels, f"Edge target '{edge.target}' not in nodes"

    def test_empty_string_returns_empty_collections(self):
        nodes, edges, leaves = parse_tree_to_graph("")
        assert nodes == []
        assert edges == []
        assert leaves == []

    def test_flat_two_branch_tree_has_two_leaves(self):
        nodes, edges, leaves = parse_tree_to_graph(FLAT_TREE)
        assert len(leaves) == 2
        assert set(leaves) == {"Step A", "Step B"}


# ---------------------------------------------------------------------------
# generate_tree_prompt
# ---------------------------------------------------------------------------

class TestGenerateTreePrompt:
    """Tests for generate_tree_prompt() — builds the LLM prompt from a single-
    column DataFrame."""

    @patch("metadata_inference.infer_column_type", return_value="Numerical")
    def test_prompt_contains_column_name(self, _mock):
        df = pd.DataFrame({"age": [25, 30, 35, np.nan, 40]})
        prompt = generate_tree_prompt(df)
        assert "age" in prompt

    @patch("metadata_inference.infer_column_type", return_value="Numerical")
    def test_prompt_contains_missing_value_info(self, _mock):
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0, np.nan, 5.0]})
        prompt = generate_tree_prompt(df)
        assert "Missing" in prompt

    @patch("metadata_inference.infer_column_type", return_value="Numerical")
    def test_prompt_contains_start_format_example(self, _mock):
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
        prompt = generate_tree_prompt(df)
        assert "Start" in prompt

    @patch("metadata_inference.infer_column_type", return_value="Categorical")
    def test_prompt_contains_inferred_type(self, _mock):
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        prompt = generate_tree_prompt(df)
        assert "Categorical" in prompt

    def test_raises_assertion_for_multi_column_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(AssertionError):
            generate_tree_prompt(df)

    @patch("metadata_inference.infer_column_type", return_value="Numerical")
    def test_prompt_is_a_non_empty_string(self, _mock):
        df = pd.DataFrame({"x": [1, 2, 3]})
        prompt = generate_tree_prompt(df)
        assert isinstance(prompt, str) and len(prompt) > 0


# ---------------------------------------------------------------------------
# generate_tree_to_graph_prompt
# ---------------------------------------------------------------------------

class TestGenerateTreeToGraphPrompt:
    """Tests for generate_tree_to_graph_prompt() — the agraph variant of the
    prompt builder (same contract, slightly different instructions)."""

    @patch("metadata_inference.infer_column_type", return_value="Numerical")
    def test_prompt_contains_column_name(self, _mock):
        df = pd.DataFrame({"price": [9.99, 14.99, 4.49]})
        prompt = generate_tree_to_graph_prompt(df)
        assert "price" in prompt

    def test_raises_assertion_for_multi_column_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(AssertionError):
            generate_tree_to_graph_prompt(df)

    @patch("metadata_inference.infer_column_type", return_value="Text")
    def test_prompt_is_a_non_empty_string(self, _mock):
        df = pd.DataFrame({"description": ["foo", "bar", "baz"]})
        prompt = generate_tree_to_graph_prompt(df)
        assert isinstance(prompt, str) and len(prompt) > 0
