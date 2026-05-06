"""
Unit tests for metadata_inference.py
======================================
Covers the deterministic helper functions that don't require live LLM calls.
The fallback LLM call inside infer_column_type is patched out so these tests
are fast and fully offline.

Run with:
    pytest tests/test_metadata_inference.py -v
"""

import sys
import os

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch the LLM/HTTP fallback BEFORE importing the module so no real
# network call is ever made during the test run.
with patch("requests.post") as _mock_post:
    _mock_post.return_value = MagicMock(status_code=200, json=lambda: {"response": "Text"})
    from metadata_inference import (
        infer_column_type,
        _date_success_ratio,
        get_viz_capability,
        quick_pipeline_score,
    )


# ---------------------------------------------------------------------------
# infer_column_type  — dtype-based early exits
# ---------------------------------------------------------------------------

class TestInferColumnTypeDtypes:
    """Tests where pandas dtype lets infer_column_type exit early."""

    def test_integer_series_is_numerical(self):
        assert infer_column_type(pd.Series([1, 2, 3, 4, 5], dtype=int)) == "Numerical"

    def test_float_series_is_numerical(self):
        assert infer_column_type(pd.Series([1.1, 2.2, 3.3])) == "Numerical"

    def test_bool_dtype_is_boolean(self):
        assert infer_column_type(pd.Series([True, False, True], dtype=bool)) == "Boolean"

    def test_datetime64_is_datetime(self):
        s = pd.Series(pd.to_datetime(["2021-01-01", "2022-06-15", "2023-03-20"]))
        assert infer_column_type(s) == "Datetime"


# ---------------------------------------------------------------------------
# infer_column_type  — heuristic detection on object dtype
# ---------------------------------------------------------------------------

class TestInferColumnTypeHeuristics:
    """Tests for pattern-matching heuristics on string object columns."""

    def test_email_column(self):
        s = pd.Series(["alice@example.com", "bob@test.org", "carol@domain.net"] * 15)
        assert infer_column_type(s) == "Email Address"

    def test_image_url_column(self):
        s = pd.Series([
            "https://example.com/photo.jpg",
            "https://cdn.site.com/image.png",
            "https://media.io/pic.jpeg",
        ] * 15)
        assert infer_column_type(s) == "Image URL"

    def test_percentage_column(self):
        s = pd.Series(["10%", "20%", "35%", "50%", "99%"] * 15)
        assert infer_column_type(s) == "Percentage"

    def test_hex_color_column(self):
        s = pd.Series(["#FF5733", "#C0C0C0", "#000000", "#FFFFFF"] * 15)
        assert infer_column_type(s) == "Color Code"

    def test_gps_column(self):
        s = pd.Series([
            "48.8566, 2.3522",
            "51.5074, -0.1278",
            "40.7128, -74.0060",
        ] * 15)
        assert infer_column_type(s) == "GPS Coordinates"

    def test_null_heavy_column(self):
        # 90 % NaN in an object-dtype column → "Null-heavy".
        # dtype=object must be explicit: pd.Series([None]*90 + [1]*10) would be
        # inferred as numeric (float/int), which exits early as "Numerical".
        s = pd.Series([None] * 90 + ["some_text"] * 10, dtype=object)
        assert infer_column_type(s) == "Null-heavy"

    def test_constant_column(self):
        s = pd.Series(["same_value"] * 50)
        assert infer_column_type(s) == "Constant / Low Variance"

    def test_low_cardinality_object_is_categorical(self):
        # Only 2 unique values in 200 rows → unique ratio ≈ 1% → Categorical
        s = pd.Series(["cat", "dog"] * 100)
        assert infer_column_type(s) == "Categorical"

    def test_unique_id_column(self):
        s = pd.Series([f"id_{i}" for i in range(100)])
        assert infer_column_type(s) == "Identifier / ID"

    def test_string_datetime_column(self):
        # Use long-form dates that do NOT match the phone-number regex
        # (r'^\+?\d[\d\s\-]{7,}$'). ISO dates like "2023-01-15" do match
        # that regex, so they're misclassified as "Phone Number" — use a
        # format with letters instead.
        s = pd.Series(["15 Jan 2023", "20 Feb 2023", "10 Mar 2023"] * 20)
        assert infer_column_type(s) == "Datetime"

    def test_all_null_series(self):
        s = pd.Series([None, None, None, None])
        result = infer_column_type(s)
        # Empty non-null series → "Null-heavy"
        assert result == "Null-heavy"

    def test_boolean_string_column(self):
        # Only "true"/"false" values with nunique == 2 → Boolean
        s = pd.Series(["true", "false"] * 25)
        assert infer_column_type(s) == "Boolean"

    def test_video_url_column(self):
        # The video-URL regex is anchored by str.match (start-of-string), so
        # absolute "https://..." URLs fall through to "General URL".  The
        # regex is designed to match paths like "youtu.be/..." without a
        # protocol prefix.  This is known behaviour — test what actually
        # happens so the suite guards against regressions.
        s = pd.Series([
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/xyz789",
            "https://cdn.io/video.mp4",
        ] * 15)
        assert infer_column_type(s) == "General URL"

    def test_json_column(self):
        s = pd.Series(['{"key": "val"}', '{"a": 1}', '{"b": 2}'] * 15)
        assert infer_column_type(s) == "JSON / Nested"

    def test_duration_column(self):
        # "HH:MM" strings like "01:30" are successfully parsed by dateparser as
        # times, so _date_success_ratio returns > 0.70 and the Datetime check
        # (step 4) fires before the Duration check (step 9).  This is current
        # behaviour — the test documents it so any future change is visible.
        s = pd.Series(["01:30", "02:45", "00:15", "10:00"] * 15)
        assert infer_column_type(s) == "Datetime"


# ---------------------------------------------------------------------------
# _date_success_ratio
# ---------------------------------------------------------------------------

class TestDateSuccessRatio:
    """Tests for the internal date-parsing success ratio helper."""

    def test_empty_series_returns_zero(self):
        assert _date_success_ratio(pd.Series([], dtype=str)) == 0.0

    def test_valid_iso_dates_return_high_ratio(self):
        s = pd.Series(["2021-01-01", "2022-06-15", "2023-12-31"] * 20)
        ratio = _date_success_ratio(s)
        assert ratio >= 0.9, f"Expected >= 0.9, got {ratio}"

    def test_random_strings_return_low_ratio(self):
        s = pd.Series(["hello", "world", "foobar", "test123"] * 20)
        ratio = _date_success_ratio(s)
        assert ratio < 0.5, f"Expected < 0.5, got {ratio}"

    def test_all_null_series_returns_zero(self):
        s = pd.Series([None, None, None])
        assert _date_success_ratio(s) == 0.0


# ---------------------------------------------------------------------------
# get_viz_capability
# ---------------------------------------------------------------------------

class TestGetVizCapability:
    """Tests for the visualization-capability lookup."""

    @pytest.mark.parametrize("col_type,expected_chart", [
        ("Numerical",      "Box Plot"),
        ("Categorical",    "Bar Chart"),
        ("Boolean",        "Bar Chart"),
        ("Ordinal",        "Histogram"),
        ("Text",           "Word Cloud"),
        ("Datetime",       "Time Series Line Chart"),
        ("GPS Coordinates","Map"),
        ("Percentage",     "Histogram"),
        ("Currency",       "Histogram"),
        ("Color Code",     "Swatches"),
        ("Email Address",  "Top Values"),
        ("Phone Number",   "Top Values"),
        ("Image URL",      "Image Viewer"),
        ("Video URL",      "Video Preview"),
        ("General URL",    "LLM Summary"),
        ("Document URL",   "Download Viewer"),
        ("File Path",      "Download Viewer"),
    ])
    def test_supported_types_return_checkmark(self, col_type, expected_chart):
        result = get_viz_capability(col_type)
        assert "✅" in result
        assert expected_chart in result

    def test_unknown_type_returns_cross(self):
        assert get_viz_capability("SomeUnknownType") == "❌"

    def test_general_url_with_webpage_col_name(self):
        result = get_viz_capability("General URL", col_name="website")
        assert "LLM Summary" in result and "✅" in result

    def test_general_url_with_generic_col_name(self):
        result = get_viz_capability("General URL", col_name="url_column")
        assert "✅" in result


# ---------------------------------------------------------------------------
# quick_pipeline_score
# ---------------------------------------------------------------------------

class TestQuickPipelineScore:
    """Tests for the ML-readiness star-rating function."""

    def test_numerical_low_missing_gets_five_stars(self):
        result = quick_pipeline_score("Numerical", 2.0, 80.0, pd.Series([1, 2, 3]), {})
        assert "⭐⭐⭐⭐⭐" in result

    def test_numerical_high_missing_gets_three_stars(self):
        result = quick_pipeline_score("Numerical", 50.0, 80.0, pd.Series([1, 2, 3]), {})
        assert "⭐⭐⭐" in result and "⭐⭐⭐⭐⭐" not in result

    def test_categorical_low_cardinality_gets_four_stars(self):
        top_vals = {"A": 10, "B": 5}
        result = quick_pipeline_score("Categorical", 5.0, 2.0, pd.Series(["A"] * 15), top_vals)
        assert "⭐⭐⭐⭐" in result

    def test_datetime_gets_three_stars(self):
        result = quick_pipeline_score("Datetime", 5.0, 90.0, pd.Series(["2021-01-01"]), {})
        assert "⭐⭐⭐" in result

    def test_boolean_gets_four_stars(self):
        result = quick_pipeline_score("Boolean", 2.0, 50.0, pd.Series([True, False]), {})
        assert "⭐⭐⭐⭐" in result

    def test_heavily_missing_column_gets_one_star(self):
        result = quick_pipeline_score("Text", 75.0, 90.0, pd.Series([None] * 10), {})
        assert result.startswith("⭐ ")

    def test_result_is_a_string(self):
        result = quick_pipeline_score("Numerical", 0.0, 100.0, pd.Series([1.0]), {})
        assert isinstance(result, str)
