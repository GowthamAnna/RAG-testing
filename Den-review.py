# Den_review.py — hardened & updated
# ------------------------------------------------------------
# - Unpacks (rows, meta) from format_responses_with_meta correctly
# - Never mixes list and dict when building responses
# - Defensive token accounting and section review handling
# - Robust Excel export using a template (keeps styles if present)
# - Always returns a dict: {"output_excel": BytesIO, "tokens": {...}, "pointout_count": int}
# ------------------------------------------------------------

from __future__ import annotations

import io
import re
from copy import copy
from typing import Dict, List, Tuple, Any

import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from dotenv import load_dotenv

# ---- LLMClient import: try common local layouts gracefully ----
try:
    # e.g., project/clients/llm_client.py -> class LLMClient
    from clients.llm_client import LLMClient  # type: ignore
except Exception:
    try:
        # e.g., project/clients/llm.py -> class LLMClient
        from clients.llm import LLMClient  # type: ignore
    except Exception:
        try:
            # legacy path some codebases use
            from mdr.client.llm import LLMClient  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import LLMClient. Adjust the import path at the top of Den_review.py."
            ) from e

# These modules are assumed to exist in your project
# - format_document_data: read_until_title, split_text_by_size, Section_chunks
# - format_dataframe: custom_dataframe, filter_dataframe
import format_document_data
import format_dataframe


# ----------------------- Utilities -----------------------

def read_markdown_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def sanitize_string(value: Any) -> Any:
    """Remove control chars that break Excel cells."""
    if isinstance(value, str):
        return re.sub(r"[\x00-\x1F]", "", value)
    return value


def save_to_excel(
    df: pd.DataFrame,
    template_path: str,
    specific_strings_json: Dict[str, Any],
    before_sheet_name: str = "レビュー結果_除外前",
    main_sheet_name: str = "レビュー結果",
) -> Tuple[io.BytesIO, int]:
    """
    Writes df -> Excel using template_path while preserving cell styles
    in the second row as a style source for all rows.
    Also writes a filtered version based on specific_strings_json.
    Returns (excel_bytes, pointout_count_of_filtered_df)
    """
    output_excel = io.BytesIO()
    wb = openpyxl.load_workbook(template_path)

    # Build filtered df via your rule set
    filtered_df = format_dataframe.filter_dataframe(df, specific_strings_json)

    # Optional "before exclusion" sheet (full df)
    if before_sheet_name in wb.sheetnames:
        ws_before = wb[before_sheet_name]
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=False), start=2):
            for c_idx, value in enumerate(row, start=1):
                new_cell = ws_before.cell(row=r_idx, column=c_idx, value=sanitize_string(value))
                base_cell = ws_before.cell(row=2, column=c_idx)
                if base_cell.has_style:
                    new_cell._style = copy(base_cell._style)

    # Main output sheet (filtered)
    if main_sheet_name not in wb.sheetnames:
        wb.create_sheet(main_sheet_name)
    ws = wb[main_sheet_name]
    for r_idx, row in enumerate(dataframe_to_rows(filtered_df, index=False, header=False), start=2):
        for c_idx, value in enumerate(row, start=1):
            new_cell = ws.cell(row=r_idx, column=c_idx, value=sanitize_string(value))
            base_cell = ws.cell(row=2, column=c_idx)
            if base_cell.has_style:
                new_cell._style = copy(base_cell._style)

    pointout_count = len(filtered_df)

    wb.save(output_excel)
    output_excel.seek(0)
    return output_excel, pointout_count


# ----------------------- Main entry -----------------------

def create_AI_review(
    system_message_chat_conversation: str,
    markdown_txt: str,
    *,
    chunk_size_kb: int = 5,
    excel_template: str = "Den_temp.xlsx",
    specific_strings_json: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Runs a two-phase review:
      1) Pre-title text in ~chunk_size_kb chunks via LLM
      2) Section-based review via Section_chunks()
    Aggregates rows, numbers them, builds a DataFrame, and exports Excel.

    Returns:
      {
        "output_excel": BytesIO,
        "tokens": {"total_tokens": int, "prompt_tokens": int, "completion_tokens": int},
        "pointout_count": int
      }
    """
    print("create_AI_review function was called in Den_review file")
    load_dotenv()

    if specific_strings_json is None:
        # tweak this dict to your filtering rules
        specific_strings_json = {"domain_terms": ["半角カナ", "全角カナ"]}

    # -------- Phase 0: prep input --------
    try:
        text_before_title = format_document_data.read_until_title(markdown_txt)
    except Exception as e:
        print("Error in read_until_title:", e)
        # Fallback: process entire text
        text_before_title = markdown_txt

    try:
        text_chunks = format_document_data.split_text_by_size(text_before_title, chunk_size_kb)
    except Exception as e:
        print("Error in split_text_by_size:", e)
        # Fallback: one chunk = original text
        text_chunks = [text_before_title]

    # -------- Phase 1: per-chunk LLM review --------
    all_pre_rows: List[Dict[str, Any]] = []
    AllNo = 1

    pre_total_tokens = 0
    pre_prompt_tokens = 0
    pre_completion_tokens = 0

    x = 1
    for text_chunk in text_chunks:
        try:
            llm = LLMClient(model="gpt-4.1", temperature=0.0)
            data = llm.generate_review(system_message_chat_conversation, text_chunk)

            # IMPORTANT: unpack (rows_list, meta_dict)
            rows, _meta = llm.format_responses_with_meta(
                result_payload=data,
                all_no=AllNo,
                chapter_titles=text_chunk,
            )

            # Only extend with row dicts
            if isinstance(rows, list):
                all_pre_rows.extend(rows)
                AllNo += len(rows)
            else:
                print("format_responses_with_meta returned non-list rows; skipping this chunk.")

            # Defensive tokens accounting
            toks = data.get("tokens", {}) if isinstance(data, dict) else {}
            pre_total_tokens += int(toks.get("total_tokens", 0) or 0)
            pre_prompt_tokens += int(toks.get("prompt_tokens", 0) or 0)
            pre_completion_tokens += int(toks.get("completion_tokens", 0) or 0)

        except Exception as e:
            print("Exception in the create_AI_review loop (Den_review.py):", e)

        print("x is ----------------------------------------------------------", x)
        x += 1

    # -------- Phase 2: section-based review --------
    try:
        section_result = format_document_data.Section_chunks(
            markdown_txt, system_message_chat_conversation
        )
        # Expected: {"all_responses": [row, ...], "tokens": {...}}
        if not isinstance(section_result, dict):
            raise ValueError("Section_chunks returned non-dict.")
    except Exception as e:
        print("error in the Section_chunks function:", e)
        section_result = {"all_responses": [], "tokens": {}}

    section_responses = section_result.get("all_responses", [])
    if not isinstance(section_responses, list):
        print("Warning: section_responses is not a list; forcing empty.")
        section_responses = []

    tokens_info = section_result.get("tokens", {})
    if not isinstance(tokens_info, dict):
        tokens_info = {}

    total_tokens = pre_total_tokens + int(tokens_info.get("section_total_tokens", 0) or 0)
    prompt_tokens = pre_prompt_tokens + int(tokens_info.get("section_prompt_tokens", 0) or 0)
    completion_tokens = pre_completion_tokens + int(tokens_info.get("section_completion_tokens", 0) or 0)

    # Continue numbering for section rows
    last_no = all_pre_rows[-1]["All_No"] if all_pre_rows and "All_No" in all_pre_rows[-1] else 0
    for r in section_responses:
        if isinstance(r, dict):
            last_no += 1
            r["All_No"] = last_no

    final_responses: List[Dict[str, Any]] = all_pre_rows + [r for r in section_responses if isinstance(r, dict)]

    print(type(final_responses), "Type of final_responses:\n")
    print("Final responses:\n", final_responses)

    # -------- DataFrame build (defensive) --------
    try:
        df = format_dataframe.custom_dataframe(final_responses)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("custom_dataframe did not return a pandas DataFrame.")
    except Exception as e:
        print("Error in custom_dataframe function:", e)
        # Fallback minimal DF
        df = pd.DataFrame(final_responses)

    # -------- Excel export (defensive) --------
    try:
        output_excel, pointout_count = save_to_excel(
            df=df,
            template_path=excel_template,
            specific_strings_json=specific_strings_json,
        )
    except Exception as e:
        print("Error in the save_to_excel function:", e)
        output_excel, pointout_count = io.BytesIO(), 0

    print("create_AI_review function ended")

    return {
        "output_excel": output_excel,
        "tokens": {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "pointout_count": pointout_count,
    }
