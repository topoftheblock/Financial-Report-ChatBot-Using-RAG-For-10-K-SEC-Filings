# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:22:52 2026

@author: patri

Parses the .html file and converts them into a markdown format
"""

# %% Libraries

from pathlib import Path
import os

import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, NavigableString

# %% Macros
BASE_DIR = "C:/Users/patri/Desktop/Financial-Report-ChatBot-Using-RAG/"

# %% Parser
 
# ── Regex patterns ────────────────────────────────────────────────────────────
_PART_RE = re.compile(r'^PART\s+(I{1,3}V?|IV)$',   re.IGNORECASE)
_ITEM_RE = re.compile(r'^Item\s+\d+[A-C]?\.\s+\S', re.IGNORECASE)
_BOLD_RE = re.compile(r'font-weight\s*:\s*(700|bold)', re.IGNORECASE)

def _clean(text: str) -> str:
    """Normalise whitespace, including HTML non-breaking spaces (\\xa0)."""
    return re.sub(r'[\s\u00a0]+', ' ', text).strip()
 
 
def _is_bold_span(tag) -> bool:
    return tag.name == 'span' and bool(_BOLD_RE.search(tag.get('style', '')))

# ── Heading detection ─────────────────────────────────────────────────────────
def _mark_headings(soup) -> None:
    """
    Walk every <div> that is NOT inside a <table>.
    Replace matching divs with Markdown heading placeholders.
 
        #   → PART I / PART II / PART III / PART IV
        ##  → Item 1. Business  /  Item 7A. Market Risk  / …
        ### → Any other standalone all-bold div (sub-section titles)
 
    Processing order is document order (outer → inner).
    The `elem.parent is not None` guard skips divs that were already
    removed because they were nested inside a previously-replaced div.
    """
    candidates = []  # [(Tag, markdown_string), …]
 
    for div in soup.find_all('div'):
        # ── Skip anything that lives inside a table ───────────────────────────
        # This excludes Table-of-Contents entries and financial-table captions,
        # which are structurally identical to real headings but should not be
        # promoted to # / ## markers.
        if div.find_parent('table'):
            continue
 
        div_text = _clean(div.get_text())
        if not div_text or len(div_text) > 200:
            continue
 
        bold_spans = [s for s in div.find_all('span') if _is_bold_span(s)]
        if not bold_spans:
            continue
 
        combined_bold = _clean(' '.join(s.get_text() for s in bold_spans))
 
        if _PART_RE.match(combined_bold):
            candidates.append((div, f'\n\n# {combined_bold}\n\n'))
 
        elif _ITEM_RE.match(combined_bold):
            candidates.append((div, f'\n\n## {combined_bold}\n\n'))
 
        elif combined_bold == div_text and len(combined_bold) < 150:
            # The div's entire visible text is bold → sub-section heading.
            # The `== div_text` guard prevents inline bold phrases from being
            # promoted; only standalone bold divs (no surrounding plain text)
            # qualify.
            candidates.append((div, f'\n\n### {combined_bold}\n\n'))
 
    # Apply replacements.  The `elem.parent is not None` check is essential:
    # find_all returns outer divs before inner divs (document order), so when
    # an outer div is replaced, its inner divs are detached from the tree.
    # Attempting to replace a detached element corrupts the parse tree.
    seen: set = set()
    for elem, md in candidates:
        if id(elem) not in seen and elem.parent is not None:
            elem.replace_with(NavigableString(md))
            seen.add(id(elem))


# ── Table processing ──────────────────────────────────────────────────────────
def _process_table(table):
    try:
        # ---------------------------------------------------------
        # STEP 1A: PARSE & INITIAL CLEANUP
        # ---------------------------------------------------------
        # Use pandas to parse the HTML table
        df_list = pd.read_html(str(table))
        if df_list:
            df = df_list[0]
            
            # Drop rows/columns that are entirely NaN
            df.dropna(how="all", inplace=True)
            df.dropna(how="all", axis=1, inplace=True)
            
            df = df.reset_index(drop=True)
            
            # ---------------------------------------------------------
            # STEP 1B: DROP SPANNING HEADER ROWS
            # ---------------------------------------------------------
            # A spanning header row has at most ONE unique non-null value
            # (the colspan label repeated across columns, e.g. "Years ended").
            # TWO conditions must BOTH be true to drop:
            #   1. At most 1 unique non-null value in the row
            #   2. The first cell is NaN (real data rows always have a first-column label)
            # Exception: single-column tables are never spanning headers.
            
            if len(df.columns) > 1:
                while len(df) > 0:
                    first_cell = df.iloc[0, 0]
                    first_cell_empty = pd.isna(first_cell) or str(first_cell).strip() in ('', 'nan', 'None')
                    unique_vals = df.iloc[0].dropna().astype(str).str.strip().unique()
                    if len(unique_vals) <= 1 and first_cell_empty:
                        df = df.drop(index=df.index[0]).reset_index(drop=True)
                    else:
                        break
            
            # ---------------------------------------------------------
            # STEP 2: PROMOTE FIRST ROW TO HEADER
            # ---------------------------------------------------------
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            
            # Rename any 'nan' or missing column headers to 'col1', 'col2', etc.
            new_columns = []
            nan_counter = 1
            for col in df.columns:
                # Check if the column name is naturally missing (NaN) or the string 'nan'
                if pd.isna(col) or str(col).strip().lower() == 'nan':
                    new_columns.append(f"col{nan_counter}")
                    nan_counter += 1
                else:
                    new_columns.append(col)
            
            # Apply the cleaned up column names back to the dataframe
            df.columns = new_columns
            
            # Drop empty columns
            df.dropna(how="all", axis = 1, inplace=True)
                            
            # Replace empty strings or pure whitespace with NaN to make logic easier
            df = df.replace(r'^\s*$', np.nan, regex=True)
            
            # ---------------------------------------------------------
            # STEP 3A: MERGE MULTI-ROW HEADERS
            # ---------------------------------------------------------
            # Check if the first column is named "col{i}" (e.g., "col1", "col2")
            first_col_name = str(df.columns[0]).strip()
            is_col_i = first_col_name.startswith('col') and first_col_name[3:].isdigit()

            # Only run the multi-row header merge if the first column is NOT 'col{i}'
            if not is_col_i:
                rows_to_drop = []

                # Take a static snapshot without duplicate columns for the condition check.
                # This prevents the loop's text modifications from breaking the duplicate detection.
                df_snapshot = df.loc[:, ~df.T.duplicated()]

                # Loop through the dataframe, stopping right before the last row
                for i in range(len(df) - 1):
                    # 1. Check the condition against the stable SNAPSHOT
                    if pd.notna(df.iloc[i, 0]) and df_snapshot.iloc[i, 1:].isna().all():
                        
                        # 2. Pull text and apply modifications to the ACTUAL dataframe
                        # (Because we pull from 'df', consecutive headers will still cascade correctly)
                        prefix_text = str(df.iloc[i, 0]).strip()
                        df.iloc[i+1, 0] = f"{prefix_text} {str(df.iloc[i+1, 0]).strip()}"
                        
                        # Mark this row for deletion
                        rows_to_drop.append(i)
                
                # Drop all the identified header rows at once
                df = df.drop(index=rows_to_drop).reset_index(drop=True)
            
            # Drop all Empty columns
            df = df.dropna(axis=1, how='all')
            
            # REMOVE ADJACENT DUPLICATES OF THE FIRST COLUMN

            first_col_name = df.columns[0]
            
            # We will build a list of column indices to keep. We ALWAYS keep index 0.
            keep_indices = [0] 
            
            # Loop through the remaining column positions
            for i in range(1, len(df.columns)):
                if df.columns[i] == first_col_name:
                    # It's an adjacent duplicate of the first column, so we skip it (drop it)
                    continue 
                else:
                    # We hit a different column name! 
                    # Keep this column and all remaining columns, then stop checking.
                    keep_indices.extend(range(i, len(df.columns)))
                    break
            
            # Slicing by iloc safely drops the adjacent duplicates without touching 
            # columns further down the table that might share a name.
            df = df.iloc[:, keep_indices]
            
            # ---------------------------------------------------------
            # STEP 3B: MERGE COLON-BASED MULTI-ROW HEADERS
            # ---------------------------------------------------------
            active_prefix = None
            rows_to_drop = []
            
            for i in range(len(df)):
                first_col_val = str(df.iloc[i, 0]).strip()
            
                if first_col_val in ('nan', 'None', ''):
                    continue
            
                # ── THE ENDER ──────────────────────────────────────────────────────────
                # A "Total …" row closes the active block.  We stop augmenting but keep
                # the row as-is (it already reads "Total net sales", which is clear).
                if active_prefix and first_col_val.lower().startswith('total'):
                    active_prefix = None
                    continue
            
                # ── HELPER: is the rest of this row "empty"? ───────────────────────────
                # Treat a cell as empty if it is NaN *or* if it is a rowspan-duplicate
                # of the first column value (the HTML parser repeats it across columns).
                rest = df.iloc[i, 1:]
                rest_meaningful = rest.dropna().astype(str).str.strip()
                rest_meaningful = rest_meaningful[~rest_meaningful.isin(('', first_col_val))]
                row_is_label_only = rest_meaningful.empty
            
                # ── THE TRIGGER ────────────────────────────────────────────────────────
                if first_col_val.endswith(':') and row_is_label_only:
                    active_prefix = first_col_val[:-1].strip()   # strip the colon
                    rows_to_drop.append(i)
                    continue
            
                # ── THE AUGMENTATION ───────────────────────────────────────────────────
                if active_prefix:
                    df.iloc[i, 0] = f"{active_prefix}: {first_col_val}"
            
            if rows_to_drop:
                df = df.drop(index=rows_to_drop).reset_index(drop=True)
                
            # ---------------------------------------------------------
            # STEP 3C: DROP COLUMNS REDUNDANT TO THE LABEL COLUMN
            # ---------------------------------------------------------
            # A column is redundant if every non-null value it contains
            # is a substring of (or equal to) the corresponding col1 value.
            # This catches rowspan-duplicate columns like col2/col3 which
            # hold the short label ("Products") while col1 holds the full
            # augmented label ("Net sales: Products").

            label_col = df.iloc[:, 0].astype(str).str.strip()
            cols_to_drop = []

            for col_idx in range(1, len(df.columns)):
                col_vals = df.iloc[:, col_idx]
                
                # Only examine non-null entries
                non_null_mask = col_vals.notna() & (col_vals.astype(str).str.strip() != '')
                
                if not non_null_mask.any():
                    continue  # fully empty column — Step 6 will handle it
                
                col_non_null = col_vals[non_null_mask].astype(str).str.strip()
                label_non_null = label_col[non_null_mask]
                
                # Check: is every value in this column a substring of col1?
                is_redundant = all(
                    cell in label_val
                    for cell, label_val in zip(col_non_null, label_non_null)
                )
                
                if is_redundant:
                    cols_to_drop.append(df.columns[col_idx])

            df = df.drop(columns=cols_to_drop)
            
            # ---------------------------------------------------------
            # STEP 4: CELL-BY-CELL CONDENSER (AMENDED)
            # ---------------------------------------------------------
            # Iterate RIGHT-TO-LEFT to merge $ and %, and eliminate duplicates.
            # By collapsing everything to the LEFT, we ensure that both the header 
            # (e.g., 'Adjusted Cost') and the data (e.g., '$28267') anchor 
            # themselves in the exact same column index, fixing the misalignment.
            for i in range(len(df)):
                for j in range(len(df.columns) - 2, -1, -1):
                    left_val = str(df.iloc[i, j]).strip()
                    right_val = str(df.iloc[i, j+1]).strip()
                    
                    # Skip empty cells
                    if left_val in ['nan', 'None', '']:
                        continue
        
                    # A. Merge Dollar Signs Left
                    if left_val == '$':
                        if right_val not in ['nan', 'None', '']:
                            df.iloc[i, j] = f"${right_val}"
                        else:
                            df.iloc[i, j] = np.nan
                        df.iloc[i, j+1] = np.nan
        
                    # B. Merge Percentages Left
                    elif right_val == '%':
                        if left_val not in ['nan', 'None', '']:
                            df.iloc[i, j] = f"{left_val}%"
                        df.iloc[i, j+1] = np.nan
        
                    # C. Eliminate Adjacent Duplicates
                    else:
                        is_duplicate = False
                        
                        # Check for exact string matches
                        if left_val == right_val:
                            is_duplicate = True
                        else:
                            # Check for numerical equivalents
                            try:
                                def clean_num(v):
                                    v = v.replace(',', '').replace('$', '').replace('%', '').strip()
                                    if v.startswith('(') and v.endswith(')'):
                                        v = '-' + v[1:-1]
                                    return float(v)
                                
                                if clean_num(left_val) == clean_num(right_val):
                                    is_duplicate = True
                            except Exception:
                                pass
                        
                        # WIPE THE RIGHT ONE to keep data anchored to the left
                        if is_duplicate:
                            df.iloc[i, j+1] = np.nan
            
            # ---------------------------------------------------------
            # STEP 6: FINAL CLEANUP & MARKDOWN CONVERSION
            # ---------------------------------------------------------
            
            # Drop all Empty columns
            df = df.dropna(axis=1, how='all')
            
            # Drop all duplicate columns (FIXED)
            # By resetting the index on the transposed dataframe, we force Pandas to include the column 
            # headers in its duplication check. This prevents the accidental dropping of distinct metrics 
            # that just happen to share identical values (e.g., columns that are all dashes).
            df_t = df.T.reset_index()
            duplicated_mask = df_t.duplicated().values
            df = df.loc[:, ~duplicated_mask]
            
            return df.to_markdown(index=False) if not df.empty else None
    except Exception:
        # =========================================================
        # FALLBACK: TEXT EXTRACTION ON FAILURE
        # =========================================================
            
        # Fallback: extract text cleanly if pandas fails
        fb = table.get_text(separator=' | ', strip=True)
        return fb if fb.strip() else None
    
    
# ── Main parser ───────────────────────────────────────────────────────────────
def parse_10k_html(file_path: str, output_path: str) -> str:
    """
    Parse a SEC 10-K .html file and write a clean Markdown file.
 
    Parameters
    ----------
    file_path   : path to the raw .html filing
    output_path : where to write the .md / .txt result
    """
    
    if not file_path.lower().endswith('.html'):
        raise ValueError("Error: The file you have given is not a .html file")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()
 
    # Strip XML declarations (common in EDGAR filings)
    html = re.sub(r'<\?xml[^>]*\?>', '', html, flags=re.IGNORECASE)
 
    # ── 1. Parse FIRST ────────────────────────────────────────────────────────
    # Critical change vs v1: we parse the complete HTML before any truncation.
    # The original code sliced the raw string at ">Part I<" — which lands in the
    # middle of a closing tag — so BeautifulSoup received malformed HTML and
    # silently dropped adjacent text nodes.
    soup = BeautifulSoup(html, 'html.parser')
 
    for tag in soup(['script', 'style', 'meta', 'noscript', 'link', 'head', 'title']):
        tag.decompose()
 
    # ── 2. Mark headings ──────────────────────────────────────────────────────
    # Must happen BEFORE table processing, otherwise table cells that happen to
    # be bold/short could be misclassified as headings.
    _mark_headings(soup)
 
    # ── 3. Convert tables to Markdown ─────────────────────────────────────────
    for table in soup.find_all('table'):
        md = _process_table(table)
        if md:
            table.replace_with(NavigableString(f'\n\n{md}\n\n'))
        else:
            table.decompose()
 
    # ── 4. Extract plain text ─────────────────────────────────────────────────
    text = soup.get_text(separator='\n')
 
    # ── 5a. Truncate before Part I ─────────────────────────────────────────────
    # Because we parsed the full document, the Table of Contents is still present.
    # We drop everything up to the first "# PART I" heading that _mark_headings
    # inserted.  The TOC headings are inside <table> tags and were skipped by
    # _mark_headings, so the first match is always the real content heading.
    m = re.search(r'(?m)^#\s+PART\s+I\b', text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    else:
        # Fallback for filings where the heading wasn't detected
        m = re.search(r'(?m)^\s*PART\s+I\s*$', text, re.IGNORECASE)
        if m:
            text = text[m.start():]
            
    # ── 5b. Remove page numbers joined with Table of Contents ─────────────────
    # Matches a line with just a number, followed by "Table of Contents" 
    # OR "Table of \n Contents" (spanning multiple lines).
    text = re.sub(r'(?im)^[ \t]*\d+[ \t]*\n[ \t]*Table[ \t]+of[ \t]*(?:\n[ \t]*)?Con?tents?[ \t]*\n?', '', text)

 
    # ── 6. Normalise whitespace ───────────────────────────────────────────────
    text = re.sub(r'[ \t]+', ' ', text)       # collapse inline spaces/tabs
    text = re.sub(r'\n{3,}', '\n\n', text)    # max two consecutive blank lines
    text = text.strip()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    with open(output_path + "/10-K.txt", 'w', encoding='utf-8') as f:
        f.write(text)
 
    print(f'Done → {output_path}')
    return #text

# %% Loop over .html documents to parse

# Get Companies
path = os.path.join(BASE_DIR, "data/raw/")
companies = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

for company in companies:
    # Get the number of years
    year_path = os.path.join(path, f"{company}/")
    years = [y for y in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, y))]
    
    for year in years:
        # Get the number of reports for each year
        report_path = os.path.join(year_path, f"{year}/")
        reports = [r for r in os.listdir(report_path)]
        
        for report in reports: 
            # Parse each .html file
            file_path = os.path.join(report_path, f"{report}")
            parse_10k_html(file_path, os.path.join(BASE_DIR, f"data/processed/{company}/{year}/"))
