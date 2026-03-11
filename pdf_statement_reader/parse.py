from tabula import read_pdf
import pdfplumber
import pandas as pd
import numpy as np


def get_raw_df(filename, num_pages, config):
    dfs = []

    for i in range(num_pages):
        if i == 0 and "first" in config["layout"]:
            area = config["layout"]["first"]["area"]
            columns = config["layout"]["first"]["columns"]
        else:
            area = config["layout"]["default"]["area"]
            columns = config["layout"]["default"]["columns"]

        no_header = config.get("no_header", False)
        pandas_opts = {"dtype": str}
        if no_header:
            pandas_opts["header"] = None

        df = read_pdf(
            filename,
            pages=i + 1,
            area=area,
            columns=columns,
            stream=True,
            guess=False,
            pandas_options=pandas_opts,
            java_options=[
                "-Dorg.slf4j.simpleLogger.defaultLogLevel=off",
                "-Dorg.apache.commons.logging.Log=org.apache.commons.logging.impl.NoOpLog",
            ],
            force_subprocess=True,
        )
        if df is not None and len(df) > 0:
            dfs.extend(df)
    statement = pd.concat(dfs, sort=False).reset_index(drop=True)
    return statement


def format_negatives(s):
    s = str(s)
    if s.endswith("-"):
        return "-" + s[:-1]
    else:
        return s


def format_cr_dr(s):
    """Handle FNB-style Cr/Dr suffixes and comma thousands separators.

    - Values ending in 'Cr' are credits (positive)
    - Plain values are debits (negative)
    - Commas are stripped (thousands separator)
    """
    s = str(s).strip()
    if s == "nan" or s == "":
        return s
    s = s.replace(",", "")
    if s.endswith("Cr"):
        return s[:-2]
    elif s.endswith("Dr"):
        return "-" + s[:-2]
    else:
        return "-" + s


def format_r_prefix(s):
    """Handle ABSA-style R prefix amounts (e.g., R398.00, -R2 105.37)."""
    s = str(s).strip()
    if s == "nan" or s == "":
        return s
    s = s.replace(" ", "")
    if s.startswith("-R"):
        return "-" + s[2:]
    elif s.startswith("R"):
        return s[1:]
    else:
        return s


def format_comma_decimal(s):
    """Handle SA comma-decimal format (e.g., -55 584,30 → -55584.30)."""
    s = str(s).strip()
    if s == "nan" or s == "":
        return s
    s = s.replace(" ", "")
    # Handle +/- prefix
    if s.startswith("+"):
        s = s[1:]
    # Replace comma with dot (decimal separator)
    s = s.replace(",", ".")
    return s


def clean_numeric(df, config):
    numeric_cols = [config["columns"][col] for col in config["cleaning"]["numeric"]]
    cr_mode = config["cleaning"].get("cr_suffix", False)

    r_prefix = config["cleaning"].get("r_prefix", False)
    comma_decimal = config["cleaning"].get("comma_decimal", False)

    for col in numeric_cols:
        if cr_mode:
            df[col] = df[col].apply(format_cr_dr)
        elif r_prefix:
            df[col] = df[col].apply(format_r_prefix)
        elif comma_decimal:
            df[col] = df[col].apply(format_comma_decimal)
        else:
            df[col] = df[col].apply(format_negatives)
        df[col] = df[col].str.replace(" ", "")
        if not comma_decimal:
            df[col] = df[col].str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce")


def clean_date(df, config, statement_year=None):
    date_cols = [config["columns"][col] for col in config["cleaning"]["date"]]
    if "date_format" in config["cleaning"]:
        date_format = config["cleaning"]["date_format"]
    else:
        date_format = None

    for col in date_cols:
        if statement_year and date_format and "%Y" not in date_format and "%y" not in date_format:
            # Date format has no year component - prepend the year
            df[col] = df[col].astype(str).apply(
                lambda x: f"{statement_year} {x}" if x != "nan" else x
            )
            date_format = "%Y " + date_format
        df[col] = pd.to_datetime(df[col], errors="coerce", format=date_format)


def clean_trans_detail(df, config):
    trans_detail = config["columns"]["trans_detail"]
    trans_type = config["columns"]["trans_type"]
    balance = config["columns"]["balance"]

    for i, row in df.iterrows():
        if i == 0:
            continue
        if np.isnan(row[balance]):
            df.loc[i - 1, trans_detail] = row[trans_type]


def merge_wrapped_descriptions(df, config):
    """Merge multi-line descriptions into the balance row for each transaction.

    Standard Bank pattern: each transaction consists of:
    - CONT lines above (description, no balance, no date) → belong to the NEXT balance row
    - A FULL row (has balance, may have date and/or description)
    - CONT lines below (continuation of same transaction, no balance) → also belong to this balance row

    The grouping is: collect all non-balance rows between consecutive balance rows,
    then assign them to the balance row that follows them.
    """
    desc_col = config["columns"]["trans_type"]
    balance_col = config["columns"]["balance"]

    # Find all balance row positions
    balance_positions = []
    for i in range(len(df)):
        val = df.iloc[i][balance_col]
        if pd.notna(val) and str(val).strip() not in ("", "nan"):
            balance_positions.append(i)

    if not balance_positions:
        return

    rows_to_drop = set()

    for pos, bal_idx in enumerate(balance_positions):
        # Non-balance rows between previous balance row and this one belong to THIS transaction
        if pos == 0:
            start = 0
        else:
            start = balance_positions[pos - 1] + 1

        fragments = []
        for i in range(start, bal_idx):
            desc = df.iloc[i][desc_col]
            if pd.notna(desc) and str(desc).strip():
                fragments.append(str(desc).strip())
            rows_to_drop.add(df.index[i])

        # Include the balance row's own description
        bal_desc = df.iloc[bal_idx][desc_col]
        if pd.notna(bal_desc) and str(bal_desc).strip():
            fragments.append(str(bal_desc).strip())

        if fragments:
            df.at[df.index[bal_idx], desc_col] = " ".join(fragments)

    # Drop trailing non-balance rows after the last balance row
    if balance_positions:
        for i in range(balance_positions[-1] + 1, len(df)):
            rows_to_drop.add(df.index[i])

    df.drop(list(rows_to_drop), inplace=True)


def clean_dropna(df, config):
    drop_cols = [config["columns"][col] for col in config["cleaning"]["dropna"]]
    df.dropna(subset=drop_cols, inplace=True)


def reorder_columns(df, config):
    if config.get("no_header", False):
        # Columns are positional (integer-indexed), assign names from config
        col_names = list(config["columns"].values())
        # Only rename as many columns as we have in config
        rename_map = {i: col_names[i] for i in range(min(len(df.columns), len(col_names)))}
        df = df.rename(columns=rename_map)
    else:
        column_mapper = {a: b for a, b in zip(df.columns, config["columns"].values())}
        df = df.rename(columns=column_mapper)
    ordered_columns = [config["columns"][col] for col in config["order"]]
    return df[ordered_columns]


def extract_year_from_pdf(filename):
    """Try to extract the statement year from PDF text."""
    import re
    import pdfplumber
    with pdfplumber.open(filename) as pdf:
        text = pdf.pages[0].extract_text() or ""
    # Look for patterns like "31 December 2025" or "2026/01/31" or "January 2026"
    match = re.search(r"(\d{4})/\d{2}/\d{2}", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\d{1,2}\s+\w+\s+(\d{4})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\w+\s+(\d{4})", text)
    if match:
        return int(match.group(1))
    return None


def parse_statement(filename, config):
    with pdfplumber.open(filename) as pdf:
        num_pages = len(pdf.pages)

    statement = get_raw_df(filename, num_pages, config)

    # For no_header configs, rename columns first so cleaning can find them by name
    if config.get("no_header", False) and "order" in config:
        statement = reorder_columns(statement, config)

    if "numeric" in config["cleaning"]:
        clean_numeric(statement, config)

    # Extract year for date formats without year
    statement_year = None
    if "date" in config["cleaning"]:
        date_format = config["cleaning"].get("date_format", "")
        if "%Y" not in date_format and "%y" not in date_format:
            statement_year = extract_year_from_pdf(filename)
        clean_date(statement, config, statement_year)

    # For header-based configs, reorder after cleaning
    if not config.get("no_header", False) and "order" in config:
        statement = reorder_columns(statement, config)

    if config["cleaning"].get("merge_wrapped"):
        merge_wrapped_descriptions(statement, config)

    if "trans_detail" in config["cleaning"]:
        clean_trans_detail(statement, config)

    if "dropna" in config["cleaning"]:
        clean_dropna(statement, config)

    return statement
