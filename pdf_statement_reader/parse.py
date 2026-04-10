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


def _clean_standardbank_description(desc):
    """Clean up Standard Bank descriptions.

    Standard Bank PDFs have descriptions in the format:
        "[Reference] - [Transaction Type] [Reference]"
    where the reference (account number, payee name) appears before the dash
    and again after the transaction type. This keeps the full transaction type
    with the reference appearing once (after the type), removing the leading
    duplicate and the " - " separator.

    Examples:
        "G J Rubenstein George loan acc - IB PAYMENT TO G J Rubenstein George loan acc"
        → "IB PAYMENT TO G J Rubenstein George loan acc"

        "ACC 022517480 - SERVICE FEE ACC 022517480"
        → "SERVICE FEE ACC 022517480"
    """
    if not desc:
        return desc

    if " - " in desc:
        parts = desc.split(" - ", 1)
        reference = parts[0].strip()
        remainder = parts[1].strip()

        # The remainder typically includes the reference already, so just
        # return the part after " - " which has the full description
        if remainder:
            return remainder

    return desc


def parse_with_pdfplumber(filename, config):
    """Parse PDFs using pdfplumber word positions and y-proximity grouping.

    Used for Standard Bank where descriptions wrap across rows and tabula
    can't correctly group them. Assigns description text to the nearest
    balance row within 12pt.
    """
    from collections import defaultdict

    col_bounds = config.get("pdfplumber_columns", {})
    date_max = col_bounds.get("date_max", 95)
    ref_min = col_bounds.get("ref_min", 95)
    ref_max = col_bounds.get("ref_max", 310)
    in_min = col_bounds.get("in_min", 310)
    in_max = col_bounds.get("in_max", 370)
    out_min = col_bounds.get("out_min", 370)
    out_max = col_bounds.get("out_max", 440)
    bal_min = col_bounds.get("bal_min", 500)
    table_top = col_bounds.get("table_top", 220)
    table_bottom = col_bounds.get("table_bottom", 760)
    proximity = col_bounds.get("proximity", 12)

    date_format = config["cleaning"].get("date_format", "%d %b %Y")
    comma_decimal = config["cleaning"].get("comma_decimal", False)

    def parse_amt(s):
        if not s:
            return None
        s = s.replace(" ", "")
        if comma_decimal:
            if s.startswith("+"):
                s = s[1:]
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    all_transactions = []

    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)

            rows = defaultdict(list)
            for w in words:
                row_key = round(w['top'] * 2) / 2
                rows[row_key].append(w)

            classified = []
            for y in sorted(rows.keys()):
                if y < table_top or y > table_bottom:
                    continue
                rw = sorted(rows[y], key=lambda w: w['x0'])

                dt = " ".join(w['text'] for w in rw if w['x1'] < date_max).strip()
                ref = " ".join(w['text'] for w in rw if ref_min <= w['x0'] and w['x1'] < ref_max).strip()
                inv = " ".join(w['text'] for w in rw if in_min <= w['x0'] and w['x1'] < in_max).strip()
                outv = " ".join(w['text'] for w in rw if out_min <= w['x0'] and w['x1'] < out_max).strip()
                bal = " ".join(w['text'] for w in rw if w['x0'] >= bal_min).strip()

                if dt in ("Date",) or ref in ("Reference",):
                    continue
                if any(w['text'] in ('#', '##', 'Please', 'Customer', 'Website', 'The') for w in rw):
                    continue
                if not (ref or inv or outv or bal):
                    continue

                has_bal = bool(bal) and any(c.isdigit() for c in bal)
                classified.append((y, has_bal, dt, ref, inv, outv, bal))

            for i, (y, is_bal, dt, ref, inv, outv, bal) in enumerate(classified):
                if not is_bal:
                    continue

                nearby = []
                for j, (y2, is_bal2, _, ref2, _, _, _) in enumerate(classified):
                    if is_bal2 or not ref2:
                        continue
                    if abs(y - y2) < proximity:
                        nearby.append((y2, ref2))

                nearby.sort(key=lambda x: x[0])
                desc_parts = [n[1] for n in nearby]
                if ref:
                    desc_parts.append(ref)
                description = _clean_standardbank_description(" ".join(desc_parts))

                in_val = parse_amt(inv)
                out_val = parse_amt(outv)
                bal_val = parse_amt(bal)
                amount = (in_val or 0) + (out_val or 0)

                date_val = None
                if dt:
                    try:
                        date_val = pd.to_datetime(dt, format=date_format)
                    except (ValueError, TypeError):
                        pass

                all_transactions.append({
                    "Date": date_val,
                    "Description": description,
                    "Amount": amount,
                    "Balance": bal_val,
                })

    return pd.DataFrame(all_transactions)


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


def parse_stdbank_tax_invoice(filename, config):
    """Parse Standard Bank 'BANK STATEMENT / TAX INVOICE' format using pdfplumber.

    This format has columns: Details | Service Fee | Debits | Credits | Date | Balance
    with multi-line descriptions, comma-decimal amounts, and MM DD date format.
    """
    from collections import defaultdict
    import re as _re

    col = config.get("pdfplumber_columns", {})
    desc_max = col.get("desc_max", 240)
    svc_min = col.get("svc_min", 240)
    svc_max = col.get("svc_max", 300)
    debit_min = col.get("debit_min", 300)
    debit_max = col.get("debit_max", 350)
    credit_min = col.get("credit_min", 350)
    credit_max = col.get("credit_max", 405)
    date_min = col.get("date_min", 405)
    date_max = col.get("date_max", 435)
    bal_min = col.get("bal_min", 480)
    table_top = col.get("table_top", 390)
    table_bottom = col.get("table_bottom", 650)

    skip_descriptions = {"BALANCE BROUGHT FORWARD", "OPENING BALANCE", "CLOSING BALANCE"}

    def parse_amt(s):
        """Parse amount like '505.61-' or '4,293.62' (dot-decimal, comma-thousands)."""
        if not s:
            return None
        s = s.strip()
        negative = s.endswith("-")
        if negative:
            s = s[:-1]
        s = s.replace(",", "")
        try:
            val = float(s)
            return -val if negative else val
        except (ValueError, TypeError):
            return None

    all_transactions = []

    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                continue

            # Extract statement year/month from page header
            page_text = page.extract_text() or ""
            year_match = _re.search(r"Statement from.*?(\d{4})", page_text)
            stmt_year = int(year_match.group(1)) if year_match else None
            # Also look for "DD Month YYYY" at top
            if not stmt_year:
                year_match = _re.search(r"\d{1,2}\s+\w+\s+(\d{4})", page_text[:500])
                if year_match:
                    stmt_year = int(year_match.group(1))

            # Group words by y-position
            rows = defaultdict(list)
            for w in words:
                if w["top"] < table_top or w["top"] > table_bottom:
                    continue
                row_key = round(w["top"])
                rows[row_key].append(w)

            # Skip header rows
            sorted_ys = sorted(rows.keys())
            classified = []
            for y in sorted_ys:
                rw = sorted(rows[y], key=lambda w: w["x0"])
                text_all = " ".join(w["text"] for w in rw)

                # Skip column headers
                if "Details" in text_all and ("Service" in text_all or "Credits" in text_all):
                    continue
                if text_all.strip() in ("Fee", "Debits"):
                    continue
                if "##" == text_all.strip() or text_all.startswith("## These fees"):
                    continue

                desc = " ".join(w["text"] for w in rw if w["x1"] < desc_max).strip()
                debit_val = " ".join(w["text"] for w in rw if debit_min <= w["x0"] and w["x1"] < debit_max).strip()
                credit_val = " ".join(w["text"] for w in rw if credit_min <= w["x0"] and w["x1"] < credit_max).strip()
                date_val = " ".join(w["text"] for w in rw if date_min <= w["x0"] and w["x1"] < date_max).strip()
                bal_val = " ".join(w["text"] for w in rw if w["x0"] >= bal_min).strip()

                has_amount = bool(debit_val or credit_val)
                has_balance = bool(bal_val) and any(c.isdigit() for c in bal_val)

                classified.append({
                    "y": y, "desc": desc, "debit": debit_val, "credit": credit_val,
                    "date": date_val, "balance": bal_val,
                    "has_amount": has_amount, "has_balance": has_balance,
                })

            # Group: balance rows are transactions; non-balance rows AFTER are continuations
            i = 0
            while i < len(classified):
                row = classified[i]
                if not row["has_balance"]:
                    i += 1
                    continue

                # Start with this row's description
                desc_parts = []
                if row["desc"]:
                    desc_parts.append(row["desc"])

                # Look forward for continuation lines (no balance = continuation)
                k = i + 1
                while k < len(classified) and not classified[k]["has_balance"]:
                    if classified[k]["desc"]:
                        desc_parts.append(classified[k]["desc"])
                    k += 1

                description = " ".join(desc_parts)

                # Skip excluded rows
                if any(skip in description.upper() for skip in skip_descriptions):
                    i = k
                    continue

                # Parse amount
                debit = parse_amt(row["debit"])
                credit = parse_amt(row["credit"])
                if debit is not None:
                    amount = -abs(debit) if debit > 0 else debit
                elif credit is not None:
                    amount = abs(credit)
                else:
                    amount = 0

                # Parse date (MM DD format)
                date_parsed = None
                if row["date"] and stmt_year:
                    parts = row["date"].split()
                    if len(parts) == 2:
                        try:
                            month = int(parts[0])
                            day = int(parts[1])
                            date_parsed = pd.Timestamp(year=stmt_year, month=month, day=day)
                        except (ValueError, TypeError):
                            pass

                all_transactions.append({
                    "Date": date_parsed,
                    "Description": description,
                    "Amount": amount,
                })

                i = k  # skip past continuation lines

    return pd.DataFrame(all_transactions)


def parse_statement(filename, config):
    # Use pdfplumber-based parser for banks that need y-proximity grouping
    if config.get("use_pdfplumber"):
        return parse_with_pdfplumber(filename, config)

    if config.get("use_pdfplumber_tax_invoice"):
        return parse_stdbank_tax_invoice(filename, config)

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
