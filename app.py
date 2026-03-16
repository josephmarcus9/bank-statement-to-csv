import streamlit as st
import pandas as pd
import json
import os
import re
import tempfile
from io import BytesIO

from pdf_statement_reader.parse import parse_statement  # noqa: direct import to avoid __init__.py loading pikepdf

BANK_CONFIGS = {
    "ABSA": "za.absa.transaction_history",
    "FNB": "za.fnb.business",
    "Standard Bank": "za.standardbank.current",
    "Nedbank": "za.nedbank.cheque",
}

PASTEL_MAX_CHARS = 36

# Transaction type prefixes to strip (longest first so longer matches win)
STRIP_PREFIXES = [
    # Standard Bank
    "REAL TIME TRANSFER FROM ",
    "REAL TIME TRANSFER TO ",
    "CREDIT TRANSFER ",
    "IB PAYMENT FROM ",
    "IB PAYMENT TO ",
    "LOAN REPAYMENT ",
    # ABSA
    "ACB CREDIT MERCH/SERV ",
    "ACB DEBIT:INTERNAL MERCH/SERV ",
    "ACB DEBIT:EXTERNAL ",
    "DIGITAL PAYMENT DT ",
    "IMDTE DIGITAL PMT ",
    "DIGITAL TRANSF CR ",
    "IMMEDIATE TRF CR ",
    "CARDLESS CASH DEP ",
    "NOTIFIC FEE SMS ",
    "PROOF OF PMT EMAIL ",
    "PROOF OF PAYMT SMS ",
    "ARCHIVE STMT ENQ ",
    "STAMPED STATEMENT ",
    "ACB CREDIT ",
    "ACB DEBIT ",
    # FNB
    "Magtape Debit ",
    "POS Purchase ",
    "Rtc Credit ",
    # Nedbank
    "Instant payment fee",
]


def clean_description_for_pastel(desc):
    """Clean a bank description to fit Pastel's 36-character ledger field.

    Strips transaction type prefixes, fee amounts in parentheses,
    account/card/reference numbers, and trailing dates to keep only
    the meaningful recipient or expense name.
    """
    if not desc or desc in ("Bank Fee/Charge", "nan", ""):
        return desc

    # Strip known transaction type prefixes
    for prefix in STRIP_PREFIXES:
        if desc.upper().startswith(prefix.upper()):
            remainder = desc[len(prefix):].strip()
            if remainder:  # only strip if something meaningful remains
                desc = remainder
            break

    # FNB: strip "FNB OB Pmt FNB OB 000000942 Sar " pattern (prefix + ref + short code)
    desc = re.sub(r"^FNB OB Pmt FNB OB \d+ \w{3}\s+", "", desc, flags=re.IGNORECASE)

    # Remove fee amounts in parentheses like "( 10,00 )" or "(19,75 )"
    desc = re.sub(r"\(\s*[\d,\.]+\s*\)\s*", "", desc)

    # Remove "ABSA BANK" after stripping ABSA payment prefixes
    desc = re.sub(r"^ABSA BANK\s+", "", desc)
    # Also handle "ABSA BANK" preceded by space (from DIGITAL PAYMENT DT cleanup)
    desc = re.sub(r"\s*ABSA BANK\s*", " ", desc)

    # Remove EFFEC date references like "(EFFEC 30012026)"
    desc = re.sub(r"\(EFFEC\s*\d*\)", "", desc)

    # Remove card numbers glued to text (e.g. "GERMI5412820023910484" → "GERMI")
    desc = re.sub(r"(\D)\d{10,}", r"\1", desc)

    # Remove standalone long digit sequences (card/account numbers, 8+ digits)
    desc = re.sub(r"\b\d{8,}\b", "", desc)

    # Remove hex reference codes like "161A2AEED7"
    desc = re.sub(r"\b[0-9A-F]{10,}\b", "", desc)

    # Remove Nedbank ADT-style reference codes glued to text (e.g. "2114879792ADTA059137")
    desc = re.sub(r"\d{5,}[A-Z]+\d+", "", desc)

    # Remove POS card references like "491050*0796"
    desc = re.sub(r"\d+\*\d+", "", desc)

    # Remove card suffixes DD/CC at end
    desc = re.sub(r"\s+(DD|CC)\s*$", "", desc)

    # Remove trailing short dates like "02 Jan"
    desc = re.sub(
        r"\s+\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*$",
        "",
        desc,
        flags=re.IGNORECASE,
    )

    # Remove "ACC" followed by account number
    desc = re.sub(r"\bACC\s+\d+\b", "", desc)

    # Remove standalone short reference numbers (3-7 digits) at end of line
    desc = re.sub(r"\s+\d{3,7}\s*$", "", desc)

    # Remove "NOTIFICATION" if duplicated (e.g. "... 0013 NOTIFICATION")
    desc = re.sub(r"\s+NOTIFICATION\s*$", "", desc, flags=re.IGNORECASE)

    # Remove percentage rates like "@17750%" or "@17,750%"
    desc = re.sub(r"@[\d,]+%", "", desc)

    # Clean up multiple spaces and trim
    desc = re.sub(r"\s+", " ", desc).strip()

    # Remove trailing/leading dashes and hyphens
    desc = desc.strip(" -")

    return desc


def build_pastel_csv(df):
    """Convert bank-specific DataFrame to Pastel format: Date, Description, Amount."""
    pastel_df = pd.DataFrame()

    # Date
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    pastel_df["Date"] = df[date_col].dt.strftime("%Y/%m/%d")

    # Description (full original)
    desc_col = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["description", "reference", "transaction"])
    ][0]
    pastel_df["Description"] = df[desc_col].fillna("").replace("", "Bank Fee/Charge")
    pastel_df["Description"] = pastel_df["Description"].replace(
        "nan", "Bank Fee/Charge"
    )

    # Ledger Description (cleaned, max 36 chars for Pastel)
    pastel_df["Ledger Description"] = pastel_df["Description"].apply(
        clean_description_for_pastel
    )

    # Amount — merge In/Out or Debit/Credit if split into separate columns
    if "Amount" in df.columns:
        pastel_df["Amount"] = df["Amount"]
    elif "In (R)" in df.columns and "Out (R)" in df.columns:
        pastel_df["Amount"] = df["In (R)"].fillna(0) + df["Out (R)"].fillna(0)
    elif "Debits" in df.columns and "Credits" in df.columns:
        pastel_df["Amount"] = df["Credits"].fillna(0) - df["Debits"].fillna(0).abs()
    else:
        amt_col = [
            c
            for c in df.columns
            if "amount" in c.lower() or "debit" in c.lower()
        ][0]
        pastel_df["Amount"] = df[amt_col]

    return pastel_df


def highlight_over_limit(row):
    """Highlight rows yellow where Ledger Description exceeds 36 characters."""
    if len(str(row["Ledger Description"])) > PASTEL_MAX_CHARS:
        return ["background-color: #FFFF00"] * len(row)
    return [""] * len(row)


def to_excel_with_highlights(pastel_df):
    """Create an Excel file with yellow highlighting for descriptions over 36 chars."""
    from openpyxl.styles import PatternFill

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pastel_df.to_excel(writer, index=False, sheet_name="Transactions")
        ws = writer.sheets["Transactions"]
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

        ledger_col_idx = list(pastel_df.columns).index("Ledger Description") + 1

        for row_idx in range(2, len(pastel_df) + 2):
            cell = ws.cell(row=row_idx, column=ledger_col_idx)
            if cell.value and len(str(cell.value)) > PASTEL_MAX_CHARS:
                for col_idx in range(1, len(pastel_df.columns) + 1):
                    ws.cell(row=row_idx, column=col_idx).fill = yellow_fill

        for col in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)

    return output.getvalue()


def load_config(config_spec):
    local_dir = os.path.join(os.path.dirname(__file__), "pdf_statement_reader")
    config_dir = os.path.join(*config_spec.split(".")[:-1])
    config_file = config_spec.split(".")[-1] + ".json"
    config_path = os.path.join(local_dir, "config", config_dir, config_file)
    with open(config_path) as f:
        return json.load(f)


# --- Page config ---
st.set_page_config(
    page_title="Bank Statement to CSV", page_icon="🏦", layout="centered"
)

st.title("Bank Statement to CSV")
st.caption(
    "Convert PDF bank statements into Sage Pastel-compatible CSV files. "
    "Descriptions are automatically cleaned to fit Pastel's 36-character ledger limit."
)

# --- Inputs ---
col1, col2 = st.columns([1, 2])
with col1:
    bank = st.selectbox("Bank", list(BANK_CONFIGS.keys()))
with col2:
    uploaded_file = st.file_uploader(
        "Upload PDF statement", type=["pdf"], label_visibility="visible"
    )

if not uploaded_file:
    st.info("Upload a PDF bank statement to get started.")
    st.stop()

# --- Process ---
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

try:
    with st.spinner("Extracting transactions..."):
        config = load_config(BANK_CONFIGS[bank])
        df = parse_statement(tmp_path, config)
        pastel_df = build_pastel_csv(df)

    st.success(f"Extracted **{len(pastel_df)}** transactions from **{bank}** statement.")

    # Count descriptions over the limit
    over_limit = pastel_df["Ledger Description"].apply(lambda x: len(str(x)) > PASTEL_MAX_CHARS)
    n_over = over_limit.sum()
    if n_over > 0:
        st.warning(
            f"**{n_over}** description(s) still exceed {PASTEL_MAX_CHARS} characters "
            f"(highlighted in yellow). Review these before importing into Pastel."
        )

    # --- Preview with highlighting ---
    st.subheader("Preview")
    styled = pastel_df.style.apply(highlight_over_limit, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Downloads ---
    st.subheader("Download")
    col_csv, col_xlsx = st.columns(2)
    file_stem = uploaded_file.name.replace(".pdf", "").replace(".PDF", "")

    with col_csv:
        csv_data = pastel_df.to_csv(index=False, float_format="%.2f")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{file_stem}.csv",
            mime="text/csv",
            type="primary",
        )

    with col_xlsx:
        xlsx_data = to_excel_with_highlights(pastel_df)
        st.download_button(
            label="Download Excel (highlighted)",
            data=xlsx_data,
            file_name=f"{file_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # --- Raw data toggle ---
    with st.expander("Show raw extracted data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Failed to parse statement: {e}")
    st.caption(
        "Make sure you selected the correct bank and uploaded a valid PDF statement."
    )

finally:
    os.unlink(tmp_path)
