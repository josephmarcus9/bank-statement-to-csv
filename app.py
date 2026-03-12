import streamlit as st
import pandas as pd
import json
import os
import tempfile

from pdf_statement_reader.parse import parse_statement  # noqa: direct import to avoid __init__.py loading pikepdf

BANK_CONFIGS = {
    "ABSA": "za.absa.transaction_history",
    "FNB": "za.fnb.business",
    "Standard Bank": "za.standardbank.current",
    "Nedbank": "za.nedbank.cheque",
}


def load_config(config_spec):
    local_dir = os.path.join(os.path.dirname(__file__), "pdf_statement_reader")
    config_dir = os.path.join(*config_spec.split(".")[:-1])
    config_file = config_spec.split(".")[-1] + ".json"
    config_path = os.path.join(local_dir, "config", config_dir, config_file)
    with open(config_path) as f:
        return json.load(f)


def build_pastel_csv(df):
    """Convert bank-specific DataFrame to Pastel format: Date, Description, Amount."""
    pastel_df = pd.DataFrame()

    # Date
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    pastel_df["Date"] = df[date_col].dt.strftime("%Y/%m/%d")

    # Description
    desc_col = [c for c in df.columns if any(k in c.lower() for k in ["description", "reference", "transaction"])][0]
    pastel_df["Description"] = df[desc_col].fillna("").replace("", "Bank Fee/Charge")
    pastel_df["Description"] = pastel_df["Description"].replace("nan", "Bank Fee/Charge")

    # Amount — merge In/Out or Debit/Credit if split into separate columns
    if "Amount" in df.columns:
        pastel_df["Amount"] = df["Amount"]
    elif "In (R)" in df.columns and "Out (R)" in df.columns:
        pastel_df["Amount"] = df["In (R)"].fillna(0) + df["Out (R)"].fillna(0)
    elif "Debits" in df.columns and "Credits" in df.columns:
        pastel_df["Amount"] = df["Credits"].fillna(0) - df["Debits"].fillna(0).abs()
    else:
        amt_col = [c for c in df.columns if "amount" in c.lower() or "debit" in c.lower()][0]
        pastel_df["Amount"] = df[amt_col]

    return pastel_df


# --- Page config ---
st.set_page_config(page_title="Bank Statement to CSV", page_icon="🏦", layout="centered")

st.title("Bank Statement to CSV")
st.caption("Convert PDF bank statements into Sage Pastel-compatible CSV files.")

# --- Inputs ---
col1, col2 = st.columns([1, 2])
with col1:
    bank = st.selectbox("Bank", list(BANK_CONFIGS.keys()))
with col2:
    uploaded_file = st.file_uploader("Upload PDF statement", type=["pdf"], label_visibility="visible")

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

    # --- Preview ---
    st.subheader("Preview")
    st.dataframe(pastel_df, use_container_width=True, hide_index=True)

    # --- Download ---
    csv_data = pastel_df.to_csv(index=False, float_format="%.2f")
    file_name = uploaded_file.name.replace(".pdf", ".csv").replace(".PDF", ".csv")

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        type="primary",
    )

    # --- Raw data toggle ---
    with st.expander("Show raw extracted data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Failed to parse statement: {e}")
    st.caption("Make sure you selected the correct bank and uploaded a valid PDF statement.")

finally:
    os.unlink(tmp_path)
