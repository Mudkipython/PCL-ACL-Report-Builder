from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import pandas as pd
import streamlit as st

from app.processing import (
    build_crosscheck,
    extract_acl_excel,
    extract_pcl_excel,
    extract_pcl_pdf,
)

PAGE_TITLE = "PCL/ACL Report Builder"
OUTPUT_OPTIONS = {
    "Combined Excel (all sheets)": "combined_xlsx",
    "Cross-check Excel only": "crosscheck_xlsx",
    "CSV bundle (zip)": "csv_zip",
}


def _parse_pdf_pages(raw_pages: str) -> list[int]:
    tokens = [token.strip() for token in raw_pages.replace(";", ",").split(",")]
    pages = [int(token) for token in tokens if token]
    if not pages or any(page <= 0 for page in pages):
        raise ValueError("PDF pages must be positive integers, for example: 84, 85")
    return pages


def _save_uploaded_file(uploaded_file: Any, destination: Path) -> Path:
    output_path = destination / Path(uploaded_file.name).name
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def _make_excel_bytes(sheet_map: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheet_map.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()


def _make_zip_bytes(file_map: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, content in file_map.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


def _process_files(
    pdf_upload: Any,
    excel_upload: Any,
    output_format: str,
    pdf_pages: list[int],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pcl_streamlit_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        pdf_path = _save_uploaded_file(pdf_upload, temp_dir)
        excel_path = _save_uploaded_file(excel_upload, temp_dir)

        pcl_pdf_df = extract_pcl_pdf(pdf_path, pages=pdf_pages)
        pcl_excel_df = extract_pcl_excel(excel_path)
        acl_excel_df = extract_acl_excel(excel_path)
        crosscheck_df = build_crosscheck(pcl_pdf_df, pcl_excel_df)

    if output_format == "crosscheck_xlsx":
        payload = _make_excel_bytes({"PCL_Crosscheck": crosscheck_df})
        file_name = "pcl_crosscheck_report.xlsx"
        mime = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif output_format == "csv_zip":
        payload = _make_zip_bytes(
            {
                "pcl_pdf.csv": pcl_pdf_df.to_csv(index=False),
                "pcl_excel.csv": pcl_excel_df.to_csv(index=False),
                "acl_excel.csv": acl_excel_df.to_csv(index=False),
                "pcl_crosscheck.csv": crosscheck_df.to_csv(index=False),
            }
        )
        file_name = "pcl_outputs.zip"
        mime = "application/zip"
    else:
        payload = _make_excel_bytes(
            {
                "PCL_PDF": pcl_pdf_df,
                "PCL_Excel": pcl_excel_df,
                "ACL_Excel": acl_excel_df,
                "PCL_Crosscheck": crosscheck_df,
            }
        )
        file_name = "pcl_combined_report.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return {
        "pcl_pdf_df": pcl_pdf_df,
        "pcl_excel_df": pcl_excel_df,
        "acl_excel_df": acl_excel_df,
        "crosscheck_df": crosscheck_df,
        "payload": payload,
        "file_name": file_name,
        "mime": mime,
    }


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 241, 225, 0.95), transparent 35%),
                radial-gradient(circle at 18% 20%, rgba(255, 232, 217, 0.9), transparent 26%),
                linear-gradient(120deg, #f6f0e8 0%, #f1efe9 58%, #e8efe9 100%);
        }
        .hero {
            padding: 0.4rem 0 1.6rem;
        }
        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 122, 77, 0.12);
            color: #d95f36;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: clamp(2rem, 3vw, 3.4rem);
            font-weight: 700;
            line-height: 1.05;
            margin: 0.8rem 0 0.7rem;
            color: #141326;
        }
        .hero-copy {
            max-width: 48rem;
            font-size: 1rem;
            line-height: 1.7;
            color: #555a74;
            margin: 0;
        }
        .panel {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(20, 19, 38, 0.08);
            border-radius: 1.2rem;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 45px rgba(20, 19, 38, 0.08);
        }
        .panel h3 {
            margin: 0 0 0.45rem;
            color: #141326;
            font-size: 1rem;
        }
        .panel p, .panel li {
            color: #555a74;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(20, 19, 38, 0.08);
            border-radius: 1rem;
            padding: 0.9rem 1rem;
        }
        .metric-label {
            color: #555a74;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #141326;
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .stButton > button, .stDownloadButton > button {
            border: none;
            border-radius: 0.85rem;
            background: linear-gradient(135deg, #ff7a4d, #ef6b3e);
            color: white;
            font-weight: 700;
            padding: 0.7rem 1rem;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            color: white;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_metric(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=":bar_chart:", layout="wide")
    _render_styles()

    st.markdown(
        """
        <section class="hero">
            <div class="hero-badge">PCL · ACL Automation</div>
            <h1 class="hero-title">Upload Excel + PDF, then download a reconciled report.</h1>
            <p class="hero-copy">
                This Streamlit app extracts segment-level PCL from the shareholders report PDF,
                aligns the workbook PCL and ACL tables, and produces a downloadable report in the
                format you need.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.6, 1], gap="large")
    with left_col:
        with st.form("report_form", clear_on_submit=False):
            pdf_upload = st.file_uploader(
                "Shareholders Report PDF",
                type=["pdf"],
                help="Expected source: the report containing the segment PCL rows.",
            )
            excel_upload = st.file_uploader(
                "Quarter Excel Workbook",
                type=["xlsx", "xls"],
                help="Expected sheets: 23 for Total PCL, 22 for ACL.",
            )
            output_label = st.selectbox(
                "Download format",
                options=list(OUTPUT_OPTIONS.keys()),
                index=0,
            )
            pdf_pages_raw = st.text_input(
                "PDF pages for segment PCL extraction",
                value="84, 85",
                help="Comma-separated page numbers. Defaults match the current report layout.",
            )
            submitted = st.form_submit_button("Generate report")

        if submitted:
            if pdf_upload is None or excel_upload is None:
                st.error("Please provide both a PDF and an Excel workbook.")
            else:
                try:
                    pdf_pages = _parse_pdf_pages(pdf_pages_raw)
                    with st.spinner("Processing report..."):
                        st.session_state["report_result"] = _process_files(
                            pdf_upload=pdf_upload,
                            excel_upload=excel_upload,
                            output_format=OUTPUT_OPTIONS[output_label],
                            pdf_pages=pdf_pages,
                        )
                    st.session_state["report_pages"] = pdf_pages
                    st.success("Report generated. Review the preview and download below.")
                except Exception as exc:
                    st.error(str(exc))

    with right_col:
        st.markdown(
            """
            <div class="panel">
                <h3>Expected Inputs</h3>
                <p>The app keeps the original extraction assumptions unless you override the PDF pages.</p>
                <ul>
                    <li>Excel sheet <strong>23</strong>: Total PCL</li>
                    <li>Excel sheet <strong>22</strong>: ACL table</li>
                    <li>Default PDF pages: <strong>84, 85</strong></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class="panel">
                <h3>Output Options</h3>
                <p>Choose the format that matches the next step in your workflow.</p>
                <ul>
                    <li><strong>Combined Excel</strong>: all extracts plus cross-check</li>
                    <li><strong>Cross-check Excel</strong>: reconciliation sheet only</li>
                    <li><strong>CSV bundle</strong>: zipped raw outputs</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    report_result = st.session_state.get("report_result")
    if not report_result:
        return

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _render_metric("PCL PDF Rows", len(report_result["pcl_pdf_df"]))
    with m2:
        _render_metric("PCL Excel Rows", len(report_result["pcl_excel_df"]))
    with m3:
        _render_metric("ACL Excel Rows", len(report_result["acl_excel_df"]))
    with m4:
        pass_count = int(report_result["crosscheck_df"]["pass"].fillna(False).sum())
        _render_metric("Cross-check Pass", pass_count)

    st.write("")
    st.download_button(
        label=f"Download {report_result['file_name']}",
        data=report_result["payload"],
        file_name=report_result["file_name"],
        mime=report_result["mime"],
        use_container_width=False,
    )

    pages_used = ", ".join(str(page) for page in st.session_state.get("report_pages", []))
    st.caption(f"PDF extraction pages used: {pages_used}")

    tab_pdf, tab_pcl, tab_acl, tab_check = st.tabs(
        ["PCL_PDF", "PCL_Excel", "ACL_Excel", "PCL_Crosscheck"]
    )
    with tab_pdf:
        st.dataframe(report_result["pcl_pdf_df"], use_container_width=True)
    with tab_pcl:
        st.dataframe(report_result["pcl_excel_df"], use_container_width=True)
    with tab_acl:
        st.dataframe(report_result["acl_excel_df"], use_container_width=True)
    with tab_check:
        st.dataframe(report_result["crosscheck_df"], use_container_width=True)


if __name__ == "__main__":
    main()
