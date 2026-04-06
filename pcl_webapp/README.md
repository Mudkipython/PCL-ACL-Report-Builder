# PCL/ACL Report Builder

Upload a Shareholders Report PDF and the quarterly Excel workbook to generate a matched PCL/ACL report.

## Run locally

```bash
cd /Users/fredericzhang/Documents/Study/CapStone/Case_1/pcl_webapp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## What changed
- The app now runs as a single Streamlit interface instead of a FastAPI backend plus static frontend.
- Report generation still reuses the extraction logic in `app/processing.py`.
- Output is downloaded directly from the Streamlit UI.

## Output formats
- Combined Excel: PCL_PDF, PCL_Excel, ACL_Excel, and PCL_Crosscheck sheets.
- Cross-check Excel: only the PCL cross-check sheet.
- CSV bundle: ZIP containing the three extracts plus cross-check.

## Notes
- PDF PCL pages default to 84-85 and can be changed directly in the Streamlit UI.
- Excel sheets expected: `23` for Total PCL, `22` for ACL.
