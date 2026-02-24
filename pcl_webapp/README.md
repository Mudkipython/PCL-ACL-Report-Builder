# PCL/ACL Report Builder

Upload a Shareholders Report PDF and the quarterly Excel workbook to generate a matched PCL/ACL report.

## Run locally

```bash
cd /Users/fredericzhang/Documents/Study/CapStone/Case_1/pcl_webapp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

## Netlify frontend + separate backend
Netlify only hosts static assets, so deploy the frontend in `netlify/` and host the FastAPI backend elsewhere.

1) Deploy the backend (Render/Fly/EC2). Ensure it is reachable over HTTPS.
2) In `netlify/index.html`, set the `api-base` meta tag to your backend URL.
3) In Netlify, set the publish directory to `netlify`.

## Render backend deployment
1) Push this repo to GitHub.
2) In Render, create a new Web Service from the repo.
3) Render will detect `render.yaml`. Deploy.
4) Copy the backend URL (e.g. `https://pcl-acl-backend.onrender.com`) into `netlify/index.html` `api-base`.

## Output formats
- Combined Excel: PCL_PDF, PCL_Excel, ACL_Excel, and PCL_Crosscheck sheets.
- Cross-check Excel: only the PCL cross-check sheet.
- CSV bundle: ZIP containing the three extracts plus cross-check.

## Notes
- PDF PCL pages are currently fixed to pages 84-85 (adjustable in `app/processing.py`).
- Excel sheets expected: `23` for Total PCL, `22` for ACL.
