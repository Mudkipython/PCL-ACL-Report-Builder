# Financial Report PCL & ACL Automated Extraction System

## Demo
https://pclforecast.streamlit.app/

## Project Overview

This project is a complete end-to-end solution designed to automate the extraction and cross-checking of **Provision for Credit Losses (PCL)** and **Allowance for Credit Losses (ACL)** data from complex financial reports. It ensures data consistency between two different financial reporting sources:
1. A Quarterly Financial Report (PDF).
2. A Supplementary Financial Information Workbook (Excel).

While the core data logic is prototyped in Jupyter Notebooks, **the highlight of this project is a production-ready Full-Stack Web Application** that wraps the extraction logic into an accessible, deployable product for non-technical users.

## 🌟 Key Highlight: Full-Stack Web Application (`pcl_webapp/`)
To demonstrate software engineering capabilities beyond data science notebooks, this project includes a robust, independently deployable web application. It transitions the analytical scripts into a real-world software product.

### Backend Engineering (FastAPI & Python)
- **RESTful API Design**: Built a high-performance backend using **FastAPI** to handle file uploads, asynchronous processing, and report generation.
- **Advanced File Processing**: Integrated Python data engineering libraries (`pandas` and `pdfplumber`) into the backend service to parse multi-page PDFs and multi-sheet Excel workbooks on the fly.
- **Dynamic File Generation**: Generates downloadable artifacts in real-time, packaging the extracted data and cross-check results into comprehensive Excel workbooks or zipped CSV bundles utilizing `io.BytesIO`.
- **Production-Ready Deployment**: Configured for cloud deployment (e.g., Render, AWS) with a `render.yaml` specification for fast Continuous Deployment (CD).

### Frontend Development (HTML/CSS/JS)
- **Interactive User Interface**: Developed a clean, responsive front-end dashboard (`netlify/` directory) using Vanilla JavaScript, HTML5, and CSS3.
- **Client-Server Communication**: Utilized modern JavaScript Fetch API to securely transmit `multipart/form-data` to the backend and handle streaming file downloads smoothly with responsive loading states.
- **Decoupled Architecture**: Designed a separate frontend architecture optimized for static edge-hosting services like **Netlify**, strictly communicating with the backend API via CORS.

## 📊 Core Data Analytics & Engineering (`pcl_acl_extract_merge.ipynb`)
- **Automated Document Parsing**: Extracts tabular data from unstructured PDFs using regular expressions and bounding-box parsing strategies.
- **Data Transformation & Cleaning**: Standardizes varying metric names, handles missing data representations (e.g., dashes), and normalizes numeric formats across different business segment contexts.
- **Automated Reconciliation**: Implements programmable cross-checking logic with defined tolerance thresholds to automatically identify numerical discrepancies between the external shareholder reports and internal supplemental schedules.

## Directory Structure
- `pcl_webapp/`: The full-stack web application source code.
  - `app/`: FastAPI backend containing API routing (`main.py`) and data parsing logic (`processing.py`).
  - `netlify/`: The decoupled frontend source code (`index.html`, `app.js`, `style.css`).
  - `requirements.txt` & `render.yaml`: Dependency management and deployment configuration.
- `pcl_acl_extract_merge.ipynb`: The primary script for data validation and rapid prototyping.
- Sample Data Files: Example financial reports (e.g., `2025Q3.xlsx`, `Q325_Shareholders_Report-EN.pdf`) used to test the pipeline.

## Getting Started

### 1. Running the Web Application Locally
Navigate to the web app directory and start the local server:
```bash
cd pcl_webapp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000` in your browser to interact with the application.

### 2. Exploring the Analysis Notebook
To peek into the core data extraction logic:
1. Open `pcl_acl_extract_merge.ipynb` in Jupyter Environment.
2. Run the cells sequentially to see the step-by-step extraction and cross-check process on the sample files.
