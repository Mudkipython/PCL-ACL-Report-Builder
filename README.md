# PCL and ACL Extraction & Cross-Check Project

## Project Overview

This project is designed to automate the extraction and cross-checking of **Provision for Credit Losses (PCL)** and **Allowance for Credit Losses (ACL)** data from financial reports. It ensures data consistency between two different financial reporting sources:
1. A Quarterly Financial Report (PDF).
2. A Supplementary Financial Information Workbook (Excel).

The project contains two main components:
- **Jupyter Notebook (`pcl_acl_extract_merge.ipynb`)**: The core script that parses the PDF and Excel files, extracts the relevant PCL/ACL numbers using regular expressions and text parsing (`pdfplumber` and `pandas`), and performs a cross-check to verify if the PCL values match exactly (or within a reasonable tolerance).
- **Web Application (`pcl_webapp/`)**: A user-friendly tool built with FastAPI and hosted static frontend that allows users to upload their own PDF and Excel files. After uploading, the app processes the files and generates a comprehensive report in Excel or CSV format containing the extracted data and the cross-check results.

## Features
- Automated PDF parsing for financial tables.
- Excel data extraction for specific business lines and borrower types.
- Discrepancy detection between external shareholder reports and supplemental schedules.
- User-friendly web interface for non-technical users to access the tool.

## Directory Structure
- `pcl_acl_extract_merge.ipynb`: The primary data analysis and extraction notebook.
- `pcl_webapp/`: Contains the FastAPI backend and Netlify frontend code.
  - `pcl_webapp/README.md`: Specific deployment and execution instructions for the web app.
- Sample Data Files (e.g., `2025Q3.xlsx`, `Q325_Shareholders_Report-EN.pdf`): Example financial reports used for testing.

## Getting Started

### Prerequisites
Make sure you have Python installed. You can install the required packages using:
```bash
pip install pandas openpyxl pdfplumber jupyter
```

### Running the Notebook
Open the `pcl_acl_extract_merge.ipynb` in Jupyter Notebook or Jupyter Lab and run the cells sequentially to see the step-by-step extraction and cross-check process.

### Running the Web App
Please refer to `pcl_webapp/README.md` for specific instructions on how to start the FastAPI server and access the frontend.
