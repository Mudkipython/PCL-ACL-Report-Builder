from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable
import zipfile

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

from .processing import (
    build_crosscheck,
    extract_acl_excel,
    extract_pcl_excel,
    extract_pcl_pdf,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / 'templates'))

app = FastAPI(title='PCL/ACL Report Builder')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(
        'index.html',
        {
            'request': request,
            'error': None,
        },
    )


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    suffix = Path(upload.filename or '').suffix
    tmp_path = dest_dir / f'upload{suffix}'
    with tmp_path.open('wb') as fh:
        fh.write(upload.file.read())
    return tmp_path


def _cleanup_dir(path: Path) -> Callable[[], None]:
    def _remove():
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    return _remove


def _write_combined_xlsx(
    output_path: Path,
    pcl_pdf_df: pd.DataFrame,
    pcl_excel_df: pd.DataFrame,
    acl_excel_df: pd.DataFrame,
    crosscheck_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pcl_pdf_df.to_excel(writer, sheet_name='PCL_PDF', index=False)
        pcl_excel_df.to_excel(writer, sheet_name='PCL_Excel', index=False)
        acl_excel_df.to_excel(writer, sheet_name='ACL_Excel', index=False)
        crosscheck_df.to_excel(writer, sheet_name='PCL_Crosscheck', index=False)


@app.post('/process')
async def process(
    request: Request,
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    excel_file: UploadFile = File(...),
    output_format: str = Form('combined_xlsx'),
):
    if not pdf_file.filename or not excel_file.filename:
        return TEMPLATES.TemplateResponse(
            'index.html',
            {
                'request': request,
                'error': 'Please provide both a PDF and an Excel file.',
            },
            status_code=400,
        )

    if Path(pdf_file.filename).suffix.lower() != '.pdf':
        return TEMPLATES.TemplateResponse(
            'index.html',
            {
                'request': request,
                'error': 'PDF file must have a .pdf extension.',
            },
            status_code=400,
        )

    if Path(excel_file.filename).suffix.lower() not in {'.xlsx', '.xls'}:
        return TEMPLATES.TemplateResponse(
            'index.html',
            {
                'request': request,
                'error': 'Excel file must have a .xlsx or .xls extension.',
            },
            status_code=400,
        )

    tmp_path = Path(tempfile.mkdtemp(prefix='pcl_webapp_'))
    pdf_path = _save_upload(pdf_file, tmp_path)
    excel_path = _save_upload(excel_file, tmp_path)

    pcl_pdf_df = extract_pcl_pdf(pdf_path)
    pcl_excel_df = extract_pcl_excel(excel_path)
    acl_excel_df = extract_acl_excel(excel_path)
    crosscheck_df = build_crosscheck(pcl_pdf_df, pcl_excel_df)

    if output_format == 'crosscheck_xlsx':
        output_path = tmp_path / 'pcl_crosscheck_report.xlsx'
        crosscheck_df.to_excel(output_path, index=False)
        download_name = 'pcl_crosscheck_report.xlsx'
    elif output_format == 'csv_zip':
        output_path = tmp_path / 'pcl_outputs.zip'
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('pcl_pdf.csv', pcl_pdf_df.to_csv(index=False))
            zf.writestr('pcl_excel.csv', pcl_excel_df.to_csv(index=False))
            zf.writestr('acl_excel.csv', acl_excel_df.to_csv(index=False))
            zf.writestr('pcl_crosscheck.csv', crosscheck_df.to_csv(index=False))
        download_name = 'pcl_outputs.zip'
    else:
        output_path = tmp_path / 'pcl_combined_report.xlsx'
        _write_combined_xlsx(
            output_path,
            pcl_pdf_df,
            pcl_excel_df,
            acl_excel_df,
            crosscheck_df,
        )
        download_name = 'pcl_combined_report.xlsx'

    background_tasks.add_task(_cleanup_dir, tmp_path)
    return FileResponse(
        path=str(output_path),
        filename=download_name,
        media_type='application/octet-stream',
        background=background_tasks,
    )
