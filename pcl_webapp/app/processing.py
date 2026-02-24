import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pdfplumber

SEGMENTS = [
    'Canadian Banking',
    'International Banking',
    'Wealth Management',
    'Banking and Markets',
    'Other',
    'Total',
]


def _parse_pcl_line(line: str) -> List[str | None]:
    line_norm = line.replace('–', '-').replace('—', '-')
    tokens = re.findall(r'\(\d+[\d,]*\)|-?\d[\d,]*|-', line_norm)
    values: List[str | None] = []
    for token in tokens:
        token = token.strip()
        if token == '-':
            values.append(None)
            continue
        if token.startswith('(') and token.endswith(')'):
            token = '-' + token[1:-1]
        values.append(token.replace(',', ''))
    if len(values) > 6:
        values = values[-6:]
    if len(values) < 6:
        values += [None] * (6 - len(values))
    return values


def extract_pcl_pdf(pdf_path: Path, pages: List[int] | None = None) -> pd.DataFrame:
    pages = pages or [84, 85]
    blocks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num in pages:
            if page_num - 1 < 0 or page_num - 1 >= len(pdf.pages):
                continue
            page = pdf.pages[page_num - 1]
            text = page.extract_text() or ''
            lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
            current = None
            for line in lines:
                match = re.match(r'^For the (three|nine) months ended (.+)$', line)
                if match:
                    if current:
                        blocks.append(current)
                    current = {
                        'period_type': match.group(1),
                        'period_end_raw': match.group(2),
                        'lines': [],
                    }
                    continue
                if current:
                    current['lines'].append(line)
            if current:
                blocks.append(current)

    pcl_rows = []
    for block in blocks:
        period_end_raw = block['period_end_raw']
        try:
            period_end = datetime.strptime(
                period_end_raw.replace(',', ''), '%B %d %Y'
            ).date().isoformat()
        except ValueError:
            period_end = period_end_raw

        for line in block['lines']:
            if line.startswith('Provision for credit losses'):
                values = _parse_pcl_line(line)
                for segment, value in zip(SEGMENTS, values):
                    pcl_rows.append(
                        {
                            'metric': 'Provision for credit losses',
                            'period_type': block['period_type'],
                            'period_end': period_end,
                            'segment': segment,
                            'value': value,
                            'source': pdf_path.name,
                        }
                    )
    return pd.DataFrame(pcl_rows)


def extract_pcl_excel(excel_path: Path) -> pd.DataFrame:
    pcl_rows = []
    df = pd.read_excel(excel_path, sheet_name='23', header=None)

    quarter_to_date = {
        'Q3/25': '2025-07-31',
        'Q2/25': '2025-04-30',
        'Q1/25': '2025-01-31',
        'Q4/24': '2024-10-31',
        'Q3/24': '2024-07-31',
    }

    header_row_idx = None
    for i in range(0, 8):
        row_vals = df.iloc[i].tolist()
        if any(isinstance(v, str) and v.strip() in quarter_to_date for v in row_vals):
            header_row_idx = i
            break

    start_idx = None
    for i in range(0, len(df)):
        cell = df.iloc[i, 0]
        if isinstance(cell, str) and 'Total PCL ($ millions)' in cell:
            start_idx = i + 1
            break

    target_labels = {
        'Canadian Banking': 'Canadian Banking',
        'International Banking': 'International Banking',
        'Global Wealth Management': 'Wealth Management',
        'Global Banking and Markets': 'Banking and Markets',
        'Other': 'Other',
        'Total PCL': 'Total',
    }

    if header_row_idx is None or start_idx is None:
        raise ValueError('Could not locate PCL table in sheet 23')

    quarter_cols: List[Tuple[int, str]] = []
    header_row = df.iloc[header_row_idx]
    for col_idx, val in header_row.items():
        if isinstance(val, str) and val.strip() in quarter_to_date:
            quarter_cols.append((col_idx, val.strip()))

    for i in range(start_idx, len(df)):
        label = df.iloc[i, 0]
        if not isinstance(label, str) or not label.strip():
            if i > start_idx:
                break
            continue
        label = label.strip()
        if label.startswith('PCL on loans') or 'Provision for Credit Losses as a %' in label:
            break
        if label not in target_labels:
            continue

        for col_start, quarter_label in quarter_cols:
            total_pcl = df.iloc[i, col_start + 2]
            if pd.isna(total_pcl):
                continue
            if isinstance(total_pcl, str):
                total_pcl = total_pcl.replace(',', '').strip()
                if total_pcl in ('-', '–'):
                    total_pcl = None
            pcl_rows.append(
                {
                    'metric': 'Provision for credit losses',
                    'period_end': quarter_to_date.get(quarter_label, quarter_label),
                    'segment': target_labels[label],
                    'value': total_pcl,
                    'source': excel_path.name,
                }
            )
    return pd.DataFrame(pcl_rows)


def extract_acl_excel(excel_path: Path) -> pd.DataFrame:
    df_acl = pd.read_excel(excel_path, sheet_name='22', header=None)

    title_idx = None
    matches = df_acl[0].astype(str).str.contains(
        'Impaired Loans by Type of Borrower', na=False
    )
    if matches.any():
        title_idx = matches[matches].index[0]

    acl_rows = []
    if title_idx is not None:
        date_row_idx = None
        date_pattern = re.compile(r'^[A-Za-z]+ \d{1,2}, \d{4}$')
        for i in range(title_idx + 1, title_idx + 5):
            row = df_acl.iloc[i].tolist()
            if any(
                isinstance(x, str) and date_pattern.match(x.strip()) for x in row
            ):
                date_row_idx = i
                break

        if date_row_idx is None:
            raise ValueError('Could not locate ACL date row in sheet 22')

        subheader_row_idx = None
        for i in range(date_row_idx + 1, date_row_idx + 4):
            cell = df_acl.iloc[i, 0]
            if isinstance(cell, str) and '($ millions' in cell:
                subheader_row_idx = i
                break
        if subheader_row_idx is None:
            subheader_row_idx = date_row_idx + 1

        end_idx = None
        for i in range(subheader_row_idx + 1, len(df_acl)):
            cell = df_acl.iloc[i, 0]
            if (
                isinstance(cell, str)
                and 'Impaired Loans, Net of Related Allowances' in cell
            ):
                end_idx = i
                break
        if end_idx is None:
            end_idx = len(df_acl)

        data = df_acl.iloc[subheader_row_idx + 1 : end_idx].copy()

        date_cols: List[Tuple[int, str]] = []
        date_row = df_acl.iloc[date_row_idx]
        for col_idx, val in date_row.items():
            if isinstance(val, str) and date_pattern.match(val.strip()):
                date_cols.append((col_idx, val.strip()))

        def clean_num(val):
            if pd.isna(val):
                return None
            if isinstance(val, str):
                val = val.strip()
                if val in ('-', '–'):
                    return None
                val = val.replace(',', '')
            return val

        for _, row in data.iterrows():
            label = row.iloc[0]
            if not isinstance(label, str) or not label.strip():
                continue
            row_vals = row.iloc[1:]
            if row_vals.isna().all():
                continue
            label = label.strip()
            for col_start, date_str in date_cols:
                gross = row.iloc[col_start]
                stage3 = row.iloc[col_start + 1] if col_start + 1 < len(row) else None
                net = row.iloc[col_start + 2] if col_start + 2 < len(row) else None
                if pd.isna(gross) and pd.isna(stage3) and pd.isna(net):
                    continue
                acl_rows.append(
                    {
                        'metric': 'Allowance for Credit Losses',
                        'borrower_type': label,
                        'period_end': datetime.strptime(
                            date_str, '%B %d, %Y'
                        ).date().isoformat(),
                        'gross': clean_num(gross),
                        'stage_3': clean_num(stage3),
                        'net': clean_num(net),
                        'source': excel_path.name,
                    }
                )
    return pd.DataFrame(acl_rows)


def build_crosscheck(
    pcl_pdf_df: pd.DataFrame, pcl_excel_df: pd.DataFrame
) -> pd.DataFrame:
    pcl_pdf_check = pcl_pdf_df[
        pcl_pdf_df['period_type'] == 'three'
    ][['period_end', 'segment', 'metric', 'value']].copy()
    pcl_excel_check = pcl_excel_df[[
        'period_end',
        'segment',
        'metric',
        'value',
    ]].copy()

    pcl_pdf_check['value'] = pd.to_numeric(pcl_pdf_check['value'], errors='coerce')
    pcl_excel_check['value'] = pd.to_numeric(
        pcl_excel_check['value'], errors='coerce'
    )

    merged = pcl_pdf_check.merge(
        pcl_excel_check,
        on=['period_end', 'segment', 'metric'],
        how='inner',
        suffixes=('_pdf', '_excel'),
    )

    merged['note'] = ''
    missing_pdf = merged['value_pdf'].isna() & merged['value_excel'].notna()
    missing_excel = merged['value_excel'].isna() & merged['value_pdf'].notna()

    near_zero_pdf = missing_excel & (merged['value_pdf'].abs() <= 1)
    near_zero_excel = missing_pdf & (merged['value_excel'].abs() <= 1)

    merged.loc[near_zero_pdf, 'note'] = 'PDF missing; treated as 0 within tolerance'
    merged.loc[near_zero_excel, 'note'] = 'Excel missing; treated as 0 within tolerance'

    merged['diff'] = (merged['value_pdf'] - merged['value_excel']).abs()
    denom = merged[['value_pdf', 'value_excel']].abs().max(axis=1).replace(0, 1)
    merged['rel_diff'] = merged['diff'] / denom

    merged['pass'] = (merged['diff'] <= 1) | (merged['rel_diff'] <= 0.01)
    merged.loc[near_zero_pdf | near_zero_excel, 'pass'] = True

    return merged
