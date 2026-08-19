from __future__ import annotations

import argparse
import time
from pathlib import Path

import fitz
import win32com.client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("docx")
    parser.add_argument("outdir")
    args = parser.parse_args()
    docx = Path(args.docx).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{docx.stem}.pdf"
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(docx), ReadOnly=True, AddToRecentFiles=False)
        document.ExportAsFixedFormat(
            OutputFileName=str(pdf), ExportFormat=17, OpenAfterExport=False,
            OptimizeFor=0, Range=0, Item=0, IncludeDocProps=True,
            KeepIRM=True, CreateBookmarks=0, DocStructureTags=True,
            BitmapMissingFonts=True, UseISO19005_1=False,
        )
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()
    deadline = time.time() + 20
    while not pdf.exists() and time.time() < deadline:
        time.sleep(0.2)
    if not pdf.exists():
        raise RuntimeError("Word did not create PDF")
    rendered = fitz.open(pdf)
    for index, page in enumerate(rendered, start=1):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        pixmap.save(outdir / f"page-{index:02d}.png")
    print(f"{docx.name}: {len(rendered)} pages")


if __name__ == "__main__":
    main()
