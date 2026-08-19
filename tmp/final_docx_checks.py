from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "reports" / "论文中文草稿框架_第三至六部分完善终稿_会议论文结构再精简版_严格证据修订稿.docx"
REPORT = ROOT / "reports" / "论文实验数据与参考文献严格审计报告.docx"
AUDIT = ROOT / "tmp" / "strict_evidence_audit.json"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def paragraph_texts(path: Path) -> list[str]:
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def check_package(path: Path) -> tuple[list[str], dict[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, f"corrupt zip member in {path.name}"
        names = archive.namelist()
        blobs = {name: archive.read(name) for name in names}
    for name, blob in blobs.items():
        if name.endswith(".xml") or name.endswith(".rels"):
            ET.fromstring(blob)
    return names, blobs


def main() -> None:
    assert REVISION.exists() and REPORT.exists()
    revision_names, revision_blobs = check_package(REVISION)
    check_package(REPORT)

    assert "word/comments.xml" in revision_names
    document_root = ET.fromstring(revision_blobs["word/document.xml"])
    comments_root = ET.fromstring(revision_blobs["word/comments.xml"])
    starts = [e.attrib[f"{{{NS['w']}}}id"] for e in document_root.findall(".//w:commentRangeStart", NS)]
    ends = [e.attrib[f"{{{NS['w']}}}id"] for e in document_root.findall(".//w:commentRangeEnd", NS)]
    refs = [e.attrib[f"{{{NS['w']}}}id"] for e in document_root.findall(".//w:commentReference", NS)]
    comments = [e.attrib[f"{{{NS['w']}}}id"] for e in comments_root.findall(".//w:comment", NS)]
    assert len(comments) == len(starts) == len(ends) == len(refs) == 15
    assert set(comments) == set(starts) == set(ends) == set(refs)
    rels = revision_blobs["word/_rels/document.xml.rels"].decode("utf-8")
    content_types = revision_blobs["[Content_Types].xml"].decode("utf-8")
    assert "comments" in rels and "comments" in content_types

    paragraphs = paragraph_texts(REVISION)
    ref_heading = max(i for i, text in enumerate(paragraphs) if text == "参考文献")
    body_text = "\n".join(paragraphs[:ref_heading])
    reference_texts = paragraphs[ref_heading + 1 :]
    reference_numbers = [int(m.group(1)) for text in reference_texts if (m := re.match(r"^\[(\d+)\]", text))]
    assert reference_numbers == list(range(1, 12)), reference_numbers
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body_text)}
    assert cited == set(range(1, 12)), sorted(cited)
    assert not re.search(r"\[(?:1[2-9]|[2-9]\d+)\]", body_text)

    banned = [
        "缺失序列由 FLAIR 填充",
        "缺失序列由FLAIR填充",
        "所有模型均能平稳收敛",
        "M4-P 获得最佳综合表现",
        "ResNet34 编码器贡献最稳定",
    ]
    for phrase in banned:
        assert phrase not in body_text, phrase

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["metadata"]["num_patients"] == 104
    assert audit["metadata"]["num_samples"] == 3629
    assert audit["effects"]["完整方案"]["positive_seeds"] == 3
    assert round(audit["summaries"]["M4-P"]["positive_macro_iou"]["mean"], 4) == 0.7664

    report_text = "\n".join(paragraph_texts(REPORT))
    for heading in ("一、审计结论摘要", "七、16条参考文献真实性核验", "九、最终修订清单与验收结论"):
        assert heading in report_text

    result = {
        "revision_bytes": REVISION.stat().st_size,
        "report_bytes": REPORT.stat().st_size,
        "comments": len(comments),
        "references": reference_numbers,
        "body_citations": sorted(cited),
        "packages_xml_valid": True,
        "unsupported_phrase_scan": "pass",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FINAL CHECK FAILED: {exc}", file=sys.stderr)
        raise
