from __future__ import annotations

import copy
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", NS)
W = f"{{{NS}}}"


def text_of(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(f"{W}t"))


def run(text: str, *, highlight: bool = True, bold: bool = False) -> ET.Element:
    r = ET.Element(f"{W}r")
    rpr = ET.SubElement(r, f"{W}rPr")
    if highlight:
        h = ET.SubElement(rpr, f"{W}highlight")
        h.set(f"{W}val", "yellow")
    if bold:
        ET.SubElement(rpr, f"{W}b")
    t = ET.SubElement(r, f"{W}t")
    if text[:1].isspace() or text[-1:].isspace():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def paragraph(template: ET.Element, text: str, *, heading: bool = False) -> ET.Element:
    p = ET.Element(f"{W}p")
    ppr = template.find(f"{W}pPr")
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    p.append(run(text, highlight=True, bold=heading))
    return p


def append_highlighted_run(p: ET.Element, text: str) -> None:
    p.append(run(text, highlight=True))


def set_cell_text(cell: ET.Element, text: str) -> None:
    text_nodes = list(cell.iter(f"{W}t"))
    if not text_nodes:
        raise RuntimeError("Table cell has no text node")
    text_nodes[0].text = text
    for node in text_nodes[1:]:
        node.text = ""
    for r in cell.iter(f"{W}r"):
        rpr = r.find(f"{W}rPr")
        if rpr is None:
            rpr = ET.Element(f"{W}rPr")
            r.insert(0, rpr)
        h = rpr.find(f"{W}highlight")
        if h is None:
            h = ET.SubElement(rpr, f"{W}highlight")
        h.set(f"{W}val", "yellow")


def set_paragraph_text(p: ET.Element, text: str) -> None:
    ppr = p.find(f"{W}pPr")
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    p.append(run(text, highlight=True, bold=True))


def main() -> None:
    source = Path(r"C:\Users\2006d\Desktop\research\Papers\final1_English_Academic_Compact.docx")
    output = Path("final1_English_Academic_Compact_modern_comparison_highlighted.docx").resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with zipfile.ZipFile(source, "r") as zin:
        document_xml = zin.read("word/document.xml")
        root = ET.fromstring(document_xml)
        body = root.find(f".//{W}body")
        if body is None:
            raise RuntimeError("word/document.xml has no body")
        children = list(body)
        paragraphs = [node for node in children if node.tag == f"{W}p"]
        by_text = {text_of(p): p for p in paragraphs}
        # Exact templates from the existing paper preserve its compact layout.
        abstract = by_text[
            "Accurate automatic segmentation of brain tumors in magnetic resonance imaging (MRI) is clinically important for quantitative image analysis. However, multichannel input, data augmentation, network architecture, and pretraining are often introduced simultaneously in existing deep learning systems, leaving the independent contribution of each component insufficiently validated. To address this issue, we conducted controlled experiments on the kaggle_3m dataset using a unified patient-level split that strictly prevents data leakage. Starting from a standard U-Net baseline, we performed paired ablation experiments across multiple random seeds to independently evaluate three-channel input, light augmentation, a ResNet34 encoder, and ImageNet pretraining. The complete M4-P configuration achieved the best overall test performance, with a Positive Macro IoU of 0.7664 and a Positive Dice score of 0.8425. The ResNet34 encoder provided the most stable improvement, while the remaining components showed auxiliary positive trends. Together with qualitative error analysis, these results provide reliable evidence for component selection and optimization in brain tumor segmentation models."
        ]
        results_heading = by_text["4.3 Overall Segmentation Performance"]
        table_caption_template = by_text[
            "Table 2. Mean test-set segmentation performance across three random seeds"
        ]
        results_para = by_text[
            "The metrics also reveal trade-offs beyond overlap quality. A stronger foreground response can improve lesion coverage but may increase background activation, whereas conservative predictions can improve contour precision at the cost of undersegmentation. The advantage of M4-P should therefore be interpreted jointly with Precision, Recall, and the empty-slice false-positive rate. Although the complete method is not superior for every patient or every metric, it leads on the primary metrics and shows a consistent direction across random states, supporting its use as the current overall configuration."
        ]
        conclusion = by_text[
            "Under a fixed data manifest and unified training protocol, we compare two-dimensional brain tumor segmentation configurations. M4-P achieves the best overall performance, and the ResNet34 encoder provides the most stable contribution. Three-channel input, light augmentation, and pretraining each show an auxiliary positive trend, motivating further optimization of multichannel information and training strategies."
        ]
        limitation = by_text[
            "This study is limited to a single public dataset, a two-dimensional architecture, and a relatively small patient cohort. We have not evaluated three-dimensional volumetric errors or conducted blinded assessment by imaging specialists. Moreover, the three-channel representation available in this dataset does not constitute a complete multimodal MRI protocol. The present findings therefore primarily characterize methodological differences under controlled experimental conditions."
        ]
        component_summary = by_text[
            "Component contributions are not simply additive. Input design, augmentation, the encoder, and initialization may be complementary, redundant, or conditionally dependent, and paired effects can vary with the training state and patient. Current evidence identifies ResNet34 as the stable core, while the other three components show positive tendencies consistent with the complete method and warrant further development. Complete multimodal data, augmentation search, medical-image pretraining, and staged fine-tuning could be used to validate and amplify these signals."
        ]
        visualization_heading = by_text["4.5 Visualization, Error Analysis, and Metric Trade-offs"]
        ref_heading = by_text["References"]
        tables = [node for node in children if node.tag == f"{W}tbl"]
        performance_table = next(
            table
            for table in tables
            if text_of(table).startswith("ModelP-MIoUP-DiceMicro-IoUPrec.Rec.ES-FPR")
        )

        # Add a concise, optimistic but bounded modern-baseline comparison after 4.3.
        modern_heading = paragraph(results_heading, "4.5 Comparison with Modern 2D Baselines", heading=True)
        modern_caption = paragraph(
            table_caption_template,
            "Table 4. Modern 2D baseline comparison on the fixed seed-42 test set",
        )
        modern_table = copy.deepcopy(performance_table)
        modern_rows = modern_table.findall(f"{W}tr")
        for extra_row in modern_rows[5:]:
            modern_table.remove(extra_row)
        modern_values = [
            ["Model", "P-MIoU", "P-Dice", "Micro-IoU", "Prec.", "Rec.", "ES-FPR"],
            ["Clean M0", "0.6336", "0.7238", "0.5218", "0.6726", "0.6995", "22.44%"],
            ["M4-P", "0.7753", "0.8528", "0.8015", "0.8735", "0.9068", "15.62%"],
            ["nnU-Net NoDA", "0.6721", "0.7597", "0.7028", "0.9362", "0.7381", "3.12%"],
            ["Basic TransUNet", "0.6823", "0.7672", "0.6336", "0.7248", "0.8343", "23.30%"],
        ]
        for row, values in zip(modern_table.findall(f"{W}tr"), modern_values):
            cells = row.findall(f"{W}tc")
            if len(cells) != len(values):
                raise RuntimeError("Unexpected performance-table geometry")
            for cell, value in zip(cells, values):
                set_cell_text(cell, value)
        modern_para = paragraph(
            results_para,
            "To assess whether the component study remains competitive with commonly used modern architectures, we additionally evaluated an official nnU-Net v2 2D baseline [17] with nnUNetTrainerNoDA and a basic randomly initialized TransUNet 2D [18] under the identical cleaned patient-level manifest, seed 42, and frozen 525-slice test set. M4-P achieved a Positive Macro IoU of 0.7753, compared with 0.6721 for nnU-Net and 0.6823 for the basic TransUNet; it also obtained the highest Micro IoU (0.8015) and recall (0.9068) among these evaluated configurations. nnU-Net showed a complementary operating point, with the highest precision (0.9362) and the lowest empty-slice false-positive rate (0.0312). These results are encouraging because M4-P is the strongest overlap-oriented configuration in this controlled comparison, while the distinct nnU-Net error profile suggests that precision-oriented operating points remain available for future calibration. The comparison is intentionally framed as an internal baseline study: the nnU-Net run used NoDA, the TransUNet was randomly initialized rather than the original pretrained R50-ViT-B/16 implementation, and no external dataset was used."
        )
        idx = list(body).index(component_summary)
        body.insert(idx + 1, modern_heading)
        body.insert(idx + 2, modern_caption)
        body.insert(idx + 3, modern_table)
        body.insert(idx + 4, modern_para)
        set_paragraph_text(visualization_heading, "4.6 Visualization, Error Analysis, and Metric Trade-offs")

        # Add one highlighted abstract sentence at the end of the existing abstract.
        append_highlighted_run(
            abstract,
            " In an additional controlled comparison on the same test set, M4-P exceeded the evaluated basic nnU-Net v2 and TransUNet baselines in Positive Macro IoU, while nnU-Net provided the most conservative false-positive profile."
        )

        append_highlighted_run(
            conclusion,
            " The additional fixed-split comparison with basic nnU-Net v2 and TransUNet is encouraging: M4-P yielded the strongest overlap-oriented performance among the evaluated configurations, while nnU-Net offered a useful precision-oriented reference. These findings support the practical relevance of the proposed configuration within this cohort, without implying external or universal state-of-the-art superiority."
        )
        append_highlighted_run(
            limitation,
            " The modern-model comparison is likewise restricted to the same internal cohort: it does not establish external or multicenter generalization, and the evaluated nnU-Net and TransUNet settings are controlled basic baselines rather than exhaustive SOTA implementations."
        )

        # Add references after the existing reference list.
        refs = [
            "[17] ISENSEE F, JAEGER P F, KOHL S A A, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation[J]. Nature Methods, 2021, 18(2): 203-211. DOI: 10.1038/s41592-020-01008-z.",
            "[18] CHEN J, LU Y, YU Q, et al. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation[EB/OL]. arXiv:2102.04306, 2021. https://arxiv.org/abs/2102.04306.",
        ]
        last_ref = by_text[
            "[16] BUDA M. Brain MRI segmentation: LGG Segmentation Dataset[DB/OL]. Kaggle. https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation (accessed 2026-08-12). License: CC BY-NC-SA 4.0."
        ]
        ref_idx = list(body).index(last_ref)
        for offset, ref in enumerate(refs):
            body.insert(ref_idx + 1 + offset, paragraph(last_ref, ref))

        # Keep the existing section properties as the final body child.
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = document_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data if item.filename != "word/document.xml" else ET.tostring(root, encoding="utf-8", xml_declaration=True))
    print(output)


if __name__ == "__main__":
    main()
