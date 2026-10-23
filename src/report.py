"""Build a portable results packet directly from verified experiment artifacts."""

from html import escape
import json
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from checkpoint import write_json
from config import Config
from prepare_data import sha256_file


def validate_samples(samples: dict, results: dict) -> None:
    config = Config.from_dict(results["config"])
    if samples["tokenizer_sha256"] != results["tokenizer_sha256"]:
        raise ValueError("Samples use a different tokenizer")
    settings = {name: getattr(config, name) for name in ("seed", "temperature", "top_k", "max_new_tokens", "early_stop")}
    if samples["sampling"] != settings or set(samples["stages"]) != {"initial", "halfway", "final"}:
        raise ValueError("Sample stages or settings differ from the frozen experiment")
    for stage, entry in samples["stages"].items():
        if entry["tokens_seen"] != results["checkpoints"][stage]:
            raise ValueError(f"Samples have the wrong checkpoint token count: {stage}")
        if [sample["prompt"] for sample in entry["samples"]] != list(config.prompts):
            raise ValueError("Samples must contain the same three preselected prompts in order")
        for sample in entry["samples"]:
            if sample["new_token_count"] != config.max_new_tokens or len(sample["new_token_ids"]) != config.max_new_tokens:
                raise ValueError("Sample length differs from the frozen protocol")


def requirement_status(results: dict, notes_present: bool) -> dict:
    config = Config.from_dict(results["config"])
    architecture = (config.n_layers, config.d_model, config.n_heads, config.context_length) == (6, 384, 6, 256)
    assignment_budget = config.target_tokens == 25_000_000
    return {
        "architecture": "verified" if architecture else "non-assignment configuration",
        "data_and_tokenizer_integrity": "verified",
        "fixed_batch_overfit": "verified",
        "approximately_25m_targets": "verified" if assignment_budget else "pilot only",
        "loss_curves_and_best_perplexity": "included",
        "training_time_and_throughput": "included",
        "gpu_memory": "MPS allocator peak" if results.get("memory_measurement") == "MPS allocator peak" and results.get("peak_gpu_memory_bytes") is not None else
                      "sampled lower bound; exact MPS peak not recorded" if results["device"].startswith("mps") else
                      "CUDA allocator peak" if results["device"].startswith("cuda") else "GPU not used",
        "nine_fixed_setting_continuations": "included",
        "qualitative_observations": "included; review before submission" if notes_present else "author review needed",
        "handwritten_prediction_and_interpretation": "author must supply",
        "single_submission_pdf_and_tool_acknowledgment": "author must assemble and review",
    }


def _display(text: str) -> str:
    # Show decoding errors explicitly without changing the saved text or token ids.
    return "".join(
        "\\uFFFD" if char == "\ufffd" else
        char if char.isprintable() or char in "\n\t" else
        f"\\x{ord(char):02X}" if ord(char) <= 0xFF else f"\\u{ord(char):04X}"
        for char in text
    )


def build_report(run_dir: Path, output_dir: Path, notes_path: Path | None = None) -> dict:
    results = json.loads((run_dir / "results.json").read_text())
    verification = json.loads((output_dir / "verification.json").read_text())
    samples = json.loads((output_dir / "samples.json").read_text())
    if not verification["passed"] or verification["source_results_sha256"] != sha256_file(run_dir / "results.json"):
        raise ValueError("Report verification is missing or stale")
    validate_samples(samples, results)
    config = Config.from_dict(results["config"])
    if notes_path is not None and not notes_path.is_file():
        raise ValueError(f"Requested qualitative notes file is missing: {notes_path}")
    notes_path = notes_path or run_dir / "report/observations.md"
    notes = notes_path.read_text().strip() if notes_path.exists() else ""
    requirements = requirement_status(results, bool(notes))
    write_json(output_dir / "requirements.json", requirements)
    memory = results["peak_gpu_memory_bytes"]
    metrics = [
        ("Processed targets", f"{results['tokens_seen']:,}"),
        ("Optimizer steps", f"{results['optimizer_steps']:,}"),
        ("Parameters", f"{results['parameters']:,}"),
        ("Best validation loss (nats/token)", f"{results['best_validation_loss']:.6f}"),
        ("Perplexity exp(validation loss)", f"{results['best_perplexity']:.6f}"),
        ("Training time (seconds)", f"{results['training_seconds']:.3f}"),
        ("Wall time (seconds)", f"{results['wall_seconds']:.3f}"),
        ("Throughput (targets/second)", f"{results['tokens_per_second']:.2f}"),
        ("GPU memory (bytes)", str(memory) if memory is not None else "Not applicable"),
        ("Memory measurement", results["memory_measurement"]),
        ("Memory scope", results.get("memory_scope", "See measurement description; legacy run")),
        ("Device / precision", f"{results['device']} / {results['precision']}"),
        ("Fixed validation targets", f"{results['validation_target_count']:,}"),
    ]
    hyperparameters = [("Optimizer", "AdamW"), ("Learning-rate schedule", "linear warmup, then cosine decay")]
    hyperparameters += [(name, str(getattr(config, name))) for name in (
        "seed", "n_layers", "d_model", "n_heads", "context_length", "learning_rate", "min_learning_rate",
        "warmup_tokens", "target_tokens", "microbatch_size", "effective_batch_size", "adam_betas",
        "weight_decay", "grad_clip", "dropout",
    )]
    title = "TinyGPT assignment results" if requirements["approximately_25m_targets"] == "verified" else "TinyGPT pilot results — not the 25M-token assignment run"
    scope = (f"Validation uses the supplied reference notebook's fixed {results['validation_target_count']:,}-target prefix, "
             "not every token in the prepared validation split. Throughput excludes validation and checkpoint writes. "
             f"Memory measurement: {results['memory_measurement']}. Allocator peaks exclude driver-only allocations and total system memory; sampled measurements are lower bounds. "
             "The assignment's 1.3–1.5 loss example is not a guaranteed outcome.")
    sampling = (f"Temperature {config.temperature}; top-k {config.top_k}; {config.max_new_tokens} new tokens; "
                f"seed {config.seed} reset before every prompt; early stopping {config.early_stop}. "
                "EOS is hidden in decoded text, so continuations may contain multiple stories. "
                "Literal \\uFFFD marks a decoding replacement character; \\xNN marks a control character. "
                "These originate in the generated text, not the PDF font. Original text and token ids remain in samples.json.")
    caption = ("Every raw training-step loss is shown without smoothing. Dashed lines connect discrete validation "
               "measurements; no measurements are implied between markers. This is one run, without uncertainty bands.")
    md = [f"# {title}", "", "## 1. Prediction", "",
          "Insert your own handwritten prediction recorded before training. This report does not invent a retrospective prediction.",
          "", "## 2. Numerical results", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {name} | {value} |" for name, value in metrics]
    md += ["", scope, "", "![Training and validation loss](loss_curve.png)", "", caption,
           "", "### Hyperparameters", "", "| Setting | Value |", "|---|---|"]
    md += [f"| {name} | {value} |" for name, value in hyperparameters]
    md += ["", "### Story comparison", "", sampling, ""]
    for stage, entry in samples["stages"].items():
        md += [f"#### {stage.title()} — {entry['tokens_seen']:,} training targets", ""]
        for i, sample in enumerate(entry["samples"], 1):
            md += [f"**Prompt {i}:** {sample['prompt']}", "", _display(sample["continuation"]), ""]
    md += ["## 3. Interpretation", "", "Add your own handwritten interpretation for submission.", "",
           "### Supporting observations", "", notes or "Qualitative review pending: compare grammar, repetition, character identities, object consistency, and plot logic across the nine outputs.",
           "", "## Requirement checklist", ""]
    md += [f"- {name}: {value}" for name, value in requirements.items()]
    md += ["", "## Provenance", "", f"Results SHA-256: `{verification['source_results_sha256']}`.",
           f"Tokenizer SHA-256: `{results['tokenizer_sha256']}`.",
           "Data preparation follows the supplied TinyStories_Reference.ipynb. Codex assisted with implementation and reporting; acknowledge tools in your submission.",
           "Figure methodology used the scientific-visualization skill: Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).",
           "", "Full data: [samples](samples.json), [loss points](loss_points.csv), [checkpoint verification](verification.json), [requirement status](requirements.json).", ""]
    (output_dir / "report.md").write_text("\n".join(md))

    # Use Matplotlib's bundled fonts so the PDF is portable and supports the
    # Unicode characters in the generated samples.
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in (("Report", "DejaVuSans.ttf"), ("ReportBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")
    styles = getSampleStyleSheet()
    for name in ("Normal", "Title", "Heading1", "Heading2", "Heading3"):
        styles[name].fontName = "Report" if name == "Normal" else "ReportBold"
    styles["Normal"].fontSize = 9.5
    styles["Normal"].leading = 13
    styles["Normal"].spaceAfter = 7
    styles.add(ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8.5, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=11))

    def paragraph(text, style="Normal"):
        text = _display(str(text)).replace("—", "-").replace("–", "-").replace("‑", "-")
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    def table(rows):
        data = [[paragraph(k.replace("_", " "), "Cell"), paragraph(v, "Cell")] for k, v in rows]
        result = Table(data, colWidths=[205, 300], hAlign="LEFT")
        result.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f2f5f8"), colors.white]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return result

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(45, 25, "TinyGPT | Question 3")
        canvas.drawRightString(A4[0] - 45, 25, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(output_dir / "report.pdf"), pagesize=A4,
                                 rightMargin=45, leftMargin=45, topMargin=38, bottomMargin=42,
                                 title=title, author="TinyGPT experiment")
    pages = [paragraph(title, "Title"), paragraph("1. Prediction", "Heading2"), paragraph(md[4]),
             paragraph("2. Numerical results", "Heading2"), table(metrics), Spacer(1, 10),
             paragraph(scope, "Small"), PageBreak(), paragraph("Learning curves", "Heading1"),
             Image(str(output_dir / "loss_curve.png"), width=480, height=288),
             paragraph(caption, "Small"), paragraph("Hyperparameters", "Heading2"), table(hyperparameters)]
    for stage, entry in samples["stages"].items():
        pages += [PageBreak(), paragraph(f"Story comparison: {stage}", "Heading1"),
                  paragraph(f"{entry['tokens_seen']:,} training targets", "Heading3"),
                  paragraph(sampling, "Small")]
        for i, sample in enumerate(entry["samples"], 1):
            pages += [paragraph(f"Prompt {i}: {sample['prompt']}", "Heading3"),
                      paragraph(sample["continuation"])]
    pages += [PageBreak(), paragraph("3. Interpretation", "Heading1"),
              paragraph("Add your own handwritten interpretation for submission."),
              paragraph("Supporting observations", "Heading2")]
    for block in (notes or "Qualitative review pending.").split("\n\n"):
        if block.strip():
            pages.append(paragraph(" ".join(block.splitlines())))
    pages += [PageBreak(), paragraph("Requirement checklist", "Heading1"), table(list(requirements.items())),
              paragraph("Provenance", "Heading2"),
              paragraph(f"Results SHA-256: {verification['source_results_sha256']}", "Small"),
              paragraph(f"Tokenizer SHA-256: {results['tokenizer_sha256']}", "Small"),
              paragraph("Data recipe: supplied TinyStories_Reference.ipynb. Codex assisted with implementation and reporting; acknowledge tools in your submission.", "Small"),
              paragraph("Figure methodology: Kassis et al. (2026), Scientific Agent Skills. https://doi.org/10.48550/arXiv.2609.00065", "Small")]
    document.build(pages, onFirstPage=footer, onLaterPages=footer)
    return requirements
