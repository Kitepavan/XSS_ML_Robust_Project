from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, HRFlowable, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "xssharden_project_report.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1F6F9F")
TEAL = colors.HexColor("#0E7490")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE_BLUE = colors.HexColor("#EAF4F8")
PALE_TEAL = colors.HexColor("#E8F6F5")
LINE = colors.HexColor("#CBD5E1")


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=self.header_footer)])

    def header_footer(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        if doc.page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, height - 9 * mm, width, 9 * mm, stroke=0, fill=1)
        else:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(doc.leftMargin, height - 14 * mm, width - doc.rightMargin, height - 14 * mm)
            canvas.setFont("Helvetica-Bold", 7.5)
            canvas.setFillColor(NAVY)
            canvas.drawString(doc.leftMargin, height - 11 * mm, "XSSHARDEN")
            canvas.setFont("Helvetica", 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(width - doc.rightMargin, height - 11 * mm, "RESEARCH PROJECT REPORT")
        canvas.setStrokeColor(LINE)
        canvas.line(doc.leftMargin, 14 * mm, width - doc.rightMargin, 14 * mm)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 10 * mm, "AI-driven adversarial testing and selective hardening")
        canvas.drawRightString(width - doc.rightMargin, 10 * mm, f"{doc.page:02d}")
        canvas.restoreState()


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=TEAL, spaceAfter=7))
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, alignment=TA_LEFT, textColor=NAVY, spaceAfter=12))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, textColor=MUTED, spaceAfter=18))
    styles.add(ParagraphStyle(name="CoverMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=INK, spaceAfter=3))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=NAVY, spaceBefore=12, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.8, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.25, leading=13.4, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportBullet", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.05, leading=12.4, leftIndent=13, firstLineIndent=-8, textColor=INK, spaceAfter=3))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=9.55, leading=13.5, leftIndent=10, rightIndent=10, textColor=NAVY, spaceBefore=4, spaceAfter=8, borderColor=TEAL, borderWidth=0.8, borderPadding=9, backColor=PALE_TEAL, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name="Quote", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, leftIndent=11, rightIndent=11, textColor=NAVY, spaceBefore=4, spaceAfter=8, borderColor=BLUE, borderWidth=0.8, borderPadding=9, backColor=PALE_BLUE, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name="Pipeline", parent=styles["BodyText"], fontName="Courier-Bold", fontSize=8.5, leading=12.5, leftIndent=9, rightIndent=9, textColor=NAVY, spaceBefore=4, spaceAfter=8, borderColor=LINE, borderWidth=0.6, borderPadding=9, backColor=colors.HexColor("#F5F8FA")))
    styles.add(ParagraphStyle(name="Table", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.3, textColor=INK))
    styles.add(ParagraphStyle(name="TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.9, leading=10.3, textColor=NAVY))
    styles.add(ParagraphStyle(name="Reference", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.35, leading=11.3, leftIndent=13, firstLineIndent=-13, textColor=INK, spaceAfter=7))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontName="Helvetica", fontSize=7.4, leading=9.5, textColor=MUTED))
    return styles


def P(text, style):
    return Paragraph(text, style)


def bullet(text, styles):
    return P(f"&#8226; {text}", styles["ReportBullet"])


def table(data, widths, styles, header_bg=PALE_BLUE):
    rows = []
    for row_index, row in enumerate(data):
        rows.append([cell if isinstance(cell, Paragraph) else P(str(cell), styles["TableHead" if row_index == 0 else "Table"]) for cell in row])
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    return t


def section(title, styles):
    return [P(title, styles["Section"]), HRFlowable(width="100%", thickness=1.1, color=TEAL, spaceAfter=8)]


def build():
    styles = make_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=24 * mm, bottomMargin=20 * mm, title="XSSHarden Project Report", author="XSSHarden Research Team", subject="AI-driven adversarial testing and selective hardening of ML-based XSS detectors")
    story = []

    story += [Spacer(1, 16 * mm), P("XSSHARDEN  /  RESEARCH REPORT", styles["Kicker"]), P("AI-Driven Adversarial Testing and Selective Hardening of Machine-Learning-Based XSS Detectors", styles["CoverTitle"]), P("Research problem, closest related work, proposed novelty, evaluation design, and case study", styles["CoverSub"]), HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=18), P("PROJECT STATUS", styles["Kicker"]), P("Foundation implementation in progress; empirical results are not yet available.", styles["Quote"]), Spacer(1, 12 * mm), P("Prepared by", styles["Small"]), P("XSSHarden Research Team", styles["CoverMeta"]), P("Scope", styles["Small"]), P("Reflected XSS in an HTML body context", styles["CoverMeta"]), P("Deliverable", styles["Small"]), P("Reproducible defensive research prototype and evaluation pipeline", styles["CoverMeta"]), Spacer(1, 18 * mm), P("Submission note", styles["Small"]), P("This report distinguishes established components from the project-specific research contribution. It makes no empirical success or failure claim before the planned experiments are run.", styles["Body"]), PageBreak()]

    story += section("1. Research Problem", styles)
    story += [P("Machine-learning-based detectors can classify web inputs as benign or malicious by learning patterns from labelled XSS datasets. They may perform strongly on standard benchmark examples, but that performance does not guarantee robustness against previously unseen payload forms. A malicious input can be rewritten through encoding, whitespace changes, syntax changes, capitalization, alternate HTML structures, or different JavaScript construction while preserving its ability to execute.", styles["Body"]), P("Large language models make it inexpensive to produce many candidate variants. However, a generated string is not automatically a working attack. Some outputs may be syntactically malformed, incompatible with the target injection context, or unable to execute in a real browser. Counting every generated string as a successful adversarial example can therefore overestimate the practical threat.", styles["Body"]), P("The research problem is:", styles["Subsection"]), P("Can behaviorally valid, detector-challenging XSS variants be selected for training so that an ML-based XSS detector becomes more robust against fresh and previously unseen attack styles without causing unacceptable false positives on clean traffic?", styles["Quote"]), P("The primary scope is reflected XSS in an HTML body context. The deliverable is a reproducible research prototype and evaluation pipeline, not a commercial WAF or a full production web-security deployment.", styles["Body"])]

    story += section("2. Existing Solution and Closest Related Work", styles)
    story += [P("A plain classifier baseline is necessary for controlled comparison, but it is not the state of the art. The planned baseline uses character-level TF-IDF with Logistic Regression. A second detector uses lexical and handcrafted structural features with XGBoost or LightGBM. Both detectors share one feature pipeline so that downstream differences are attributable to the experimental arms rather than inconsistent preprocessing.", styles["Body"]), P("Baseline flow", styles["Subsection"]), P("XSS dataset  ->  cleaning and splitting  ->  feature extraction  ->  ML detector  ->  benign / malicious prediction", styles["Pipeline"]), P("CLOSEST RELATED WORK", styles["Kicker"]), P("Gabbireddy and Saha's 2026 study, <link href='https://arxiv.org/abs/2604.19526' color='#1F6F9F'>Evaluating LLM-Generated Obfuscated XSS Payloads for Machine Learning-Based Detection</link> [1], already combines deterministic mutation, LLM-based generation, browser-based behavioral validation, and downstream classifier comparison. It is the appropriate state-of-the-art comparison point for XSSHarden, rather than a naive train-a-classifier baseline.", styles["Callout"]), P("XSSHarden is a targeted extension of that pipeline. Gabbireddy and Saha use runtime validity primarily as a filter and report a small augmentation setting; they do not select valid examples according to whether they fool the detector, compare against a budget-matched random-valid control, or reserve mutation categories for a leakage-controlled final test. This project tests whether that additional selection layer produces robustness gains beyond validity filtering and additional data alone. The framing follows the problem-space perspective of Pierazzi et al. [2]: an adversarial example must remain realizable in the target environment, not merely appear adversarial in feature space.", styles["Body"]), P("Important evaluation requirements", styles["Subsection"]), bullet("Calibrate the decision threshold on clean validation data at a fixed operating point, such as a target false-positive rate.", styles), bullet("Keep train, validation, clean-test, adversarial-development, and final adversarial-test data explicitly separated.", styles), bullet("Use browser execution in a controlled local sandbox as the validity decision; the LLM is not the validator.", styles)]

    story += section("3. Proposed Novelty and Contribution", styles)
    story += [P("The proposed contribution is a domain-specific extension and empirical comparison, not a claim that LLM attack generation or adversarial training is individually new. Building on the closest related work [1] and the realizability framing in [2], XSSHarden connects behavioral validity with detector impact and tests whether selection adds value beyond simply adding more valid data.", styles["Body"]), P("Novelty 1: Behavior-validated adversarial corpus", styles["Subsection"]), P("Candidate variants are executed against a controlled local vulnerable page using Playwright and headless Chromium. A variant is valid only when the intended probe is observed within a defined timeout. Invalid outputs remain recorded because the invalid-generation rate is itself a reportable result.", styles["Body"]), P("Novelty 2: Validity-gated detector-impact selection", styles["Subsection"]), P("Validity is a hard gate. After validation and deduplication, the system measures detector impact and ranks only valid variants. A possible impact score is I(x) = 1 - P(malicious | x), where a lower malicious probability indicates a more difficult valid attack. The top examples are selected under a fixed budget and balanced across mutation categories.", styles["Body"]), P("Novelty 3: Fair control for selection claims", styles["Subsection"]), P("The selective arm is compared with a budget-matched random-valid arm drawn from the same eligible pool. If selective augmentation outperforms random-valid augmentation, the evidence supports the value of detector-impact selection rather than merely the value of adding more valid examples.", styles["Body"]), P("Conservative contribution statement", styles["Subsection"]), P("A behavior-validated, detector-impact-based selective augmentation pipeline for ML-based XSS detection, evaluated at meaningful augmentation scale using leakage-controlled, category-holdout adversarial testing and compared with naive and budget-matched random-valid augmentation.", styles["Callout"])]

    story += section("4. Overall Proposed Solution", styles)
    story += [P("The complete system is a command-line research pipeline. It begins with an audited dataset and ends with a report containing clean performance, validity statistics, the realizability gap, adversarial evasion, and comparisons across the four hardening arms.", styles["Body"]), table([["Stage", "Purpose", "Required safeguard"], ["Dataset", "Clean, deduplicate, audit labels, and split records.", "Source/cluster-aware splits; no leakage."], ["Baseline", "Train LR and XGBoost/LightGBM detectors.", "One shared feature pipeline."], ["Generation", "Create deterministic and LLM-based variants.", "Keep provenance and varied probes."], ["Validation", "Check whether variants execute in Chromium.", "Controlled local sandbox; retain invalid records."], ["Attack and selection", "Measure evasion and rank valid difficult examples.", "Validity hard gate; fixed budget."], ["Hardening and evaluation", "Train four arms and evaluate fresh attacks.", "Category-holdout final test."]], [29 * mm, 70 * mm, 71 * mm], styles), Spacer(1, 7), P("Four-arm comparison", styles["Subsection"]), table([["Arm", "Training data", "Question answered"], ["A - Baseline", "Original training data only", "How robust is the unaugmented detector?"], ["B - Naive", "Original data + all valid variants", "Does broad valid augmentation help?"], ["C - Random-valid", "Original data + random valid subset of size B", "Does any valid data of this size help?"], ["D - Selective", "Original data + top-B valid variants by impact", "Does impact selection add value?"]], [31 * mm, 76 * mm, 63 * mm], styles)]

    story += section("5. Simple Case Study", styles)
    story += [P("Consider a web application that reflects a user-controlled search value into an HTML response. The organization deploys an ML classifier before the value reaches the application. The classifier correctly identifies many familiar XSS patterns in its training and clean test data.", styles["Body"]), P("An attacker then produces alternative forms of the same attack using encoding transformations, whitespace changes, event-handler substitutions, or semantic rewrites. Some LLM outputs look suspicious but do not execute in the target context. Other outputs execute successfully but are classified as benign by the detector. Only the latter are practical adversarial examples for this experiment.", styles["Body"]), P("Proposed workflow", styles["Subsection"]), P("Seed attack  ->  programmatic or LLM variant generation  ->  browser execution check  ->  detector prediction  ->  impact measurement  ->  valid-example selection  ->  retraining  ->  independent evaluation", styles["Pipeline"]), P("Suppose 8,000 unique generated variants are behaviorally valid and the augmentation budget is B = 1,000. Arm C randomly samples 1,000 valid variants from the eligible development pool. Arm D selects the 1,000 valid variants with the greatest detector impact, subject to the same category-balancing policy. Both arms then use identical training procedures and are evaluated on the same final adversarial test set.", styles["Body"]), P("If Arm D has lower valid attack success rate than Arm C while maintaining acceptable clean false-positive performance, the result supports the selective strategy. If both arms improve similarly, the conclusion is that validity filtering or additional valid data matters, but detector-impact selection is not shown to provide extra benefit. If Arm D performs worse, the selection method may be too narrow or may overfit to development-time attack patterns.", styles["Body"]), P("Safety boundary", styles["Subsection"]), P("All execution must occur inside the controlled local sandbox. The experiment is intended to measure defensive robustness, not to deploy payloads against external systems.", styles["Callout"])]

    story += section("6. Core Research Contribution", styles)
    story += [P("The central contribution is an evidence-based test of whether selecting behaviorally real, detector-challenging XSS variants is better than selecting valid variants at random. The random-valid arm is not an optional extra model; it is the counterfactual that makes the selective claim interpretable.", styles["Body"]), P("Primary research question", styles["Subsection"]), P("For ML-based XSS detectors, does behavior-validated, detector-impact-selected augmentation reduce evasion on a leakage-free, category-holdout independent test set more effectively than naive all-valid augmentation and budget-matched random-valid augmentation, while preserving acceptable clean-traffic performance?", styles["Quote"]), P("Expected evidence", styles["Subsection"]), bullet("A lower V-ASR for Arm D than Arm C supports the added value of selection.", styles), bullet("A lower V-ASR for both C and D than A supports valid-data augmentation generally.", styles), bullet("A similar result for C and D means the selection advantage is not demonstrated.", styles), bullet("Clean F1, precision, recall, and false-positive rate must be considered alongside adversarial robustness.", styles)]

    story += section("7. Planned Evaluation and Current Status", styles)
    story += [P("The final adversarial test must contain fresh seeds and mutation categories or styles held out from hardening. Thresholds are calibrated using clean validation data and then held fixed for adversarial comparisons. Each detector architecture should be retrained under all four arms with multiple random seeds, with confidence intervals and paired statistical tests reported where appropriate.", styles["Body"]), table([["Metric", "Interpretation"], ["V-ASR", "Proportion of valid adversarial payloads that evade detection; lower is better."], ["Realizability gap", "Difference between results on all generated variants and behaviorally valid variants."], ["Clean F1 / precision / recall", "Quality on ordinary held-out data."], ["Clean FPR", "Benign traffic incorrectly flagged; must remain acceptable."], ["Per-category evasion", "Whether robustness generalizes across mutation styles."]], [45 * mm, 125 * mm], styles), P("Current implementation status", styles["Subsection"]), P("The repository currently contains the package scaffold plus dataset schema, cleaning, normalization, and deduplication components. The leakage-safe split and reproducibility task has tests drafted but is not complete. ML training, browser validation, generation, selection, hardening, and final reporting remain future implementation stages. No empirical success or failure claim should be made until those experiments are run.", styles["Body"]), P("Conclusion", styles["Subsection"]), P("XSSHarden is best understood as a controlled robustness study. Its value depends less on producing a large number of generated payloads and more on proving that the examples are real attacks, that the train/test boundary is protected, and that selective augmentation beats a fair random-valid control without damaging clean-traffic behavior.", styles["Callout"])]

    story += section("References and Project Sources", styles)
    story += [P("[1] D. Gabbireddy and S. Saha, <i>Evaluating LLM-Generated Obfuscated XSS Payloads for Machine Learning-Based Detection</i>, arXiv preprint arXiv:2604.19526, 2026. <link href='https://arxiv.org/abs/2604.19526' color='#1F6F9F'>https://arxiv.org/abs/2604.19526</link>", styles["Reference"]), P("[2] F. Pierazzi, F. Pendlebury, J. Cortellazzi, and L. Cavallaro, <i>Intriguing Properties of Adversarial ML Attacks in the Problem Space</i>, in IEEE Symposium on Security and Privacy, 2020. <link href='https://arxiv.org/abs/1911.02142' color='#1F6F9F'>arXiv:1911.02142</link>.", styles["Reference"]), P("Project sources: <i>XSS_ML_Robustness_handoff_doc.md</i> and <i>xss-robustness-project-proposal.md</i>. These documents define the scope, architecture, evaluation arms, and research safeguards summarized in this report.", styles["Reference"])]
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
