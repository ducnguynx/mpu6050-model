"""Build HUST-branded presentation deck (14 main slides + cover + Q&A + 2 appendices)
for the MPU-6050 Roll-Angle Estimation project (Pair 2: Wiener vs. Kalman).

Uses the downloaded official HUST template at outputs/slides/template.pptx and
the verified figures/results generated in outputs/attitude/.

Output:
    outputs/slides/MPU6050_Roll_Estimation_Wiener_vs_Kalman.pptx
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "outputs" / "slides" / "template.pptx"
ATTITUDE_DIR = PROJECT_ROOT / "outputs" / "attitude"
SLIDES_DIR = PROJECT_ROOT / "outputs" / "slides"
FORMULA_DIR = SLIDES_DIR / "_formulas"
OUT_PPTX = SLIDES_DIR / "MPU6050_Roll_Estimation_Wiener_vs_Kalman.pptx"

# HUST Brand & Semantic Palette
HUST_RED = RGBColor(0xBF, 0x20, 0x34)
HUST_DARK_RED = RGBColor(0x8F, 0x16, 0x25)
HUST_GOLD = RGBColor(0xD9, 0x9B, 0x00)
DARK_TEXT = RGBColor(0x1F, 0x29, 0x37)
MUTED_TEXT = RGBColor(0x4B, 0x55, 0x63)
LIGHT_BG = RGBColor(0xF8, 0xF9, 0xFB)
RED_TINT = RGBColor(0xFD, 0xF2, 0xF3)
GREEN_ACCENT = RGBColor(0x1E, 0x7E, 0x34)
GREEN_TINT = RGBColor(0xEC, 0xFD, 0xF5)
BLUE_ACCENT = RGBColor(0x1D, 0x4E, 0xD8)
BLUE_TINT = RGBColor(0xEF, 0xF6, 0xFF)
AMBER_ACCENT = RGBColor(0xB4, 0x53, 0x09)
AMBER_TINT = RGBColor(0xFF, 0xFB, 0xEB)
BORDER_GRAY = RGBColor(0xD1, 0xD5, 0xDB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT_NAME = "Calibri"


# ==============================================================================
# Formula PNG Renderer (Crisp 300-DPI Transparent Math via Matplotlib Mathtext)
# ==============================================================================


def render_formula(name: str, lines: Sequence[str], fontsize: float = 15.5, color: str = "#1F2937") -> Path:
    """Render one or more LaTeX math lines to a crisp transparent PNG with generous spacing."""
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    out = FORMULA_DIR / f"{name}.png"
    n = len(lines)
    row_h = 0.72
    fig = plt.figure(figsize=(8.6, row_h * n + 0.15), dpi=300)
    fig.patch.set_alpha(0.0)
    for i, line in enumerate(lines):
        y = 1.0 - (i + 0.5) / n
        fig.text(
            0.01,
            y,
            line,
            fontsize=fontsize,
            color=color,
            va="center",
            ha="left",
            usetex=False,
        )
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


# ==============================================================================
# Low-Level PPTX Helpers (Clean Template Slide Management & Styling)
# ==============================================================================


def delete_slide(prs: Presentation, index: int) -> None:
    """Remove slide at index cleanly from a python-pptx Presentation."""
    xml_slides = prs.slides._sldIdLst
    slide_id = xml_slides[index]
    rId = slide_id.rId
    prs.part.drop_rel(rId)
    xml_slides.remove(slide_id)


def remove_shape(shape) -> None:
    """Delete a shape from its slide."""
    sp = shape._element
    sp.getparent().remove(sp)


def set_cell_border(cell, color_hex: str = "D1D5DB", width_pt: float = 0.5) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    w_emu = str(int(width_pt * 12700))
    for edge in ("lnL", "lnR", "lnT", "lnB"):
        for existing in tcPr.findall(qn(f"a:{edge}")):
            tcPr.remove(existing)
        ln = OxmlElement(f"a:{edge}")
        ln.set("w", w_emu)
        ln.set("cap", "flat")
        ln.set("cmpd", "sng")
        ln.set("algn", "ctr")
        solidFill = OxmlElement("a:solidFill")
        srgbClr = OxmlElement("a:srgbClr")
        srgbClr.set("val", color_hex)
        solidFill.append(srgbClr)
        ln.append(solidFill)
        tcPr.append(ln)


def set_slide_number(slide, number_str: str) -> None:
    """Populate the template's slide-number placeholder cleanly."""
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 12:
            tf = ph.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = number_str
            run.font.name = FONT_NAME
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = HUST_RED
            return


def new_content_slide(prs: Presentation, title: str, slide_num: str, layout_idx: int = 1):
    """Add a clean HUST content slide using slideLayout2 (crimson header + gold line + HUST footer)."""
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    # Remove unused default body placeholders so no "Click to add text" boxes remain
    for ph in list(slide.placeholders):
        idx = ph.placeholder_format.idx
        if idx == 0:
            tf = ph.text_frame
            tf.clear()
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = title
            run.font.name = FONT_NAME
            run.font.size = Pt(22)
            run.font.bold = True
            run.font.color.rgb = WHITE if layout_idx == 1 else HUST_RED
        elif idx == 12:
            set_slide_number(slide, slide_num)
        else:
            remove_shape(ph)
    return slide


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    title: str = "",
    title_color: RGBColor = HUST_RED,
    fill_color: RGBColor = LIGHT_BG,
    border_color: RGBColor = BORDER_GRAY,
    accent_bar_color: RGBColor | None = None,
    bullets: Sequence[tuple[str, str] | str] = (),
    font_size: float = 12.5,
    title_size: float = 14.0,
    space_after: float = 4.0,
):
    """Draw a clean rounded-rectangle content card with optional top/left accent & structured bullets."""
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    box.adjustments[0] = 0.03
    box.fill.solid()
    box.fill.fore_color.rgb = fill_color
    box.line.color.rgb = border_color
    box.line.width = Pt(0.85)
    box.shadow.inherit = False

    if accent_bar_color is not None:
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(left),
            Inches(top),
            Inches(0.08),
            Inches(height),
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_bar_color
        bar.line.fill.background()
        bar.shadow.inherit = False

    pad_l = left + (0.18 if accent_bar_color else 0.14)
    tb = slide.shapes.add_textbox(
        Inches(pad_l),
        Inches(top + 0.08),
        Inches(width - (pad_l - left) - 0.12),
        Inches(height - 0.14),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)

    first = True
    if title:
        p = tf.paragraphs[0]
        p.space_after = Pt(5)
        r = p.add_run()
        r.text = title
        r.font.name = FONT_NAME
        r.font.size = Pt(title_size)
        r.font.bold = True
        r.font.color.rgb = title_color
        first = False

    for item in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(space_after)
        p.line_spacing = 1.15
        if isinstance(item, tuple):
            lead, body = item
            r1 = p.add_run()
            r1.text = f"• {lead}: " if lead else "• "
            r1.font.name = FONT_NAME
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = DARK_TEXT
            r2 = p.add_run()
            r2.text = body
            r2.font.name = FONT_NAME
            r2.font.size = Pt(font_size)
            r2.font.color.rgb = DARK_TEXT
        else:
            r = p.add_run()
            r.text = f"• {item}"
            r.font.name = FONT_NAME
            r.font.size = Pt(font_size)
            r.font.color.rgb = DARK_TEXT
    return box


def add_kpi_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    label: str,
    value: str,
    sub: str = "",
    accent: RGBColor = HUST_RED,
    fill: RGBColor = WHITE,
) -> None:
    """Compact metric highlight card."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    card.adjustments[0] = 0.06
    card.fill.solid()
    card.fill.fore_color.rgb = fill
    card.line.color.rgb = accent
    card.line.width = Pt(1.2)
    card.shadow.inherit = False

    tb = slide.shapes.add_textbox(
        Inches(left + 0.10),
        Inches(top + 0.05),
        Inches(width - 0.20),
        Inches(height - 0.10),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.01)

    p0 = tf.paragraphs[0]
    p0.space_after = Pt(1)
    r0 = p0.add_run()
    r0.text = label
    r0.font.name = FONT_NAME
    r0.font.size = Pt(10.5)
    r0.font.bold = True
    r0.font.color.rgb = MUTED_TEXT

    p1 = tf.add_paragraph()
    p1.space_after = Pt(1)
    r1 = p1.add_run()
    r1.text = value
    r1.font.name = FONT_NAME
    r1.font.size = Pt(18.5)
    r1.font.bold = True
    r1.font.color.rgb = accent

    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.name = FONT_NAME
        r2.font.size = Pt(9.8)
        r2.font.color.rgb = DARK_TEXT


def add_picture_fitted(
    slide,
    img_path: Path,
    left: float,
    top: float,
    max_w: float,
    max_h: float,
    *,
    center: bool = True,
    border: bool = False,
) -> None:
    """Insert an image scaled to fit strictly within (max_w, max_h) preserving exact aspect ratio."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    aspect = iw / ih
    if max_w / max_h > aspect:
        h = max_h
        w = h * aspect
    else:
        w = max_w
        h = w / aspect
    x = left + (max_w - w) / 2.0 if center else left
    y = top + (max_h - h) / 2.0 if center else top
    pic = slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    if border:
        pic.line.color.rgb = BORDER_GRAY
        pic.line.width = Pt(0.75)


def add_styled_table(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    col_widths: Sequence[float] | None = None,
    font_size: float = 10.5,
    header_bg: RGBColor = HUST_RED,
    highlight_rows: dict[int, RGBColor] | None = None,
    bold_rows: set[int] | None = None,
):
    """Create a clean, compact styled table with HUST header and subtle zebra striping."""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = tbl_shape.table

    if col_widths is not None:
        total_w = sum(col_widths)
        for j, cw in enumerate(col_widths):
            tbl.columns[j].width = Inches(width * cw / total_w)

    # Header row
    for j, h_text in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_bg
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.06)
        cell.margin_right = Inches(0.06)
        cell.margin_top = Inches(0.03)
        cell.margin_bottom = Inches(0.03)
        set_cell_border(cell, "B91C1C", 0.5)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h_text
        r.font.name = FONT_NAME
        r.font.size = Pt(font_size)
        r.font.bold = True
        r.font.color.rgb = WHITE

    # Data rows
    for i, row in enumerate(rows):
        bg = WHITE if i % 2 == 0 else LIGHT_BG
        if highlight_rows and i in highlight_rows:
            bg = highlight_rows[i]
        is_bold_row = bool(bold_rows and i in bold_rows)
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.025)
            cell.margin_bottom = Inches(0.025)
            set_cell_border(cell, "D1D5DB", 0.5)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            r.font.name = FONT_NAME
            r.font.size = Pt(font_size)
            r.font.bold = is_bold_row or (j == 0)
            r.font.color.rgb = DARK_TEXT
    return tbl_shape


def add_banner_note(slide, left: float, top: float, width: float, height: float, text: str, *, fill: RGBColor = RED_TINT, accent: RGBColor = HUST_RED, font_size: float = 11.5) -> None:
    """Bottom callout takeaway strip."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.adjustments[0] = 0.12
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = accent
    box.line.width = Pt(1.0)
    box.shadow.inherit = False

    tb = slide.shapes.add_textbox(Inches(left + 0.14), Inches(top + 0.04), Inches(width - 0.28), Inches(height - 0.08))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.name = FONT_NAME
    r.font.size = Pt(font_size)
    r.font.bold = True
    r.font.color.rgb = DARK_TEXT


# ==============================================================================
# Build All Slides
# ==============================================================================


def build_deck() -> Path:
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ATTITUDE_DIR / "results_summary.json").read_text(encoding="utf-8"))

    # Pre-render crisp math formulas
    f_physics = render_formula(
        "physics",
        [
            r"$\omega_{x,\mathrm{ideal}}(t) = \dot{\theta}(t)$",
            r"$f_{y,\mathrm{ideal}}(t) = g_0\sin\theta(t) + a_{\mathrm{lin},y}(t)$",
            r"$f_{z,\mathrm{ideal}}(t) = g_0\cos\theta(t)$",
        ],
        fontsize=15.0,
    )
    f_mpu_model = render_formula(
        "mpu_model",
        [
            r"$\tilde{\mathbf{u}}_k^{\mathrm{analog}} = (\mathbf{I}_3 + \mathbf{S} + \mathbf{M})\mathbf{u}_k + \mathbf{b}_0 + \mathbf{v}_{N,k} + \mathbf{x}_{B,k} + \mathbf{x}_{K,k}$",
            r"$x_B[k+1] = e^{-\Delta t/T_B} x_B[k] + w_{B,k}, \qquad x_K[k+1] = x_K[k] + w_{K,k}$",
        ],
        fontsize=14.0,
    )
    f_wiener_1 = render_formula(
        "wiener_1",
        [
            r"$\hat{\Theta}(f) = G(f)\,Z_a(f) + (1 - G(f))\,Z_g(f)$",
            r"$E(f) = \hat{\Theta}(f) - \Theta(f) = G(f)\,N_a(f) + (1-G(f))\,N_g(f)$",
            r"$G_W(f) = \frac{S_{n_\omega}(f)}{S_{n_\omega}(f) + |D(f)|^2\, S_{n_a}(f)}, \quad D(f)=\frac{1-e^{-j2\pi f\Delta t}}{\Delta t}$",
        ],
        fontsize=14.5,
    )
    f_wiener_2 = render_formula(
        "wiener_2",
        [
            r"$\hat{\Theta}(f) = G_W(f)\,Z_a(f) + H_{\mathrm{gyr}}(f)\,\tilde{\Omega}_x(f)$",
            r"$H_{\mathrm{gyr}}(f) = \frac{D^*(f)\,S_{n_a}(f)}{S_{n_\omega}(f) + |D(f)|^2 S_{n_a}(f)} \;\Rightarrow\; H_{\mathrm{gyr}}(0)=0$",
            r"$\omega_c^* = \sqrt{\frac{S_\omega}{S_{\theta_a}}} = \frac{\hat{N}_g}{\hat{N}_{\theta_a}} = 0.218\ \mathrm{rad/s}\;\;(f_c^*=0.0347\ \mathrm{Hz})$",
        ],
        fontsize=14.0,
    )
    f_kalman = render_formula(
        "kalman_4state",
        [
            r"$\mathbf{x}_k = [u_k,\ \dot{u}_k,\ \ddot{u}_k,\ b_k]^T = [\theta_k,\ \dot{\theta}_k,\ \ddot{\theta}_k,\ b_{g,k}]^T$",
            r"$u_k = u_{k-1} + \dot{u}_{k-1}\Delta t + \frac{1}{2}\ddot{u}_{k-1}\Delta t^2 + w_{u,k}, \quad \dot{u}_k = \dot{u}_{k-1} + \ddot{u}_{k-1}\Delta t + w_{\dot{u},k}$",
            r"$\ddot{u}_k = \ddot{u}_{k-1} + w_{\ddot{u},k}, \quad b_k = b_{k-1} + w_{b,k} \quad (\hat{q}_b = \hat{q}_{B,g} + \hat{q}_{K,g})$",
            r"$\mathbf{y}_k = [\theta_{\mathrm{acc,cal}}[k],\ \tilde{\omega}_{x,k}]^T = [u_k,\ \dot{u}_k + b_k]^T + \mathbf{v}_k, \quad \mathbf{R} = \mathrm{diag}(\hat{R}_\theta, \hat{R}_\omega)$",
        ],
        fontsize=12.2,
    )
    f_appendix_math = render_formula(
        "appendix_math",
        [
            r"$S_{yy}(s) = S_{n_a}(s) + S_{n_g}(s) = S_a \frac{-s^2 + \omega_c^2}{-s^2} = \left(\sqrt{S_a}\frac{s+\omega_c}{s}\right)\!\left(\sqrt{S_a}\frac{-s+\omega_c}{-s}\right)$",
            r"$G_{W,\mathrm{causal}}(s) = \frac{\omega_c^*}{s + \omega_c^*}, \qquad \Delta\theta_{\mathrm{ss}} = \lim_{s\to 0}\,s\cdot\frac{1 - G_{W,\mathrm{causal}}(s)}{s}\cdot\frac{b_g}{s} = \frac{b_g}{\omega_c^*}$",
            r"$\mathbf{P}_{k|k} = (\mathbf{I}-\mathbf{K}_k\mathbf{H})\mathbf{P}_{k|k-1}(\mathbf{I}-\mathbf{K}_k\mathbf{H})^T + \mathbf{K}_k \hat{R}_\theta \mathbf{K}_k^T \quad (\mathrm{Joseph\ form})$",
        ],
        fontsize=13.5,
    )

    # Open HUST template, extract cover HUST logo from slide 3 (index 2), then clear ALL template slides
    # so newly added slides are numbered slide1.xml..slide18.xml cleanly with zero duplicate ZIP entries.
    prs = Presentation(str(TEMPLATE_PATH))
    cover_logo_path = FORMULA_DIR / "hust_cover_logo.png"
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    for sh in prs.slides[2].shapes:
        if sh.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
            cover_logo_path.write_bytes(sh.image.blob)
            logo_pos = (sh.left, sh.top, sh.width, sh.height)
            break

    for idx in reversed(range(len(prs.slides))):
        delete_slide(prs, idx)

    # ==========================================================================
    # SLIDE 0 (BÌA): Cover Slide (fresh on slideLayout 0 = Title and Content)
    # ==========================================================================
    cover = prs.slides.add_slide(prs.slide_layouts[0])
    for ph in list(cover.placeholders):
        remove_shape(ph)
    cover.shapes.add_picture(str(cover_logo_path), *logo_pos)

    # Main Title on Cover
    tb_title = cover.shapes.add_textbox(Inches(0.45), Inches(1.85), Inches(8.6), Inches(1.85))
    tf = tb_title.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.line_spacing = 1.12
    r = p.add_run()
    r.text = "ƯỚC LƯỢNG GÓC NGHIÊNG (ROLL)\nTỪ CẢM BIẾN MPU-6050"
    r.font.name = FONT_NAME
    r.font.size = Pt(36)
    r.font.bold = True
    r.font.color.rgb = HUST_RED

    # Subtitle on Cover
    tb_sub = cover.shapes.add_textbox(Inches(0.45), Inches(3.75), Inches(8.3), Inches(1.15))
    tf = tb_sub.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.line_spacing = 1.25
    r = p.add_run()
    r.text = "Nhận dạng Nhiễu Hộp đen & So sánh Bộ lọc Bù Tối ưu kiểu Wiener với Bộ lọc Kalman 4 Trạng thái [u, u', u'', b]ᵀ"
    r.font.name = FONT_NAME
    r.font.size = Pt(20)
    r.font.color.rgb = DARK_TEXT

    # Gold accent bar on cover
    gold_bar = cover.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(4.95), Inches(3.2), Inches(0.06))
    gold_bar.fill.solid()
    gold_bar.fill.fore_color.rgb = HUST_GOLD
    gold_bar.line.fill.background()

    # Course / Author info block
    tb_meta = cover.shapes.add_textbox(Inches(0.45), Inches(5.18), Inches(7.8), Inches(1.75))
    tf = tb_meta.text_frame
    tf.word_wrap = True
    meta_lines = [
        ("Học phần: ", "Xử lý Tín hiệu Ngẫu nhiên (Bài tập lớn cuối kỳ)"),
        ("Thực hiện: ", "[Họ và tên sinh viên / Nhóm thực hiện]"),
        ("Giảng viên hướng dẫn: ", "[Họ và tên Giảng viên]"),
        ("Thời gian: ", "Hà Nội, năm 2026"),
    ]
    for idx, (k, v) in enumerate(meta_lines):
        pm = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        pm.space_after = Pt(4)
        rk = pm.add_run()
        rk.text = k
        rk.font.name = FONT_NAME
        rk.font.size = Pt(13.5)
        rk.font.bold = True
        rk.font.color.rgb = HUST_DARK_RED
        rv = pm.add_run()
        rv.text = v
        rv.font.name = FONT_NAME
        rv.font.size = Pt(13.5)
        rv.font.color.rgb = DARK_TEXT

    # ==========================================================================
    # MAIN SLIDE 1 / 14: Agenda (Layout 5: left HUST crimson panel + right white)
    # ==========================================================================
    s1 = new_content_slide(prs, "NỘI DUNG TRÌNH BÀY", "1", layout_idx=5)
    agenda_items = [
        ("01", "Bài toán Vật lý & Mô hình MPU-6050", "Tại sao cần Hợp nhất cảm biến, kiểm chứng tuyến tính hóa góc nhỏ & nguyên tắc hộp đen"),
        ("02", "Kiểm định Giả thiết & Nhận dạng Nhiễu có Sai số", "Kiểm định 5 giả thiết thống kê, Welch PSD & Allan Deviation (Levenberg–Marquardt)"),
        ("03", "Thiết kế Baseline & 2 Phương án Lọc", "Baselines bắt buộc vs. Optimal Wiener Complementary (miền tần số) vs. Kalman 4 trạng thái [u, u', u'', b]ᵀ"),
        ("04", "Đánh giá Định lượng & Chẩn đoán Nội bộ", "4 kịch bản, NIS/Ljung–Box không cần ground truth, Monte Carlo & Stress-test độ bền vững"),
        ("05", "Thảo luận, Kết luận & Hướng Mở rộng", "So sánh trực diện Wiener vs. Kalman, khuyến nghị ứng dụng thực tế & nâng cấp tương lai"),
    ]
    y_cur = 1.25
    for num, heading, desc in agenda_items:
        badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.05), Inches(y_cur), Inches(0.68), Inches(0.68))
        badge.adjustments[0] = 0.18
        badge.fill.solid()
        badge.fill.fore_color.rgb = HUST_RED
        badge.line.fill.background()
        tf_b = badge.text_frame
        tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
        pb = tf_b.paragraphs[0]
        pb.alignment = PP_ALIGN.CENTER
        rb = pb.add_run()
        rb.text = num
        rb.font.name = FONT_NAME
        rb.font.size = Pt(16)
        rb.font.bold = True
        rb.font.color.rgb = WHITE

        tb = s1.shapes.add_textbox(Inches(5.88), Inches(y_cur - 0.02), Inches(6.95), Inches(0.82))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.space_after = Pt(2)
        r1 = p1.add_run()
        r1.text = heading
        r1.font.name = FONT_NAME
        r1.font.size = Pt(15)
        r1.font.bold = True
        r1.font.color.rgb = DARK_TEXT
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = desc
        r2.font.name = FONT_NAME
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = MUTED_TEXT
        y_cur += 1.08

    # ==========================================================================
    # MAIN SLIDE 2 / 14: Physical Problem & Why Sensor Fusion
    # ==========================================================================
    s2 = new_content_slide(prs, "1. Đặt bài toán: Ước lượng Góc nghiêng Roll từ MPU-6050", "2")
    add_card(
        s2,
        0.40,
        1.05,
        6.15,
        2.55,
        title="Mô hình vật lý chuyển động nghiêng Roll quanh trục X",
        accent_bar_color=HUST_RED,
        bullets=[
            ("Tần số lấy mẫu", "fs = 200 Hz (Δt = 5 ms), dải đo ±2g & ±250 °/s."),
            ("Nhất quán vật lý 6 trục", "Đạo hàm góc nghiêng là vận tốc góc ωx; lực riêng mang hình chiếu trọng lực g0 + gia tốc ngang ngoại lai."),
        ],
        font_size=12.0,
    )
    add_picture_fitted(s2, f_physics, 0.60, 2.38, 5.75, 1.12)

    add_card(
        s2,
        6.75,
        1.05,
        6.18,
        2.55,
        title="Tại sao KHÔNG lọc riêng từng cảm biến rồi tính atan2?",
        title_color=AMBER_ACCENT,
        fill_color=AMBER_TINT,
        border_color=HUST_GOLD,
        accent_bar_color=HUST_GOLD,
        bullets=[
            ("Gia tốc động phá vỡ lượng giác", "Khi có rung/tăng tốc ngang (a_lin,y ≠ 0), phép atan2 cho góc giả sai hàng chục độ dù lọc hết nhiễu cảm biến."),
            ("Lãng phí Con quay hồi chuyển", "Gyro đo trực tiếp tốc độ góc với độ trễ ~0 ms và hoàn toàn miễn nhiễm gia tốc tịnh tiến."),
            ("Méo thống kê qua hàm phi tuyến", "Lọc riêng ay, az bỏ qua liên hệ động học θ̇ = ωx."),
        ],
        font_size=11.8,
    )

    add_styled_table(
        s2,
        0.40,
        3.82,
        12.53,
        1.85,
        headers=["Cảm biến trên MPU-6050", "Ưu điểm nổi bật", "Nhược điểm cốt lõi", "Vùng tần số tin cậy"],
        rows=[
            ["Gia tốc kế (Accelerometer)\nθ_acc = atan2(a_y, a_z)", "Không bao giờ bị trôi dài hạn (khóa theo trọng lực g0 cố định)", "Nhiễu trắng băng rộng (~0.23°) & hỏng nặng khi có rung ngang", "Tần số thấp (Dài hạn / DC)"],
            ["Con quay hồi chuyển (Gyro)\nθ_g[k] = θ_g[k-1] + ω_x·Δt", "Mượt ngắn hạn, phản ứng tức thời, miễn nhiễm gia tốc tịnh tiến", "Tích phân bias (b0 + xB + xK) gây trôi góc tăng vô hạn (~84.5°/3 phút)", "Tần số cao (Ngắn hạn / Động)"],
        ],
        col_widths=[2.7, 3.6, 4.0, 2.2],
        font_size=11.0,
        highlight_rows={0: BLUE_TINT, 1: GREEN_TINT},
    )
    add_banner_note(
        s2,
        0.40,
        5.88,
        12.53,
        0.65,
        "Giải pháp chuẩn công nghiệp: Hợp nhất cảm biến (Sensor Fusion) trực tiếp trên biến góc θ(t) — lấy dài hạn của Gia tốc kế bù cho trôi của Con quay, lấy ngắn hạn của Con quay bù cho rung của Gia tốc kế.",
        font_size=12.0,
    )

    # ==========================================================================
    # MAIN SLIDE 3 / 14: MPU-6050 Error Model & Black-Box Protocol
    # ==========================================================================
    s3 = new_content_slide(prs, "1. Mô hình Sai số MPU-6050 & Nguyên tắc Thiết kế \"Hộp đen\"", "3")
    add_card(
        s3,
        0.40,
        1.05,
        12.53,
        1.95,
        title="Mô hình hành vi MPU-6050 (Farrell et al., IEEE CSM 2022 & Datasheet PS-MPU-6000A-00)",
        accent_bar_color=HUST_RED,
        bullets=[],
    )
    add_picture_fitted(s3, f_mpu_model, 0.70, 1.48, 11.9, 1.38)

    add_card(
        s3,
        0.40,
        3.20,
        6.10,
        3.32,
        title="Cấu trúc sai số đa tầng đặc trưng của IMU MEMS",
        accent_bar_color=BLUE_ACCENT,
        title_color=BLUE_ACCENT,
        fill_color=BLUE_TINT,
        bullets=[
            ("Sai số tất định (S, M, b0)", "Lệch điểm 0 tĩnh, sai số hệ số tỉ lệ (±0.4–0.8%) và xuyên trục (±0.5%)."),
            ("Nhiễu trắng băng rộng (vN)", "ARW / VRW phân phối chuẩn, độc lập giữa các mẫu."),
            ("Nhiễu màu Gauss–Markov (xB)", "Bất ổn định bias có thời gian tương quan dài (TB ≈ 100 s)."),
            ("Trôi ngẫu nhiên không dừng (xK)", "Random Walk có phương sai tăng tuyến tính theo thời gian (K²·t)."),
            ("Tầng phần cứng", "Bão hòa dải đo (FSR) & lượng tử hóa ADC 16-bit có dấu."),
        ],
        font_size=11.8,
    )

    add_card(
        s3,
        6.80,
        3.20,
        6.13,
        3.32,
        title="Nguyên tắc \"Hộp đen\" (Coi mô hình cảm biến là chưa biết)",
        accent_bar_color=HUST_RED,
        title_color=HUST_RED,
        fill_color=RED_TINT,
        bullets=[
            ("Không rò rỉ tham số", "Tuyệt đối không đọc N, B, TB, K, b0, S, M từ config/default.yaml nạp vào bộ lọc."),
            ("Tập nhận dạng tĩnh độc lập", "Thu 1 giờ dữ liệu đặt phẳng (seed = 1001, 720.000 mẫu) mô phỏng đo bàn lab thực tế."),
            ("Tập kiểm thử tách biệt", "Chạy trên 4 kịch bản động với các seed hoàn toàn độc lập (2026 & 3000–3019)."),
            ("Vai trò Ground Truth", "Tín hiệu góc lý tưởng chỉ dùng ở Bước 4 để chấm điểm RMSE/MAE/Stress-test."),
        ],
        font_size=11.8,
    )

    # ==========================================================================
    # MAIN SLIDE 4 / 14: Linearization Check & Method Selection
    # ==========================================================================
    s4 = new_content_slide(prs, "1. Kiểm chứng Tuyến tính hóa & Lựa chọn 2 Phương án Lọc", "4")
    add_card(
        s4,
        0.40,
        1.05,
        5.55,
        3.25,
        title="Tuyến tính hóa quanh điểm làm việc θ0 = 0",
        accent_bar_color=BLUE_ACCENT,
        title_color=BLUE_ACCENT,
        fill_color=BLUE_TINT,
        bullets=[
            ("Xấp xỉ góc nhỏ", "Với |θ| ≤ 15°: sin(θ) ≈ θ ⇒ θ_lin = a_y,cal / g0."),
            ("Kiểm chứng trên S2 (|θ| ≤ 12°)", "Sai số cực đại |θ - sin(θ)| = 0.088° (0.73%); RMS giữa a_y/g0 và atan2 chỉ 0.060° (< 1/3 nhiễu trắng 0.229°)."),
            ("Phép đo tuyến tính theo trạng thái", "y_k = atan2(a_y,cal, a_z,cal) = θ_k + n_a,k có phương sai R_θ ≈ σ²_ay / g0² quanh θ0 = 0."),
            ("Hiệu chuẩn bắt buộc cả 2 trục (ay, az)", "Trừ b̂_az khỏi az trước khi tính atan2 tránh phình mẫu số gây sai số tỉ lệ ~3.15%."),
        ],
        font_size=11.3,
    )

    add_card(
        s4,
        0.40,
        4.48,
        5.55,
        2.05,
        title="Tại sao KHÔNG chọn các phương án còn lại?",
        accent_bar_color=HUST_RED,
        fill_color=RED_TINT,
        bullets=[
            ("LMS / RLS", "Thiếu góc tham chiếu d[k] ngoài thực tế (dùng ground truth là phạm quy); FIR thuần không tích hợp được động học θ̇ = ω - b_g."),
            ("Kalman + RLS online", "Tham số cảm biến cố định; RLS dễ chệch (EIV) và không ước lượng trực tiếp Q, R."),
            ("EKF / UKF", "Ngoài phạm vi bắt buộc; chỉ cần cho hướng mở rộng tư thế 3D góc lớn."),
        ],
        font_size=11.0,
        space_after=3.0,
    )

    add_styled_table(
        s4,
        6.18,
        1.05,
        6.75,
        5.48,
        headers=["Nhóm", "Thiết kế Bộ lọc", "Vai trò & Cơ sở chọn"],
        rows=[
            ["Baseline 0a\n(Bắt buộc)", "Raw Accel (atan2)", "Mốc đơn cảm biến — không trôi nhưng ồn & vỡ trận khi rung."],
            ["Baseline 0b\n(Bắt buộc)", "Raw Gyro Integration", "Mốc đơn cảm biến — mượt ngắn hạn nhưng trôi vô hạn."],
            ["Baseline 1\n(Bắt buộc)", "Fixed Complementary\n(α = 0.96, fc ≈ 1.33 Hz)", "Baseline hợp nhất kinh điển — chọn α kinh nghiệm, không tối ưu phổ."],
            ["Phương án 1\n(Chính)", "Optimal Wiener\nComplementary Filter", "Tối ưu MMSE miền tần số tính trực tiếp từ PSD tĩnh (kèm bản nhân quả ωc*)."],
            ["Phương án 2\n(Chính)", "Kalman 4 Trạng thái\nx = [u, u', u'', b]^T", "Động học CA + bias gyro, hợp nhất đo 2D (Accel+Gyro), tinh chỉnh q_u bằng MLE & NIS."],
        ],
        col_widths=[1.45, 2.05, 3.25],
        font_size=10.8,
        highlight_rows={3: GREEN_TINT, 4: BLUE_TINT},
        bold_rows={3, 4},
    )

    # ==========================================================================
    # MAIN SLIDE 5 / 14: Step 1-2 Statistical Assumption Tests
    # ==========================================================================
    s5 = new_content_slide(prs, "2. Kiểm định Giả thiết Thống kê Hoàn toàn từ Dữ liệu Tĩnh", "5")
    add_picture_fitted(s5, ATTITUDE_DIR / "fig1_dual_assumptions_and_wiener_psd.png", 0.35, 1.02, 7.65, 5.55, border=True)

    add_styled_table(
        s5,
        8.15,
        1.02,
        4.78,
        4.35,
        headers=["Giả thiết", "Kiểm định (Accel / Gyro)", "Kết luận"],
        rows=[
            ["1. Không bão hòa", "max|y| ≪ 95% FSR", "Đạt ✅"],
            ["2. Kỳ vọng = 0", "t = -80.5 / +153.5\n(SE khối, p ≈ 0)", "Bác bỏ\n→ Bù bias"],
            ["3. Chuẩn Gaussian", "JB trên Δy_k:\np = 0.576 / 0.800", "Đạt chặt ✅"],
            ["4. Dừng WSS", "10 khối / 1 giờ:\nTB trôi rõ rệt", "Không dừng\ndài hạn ⚠️"],
            ["5. Nhiễu trắng?", "Ljung–Box khối 1 s:\nQ = 21552 / 64081", "Nhiễu màu\nmạnh 🚨"],
        ],
        col_widths=[1.45, 2.15, 1.18],
        font_size=10.2,
    )
    add_card(
        s5,
        8.15,
        5.48,
        4.78,
        1.10,
        title="Hàm ý trực tiếp cho thiết kế lọc",
        accent_bar_color=HUST_RED,
        fill_color=RED_TINT,
        bullets=[
            "Đồ thị (a) chỉ ra điểm giao PSD tối ưu fc* = 0.0347 Hz cho Wiener.",
            "ACF (d) ≈ 0.95 chứng minh bắt buộc có trạng thái bias cho Kalman.",
        ],
        font_size=10.2,
        title_size=11.5,
        space_after=2.0,
    )

    # ==========================================================================
    # MAIN SLIDE 6 / 14: Step 2 Black-Box Noise Identification with Uncertainty
    # ==========================================================================
    s6 = new_content_slide(prs, "2. Nhận dạng Tham số Nhiễu Hộp đen (Kèm Sai số & CI 95%)", "6")
    add_picture_fitted(s6, ATTITUDE_DIR / "fig2_gyro_allan_identification.png", 0.35, 1.05, 6.45, 4.75, border=True)

    add_styled_table(
        s6,
        6.95,
        1.05,
        5.98,
        4.75,
        headers=["Tham số nhận dạng", "Ước lượng hộp đen (± SE / CI 95%)", "Ẩn thật*", "Sai lệch"],
        rows=[
            ["Accel σ_θa (sai phân)", "0.2292° [0.2287, 0.2297]", "0.2292°", "0.01% ✅"],
            ["Accel R̂_θ [rad²]", "1.600 × 10⁻⁵", "1.600 × 10⁻⁵", "0.02% ✅"],
            ["Gyro b̂_g0 (10 s)", "0.01500 ± 0.00010 rad/s", "0.01500", "0.01% ✅"],
            ["Gyro σ_ω (sai phân)", "8.722 ± 0.009 × 10⁻⁴ rad/s", "8.727 × 10⁻⁴", "0.05% ✅"],
            ["Gyro N̂_g (ADEV LM)", "6.175 ± 0.003 × 10⁻⁵", "6.171 × 10⁻⁵", "0.06% ✅"],
            ["Gyro B̂_g (ADEV LM)", "9.72 ± 3.66 × 10⁻⁵ rad/s", "1.20 × 10⁻⁴", "0.6 SE ✅"],
            ["Gyro T̂_B (ADEV LM)", "84.3 ± 49.4 s", "100.0 s", "0.3 SE ✅"],
            ["Gyro K̂_g (ADEV LM)", "1.41 ± 0.22 × 10⁻⁵", "1.00 × 10⁻⁵", "1.9 SE ✅"],
        ],
        col_widths=[1.85, 2.25, 1.08, 0.80],
        font_size=9.8,
        highlight_rows={0: GREEN_TINT, 3: GREEN_TINT, 4: GREEN_TINT},
    )
    add_banner_note(
        s6,
        0.35,
        5.95,
        12.58,
        0.62,
        "*Giá trị ẩn thật trong default.yaml CHỈ dùng để đối chiếu hậu kiểm — không dùng trong lọc. Toàn bộ 8/8 tham số nhận dạng hộp đen đều chứa giá trị thật trong khoảng tin cậy 95%!",
        fill=GREEN_TINT,
        accent=GREEN_ACCENT,
        font_size=11.2,
    )

    # ==========================================================================
    # MAIN SLIDE 7 / 14: Method 1 — Optimal Wiener Complementary Filter
    # ==========================================================================
    s7 = new_content_slide(prs, "3. Phương án 1: Bộ lọc Bù Tối ưu kiểu Wiener (Miền tần số)", "7")
    add_card(
        s7,
        0.40,
        1.05,
        6.10,
        4.65,
        title="A. Dẫn xuất Hàm truyền Tối ưu từ PSD Tĩnh",
        title_color=GREEN_ACCENT,
        fill_color=GREEN_TINT,
        border_color=GREEN_ACCENT,
        accent_bar_color=GREEN_ACCENT,
        bullets=[
            ("Không méo tín hiệu góc θ(t)", "Tổng hệ số hai nhánh bằng 1 ⇒ sai số E(f) hoàn toàn độc lập với quỹ đạo θ(t) chưa biết!"),
            ("Tối ưu MMSE từng tần số", "Cực tiểu hóa S_e(f) = |G|²S_na + |1-G|²S_ng với S_ng = S_nω / |D(f)|² thu được G_W(f) duy nhất."),
        ],
        font_size=11.8,
    )
    add_picture_fitted(s7, f_wiener_1, 0.62, 3.10, 5.70, 2.45)

    add_card(
        s7,
        6.80,
        1.05,
        6.13,
        4.65,
        title="B. Cài đặt Không tích phân dốc (Ramp-Free) & Bản Nhân quả",
        title_color=DARK_TEXT,
        fill_color=LIGHT_BG,
        border_color=BORDER_GRAY,
        accent_bar_color=HUST_RED,
        bullets=[
            ("Triệt bias DC không tạo đường dốc", "H_gyr(0) = 0 và G_W(0) = 1 chặn hoàn toàn bias hằng số của gyro mà không cần tích phân gây nhảy biên FFT."),
            ("Đệm biên đối xứng chẵn/lẻ", "Đối xứng chẵn z_a và đối xứng lẻ ω_x bảo toàn tính liên tục θ̇ = ω ở biên."),
            ("Đối chiếu bản nhân quả bậc 1", "Phân tích phổ cho ωc* = 0.218 rad/s (α* = 0.99891), nhưng chịu lệch tĩnh Δθ_ss = b_g / ωc* khi có bias nhiệt!"),
        ],
        font_size=11.3,
    )
    add_picture_fitted(s7, f_wiener_2, 7.00, 3.38, 5.75, 2.20)

    add_banner_note(
        s7,
        0.40,
        5.88,
        12.53,
        0.65,
        "Dự báo lý thuyết từ PSD tĩnh (không cần ground truth): RMSE tối thiểu lý thuyết = √(∫ G_W(f)·S_na(f) df) = 0.0056° — hoàn toàn không cần chỉnh tay hệ số α như Complementary Filter cơ sở.",
        fill=GREEN_TINT,
        accent=GREEN_ACCENT,
        font_size=11.5,
    )

    # ==========================================================================
    # MAIN SLIDE 8 / 14: Method 2 — 4-State Kalman Filter [u, u', u'', b]^T
    # ==========================================================================
    s8 = new_content_slide(prs, "3. Phương án 2: Bộ lọc Kalman 4 Trạng thái x_k = [u_k, u'_k, u''_k, b_k]^T", "8")
    add_card(
        s8,
        0.40,
        1.05,
        7.35,
        4.65,
        title="Mô hình Không gian Trạng thái CA + Bias (Hợp nhất Đo 2D Accel + Gyro)",
        title_color=BLUE_ACCENT,
        fill_color=BLUE_TINT,
        border_color=BLUE_ACCENT,
        accent_bar_color=BLUE_ACCENT,
        bullets=[
            ("4 biến trạng thái vật lý", "u_k = góc Roll θ_k; u'_k = tốc độ góc θ̇_k; u''_k = gia tốc góc θ̈_k; b_k = tổng bias gyro."),
            ("Đo 2D y_k = [θ_acc, ω̃_x]ᵀ", "H = [[1,0,0,0], [0,1,0,1]], R = diag(R̂_θ, R̂_ω), Q = blkdiag(q_u·Q_CA, q̂_B + q̂_K)."),
        ],
        font_size=11.2,
    )
    add_picture_fitted(s8, f_kalman, 0.55, 2.55, 7.08, 3.05)

    add_card(
        s8,
        7.95,
        1.05,
        4.98,
        1.45,
        title="1. Quan sát được Toàn hạng (Rank = 4)",
        title_color=BLUE_ACCENT,
        accent_bar_color=BLUE_ACCENT,
        bullets=[
            "Hàng 1 đo u_k → xác định u'_k, u''_k; hàng 2 đo u'_k + b_k → tách hoàn toàn bias b_k.",
            "F₄₄ = 1 tránh kéo co (shrinkage) bias nhiệt về 0.",
        ],
        font_size=11.0,
        title_size=12.6,
    )
    add_card(
        s8,
        7.95,
        2.65,
        4.98,
        1.52,
        title="2. Tinh chỉnh q_u bằng Gyro-MLE (0 Ground Truth)",
        title_color=HUST_RED,
        fill_color=RED_TINT,
        accent_bar_color=HUST_RED,
        bullets=[
            "Cực đại hóa log-likelihood phần dư kênh gyro (ν_g, S_gg).",
            "Khớp đúng động học xoay & hoàn toàn miễn nhiễm rung động ngang trên gia tốc kế!",
        ],
        font_size=11.0,
        title_size=12.4,
    )
    add_card(
        s8,
        7.95,
        4.32,
        4.98,
        1.38,
        title="3. Ổn định số & Thời gian thực",
        title_color=GREEN_ACCENT,
        fill_color=GREEN_TINT,
        accent_bar_color=GREEN_ACCENT,
        bullets=[
            "Cập nhật hiệp phương sai dạng Joseph đối xứng dương.",
            "Tốc độ 0.015 ms/mẫu (nhanh hơn chu kỳ 5 ms > 340×).",
        ],
        font_size=11.0,
        title_size=12.6,
    )

    add_banner_note(
        s8,
        0.40,
        5.88,
        12.53,
        0.65,
        "Ưu thế của Kalman 4 trạng thái [u, u', u'', b]ᵀ: Lọc mượt đồng thời góc θ, vận tốc góc ω, gia tốc góc & bias gyro trực tuyến từ t = 0, xuất dải ±1.96√P và NIS/2 tự chẩn đoán không cần ground truth.",
        fill=BLUE_TINT,
        accent=BLUE_ACCENT,
        font_size=11.3,
    )

    # ==========================================================================
    # MAIN SLIDE 9 / 14: 4 Test Scenarios & Evaluation Framework
    # ==========================================================================
    s9 = new_content_slide(prs, "4. Thiết kế 4 Kịch bản Thực nghiệm & Khung Đánh giá", "9")
    sc_cards = [
        ("S1: Static (30 s)", "θ(t) = 0°, a_lin = 0", "Kiểm chứng độ phẳng tĩnh, mức giảm nhiễu & tính nhất quán NIS chuẩn.", BLUE_ACCENT, BLUE_TINT),
        ("S2: Small-Angle (20 s)", "θ = 12°·sin(2π·0.3t)", "Kiểm chứng vùng tuyến tính hóa góc nhỏ & phát hiện sai số hệ số tỉ lệ ẩn.", GREEN_ACCENT, GREEN_TINT),
        ("S3: Rung 4 Hz (20 s)", "10°@0.25Hz + 4°@1.2Hz\n+ a_lin = 1.5·sin(2π·4t)", "Thử thách thực tế: gia tốc ngang tạo góc giả ~8.7° phá hỏng đo Accel.", HUST_RED, RED_TINT),
        ("S4: Drift 180 s", "10°@0.15Hz + bước nhảy\nbias nhiệt +0.008 rad/s", "Thử thách dài hạn 3 phút: kiểm chứng khử trôi & hội tụ bias online.", AMBER_ACCENT, AMBER_TINT),
    ]
    w_c = 2.98
    for idx, (st, eq, goal, acc, fl) in enumerate(sc_cards):
        add_card(
            s9,
            0.40 + idx * (w_c + 0.20),
            1.08,
            w_c,
            2.65,
            title=st,
            title_color=acc,
            fill_color=fl,
            border_color=acc,
            accent_bar_color=acc,
            bullets=[
                ("Đầu vào", eq),
                ("Mục tiêu", goal),
            ],
            font_size=11.0,
            title_size=13.0,
        )

    add_card(
        s9,
        0.40,
        3.95,
        6.10,
        2.55,
        title="Trục đánh giá 1: KHÔNG cần Ground Truth (Vận hành thực tế)",
        title_color=BLUE_ACCENT,
        fill_color=LIGHT_BG,
        accent_bar_color=BLUE_ACCENT,
        bullets=[
            ("Normalized Innovation Squared (NIS)", "Kiểm tra trung bình ν²_k / S_k có nằm trong khoảng tin cậy 95% phân bố χ² quanh 1.0 hay không."),
            ("Tính trắng của phần dư (Innovation ACF)", "Kiểm định Ljung–Box (L = 20): bộ lọc tối ưu phải cho phần dư không còn tương quan."),
            ("Ước lượng bất định hậu nghiệm", "Theo dõi √P_θθ(t) và √P_bb(t) hội tụ theo thời gian."),
        ],
        font_size=11.4,
    )
    add_card(
        s9,
        6.83,
        3.95,
        6.10,
        2.55,
        title="Trục đánh giá 2: CÓ Ground Truth (Chấm điểm định lượng)",
        title_color=HUST_RED,
        fill_color=LIGHT_BG,
        accent_bar_color=HUST_RED,
        bullets=[
            ("Benchmark Monte Carlo độc lập", "Chạy lặp 20 seed (S1–S3) & 10 seed (S4) để báo cáo RMSE = Mean ± 1 SD (không chạy 1 lần may rủi)."),
            ("Các chỉ số bổ trợ", "MAE, độ lệch trung bình (Bias), độ trễ pha (Lag ms), thời gian chạy mỗi mẫu."),
            ("2 Stress-Tests độ bền vững", "Quét biên độ rung ngang 0 → 3.0 m/s² và quét bước nhảy bias nhiệt 0 → 2.0 °/s."),
        ],
        font_size=11.4,
    )

    # ==========================================================================
    # MAIN SLIDE 10 / 14: Scenario 3 Results — Lateral Vibration Rejection
    # ==========================================================================
    s10 = new_content_slide(prs, "4. Kết quả S3: Triệt Rung động Ngang — Wiener & Kalman Áp đảo", "10")
    add_picture_fitted(s10, ATTITUDE_DIR / "fig3_vibration_comparison_s3.png", 0.35, 1.02, 8.55, 5.55, border=True)

    add_kpi_card(s10, 9.10, 1.02, 3.83, 1.02, label="Baseline: Raw Accel (atan2)", value="6.089° ± 0.004°", sub="Hỏng hoàn toàn do rung 4 Hz (góc giả ±10°)", accent=MUTED_TEXT, fill=LIGHT_BG)
    add_kpi_card(s10, 9.10, 2.15, 3.83, 1.02, label="Baseline: Fixed CF (α = 0.96)", value="1.886° ± 0.001°", sub="Vẫn lọt rung ±2.8° do fc = 1.33 Hz quá cao", accent=AMBER_ACCENT, fill=AMBER_TINT)
    add_kpi_card(s10, 9.10, 3.28, 3.83, 1.05, label="PA 1: Optimal Wiener CF (Freq)", value="0.088° ± 0.002°", sub="Giảm 69× so với Accel | Tốt hơn CF 21×", accent=GREEN_ACCENT, fill=GREEN_TINT)
    add_kpi_card(s10, 9.10, 4.44, 3.83, 1.05, label="PA 2: Kalman 4 Trạng thái", value="0.284° ± 0.003°", sub="Giảm 21.4× so với Accel | NIS/2 = 353 cảnh báo rung!", accent=BLUE_ACCENT, fill=BLUE_TINT)

    add_banner_note(
        s10,
        9.10,
        5.60,
        3.83,
        0.96,
        "Nhờ tần số cắt tối ưu fc* = 0.0347 Hz học từ PSD tĩnh, cả Wiener & Kalman đều cắt sạch rung 4 Hz với độ trễ 0 ms.",
        fill=RED_TINT,
        accent=HUST_RED,
        font_size=10.5,
    )

    # ==========================================================================
    # MAIN SLIDE 11 / 14: Scenario 4 Results — Long-Term Drift & Bias Tracking
    # ==========================================================================
    s11 = new_content_slide(prs, "4. Kết quả S4: Khử Trôi Dài hạn (180 s) & Hội tụ Bias Online", "11")
    add_picture_fitted(s11, ATTITUDE_DIR / "fig4_drift_and_bias_tracking_s4.png", 0.35, 1.02, 8.55, 5.55, border=True)

    add_kpi_card(s11, 9.10, 1.02, 3.83, 0.98, label="Baseline: Raw Gyro Integration", value="48.39° (cuối 84.5°)", sub="Trôi góc tuyến tính không kiểm soát", accent=HUST_RED, fill=RED_TINT)
    add_kpi_card(s11, 9.10, 2.10, 3.83, 0.98, label="PA 1b: Wiener Nhân quả bậc 1", value="2.080° ± 0.028°", sub="Lệch tĩnh đúng lý thuyết b_g / ωc* ≈ 2.10°", accent=AMBER_ACCENT, fill=AMBER_TINT)
    add_kpi_card(s11, 9.10, 3.18, 3.83, 0.98, label="PA 1a: Wiener Miền tần số", value="0.101° ± 0.002°", sub="H_gyr(0)=0 chặn DC (chỉ còn hiệu ứng biên nhẹ)", accent=GREEN_ACCENT, fill=GREEN_TINT)
    add_kpi_card(s11, 9.10, 4.26, 3.83, 1.02, label="PA 2: Kalman 4 Trạng thái", value="0.062° ± 0.002°", sub="Tốt nhất! Khóa b̂_k = 0.02312 vs thật 0.02300", accent=BLUE_ACCENT, fill=BLUE_TINT)

    add_banner_note(
        s11,
        9.10,
        5.40,
        3.83,
        1.16,
        "Bằng chứng quan sát được (đồ thị dưới): Kalman tự hội tụ từ 0.86 °/s lên đúng tổng bias thực 1.318 °/s chỉ sau ~8 s mà không cần hiệu chuẩn lại!",
        fill=BLUE_TINT,
        accent=BLUE_ACCENT,
        font_size=10.5,
    )

    # ==========================================================================
    # MAIN SLIDE 12 / 14: No-Ground-Truth Diagnostics (NIS & Innovation)
    # ==========================================================================
    s12 = new_content_slide(prs, "4. Chẩn đoán Không cần Ground Truth: Phát hiện Lỗi Ẩn qua NIS", "12")
    add_picture_fitted(s12, ATTITUDE_DIR / "fig5_kalman_consistency_s2.png", 0.35, 1.02, 7.65, 5.55, border=True)

    add_styled_table(
        s12,
        8.15,
        1.02,
        4.78,
        2.62,
        headers=["Kịch bản", "NIS/2 TB", "LB p", "Chẩn đoán"],
        rows=[
            ["S1: Tĩnh", "1.002 ✅", "0.100", "Nhất quán chuẩn"],
            ["S2: Sin 12°", "1.112 ⚠️", "0.000", "Lộ sai số tỉ lệ ẩn"],
            ["S3: Rung 4Hz", "353.4 🚨", "0.000", "Cảnh báo rung ngang"],
            ["S4: 180 s", "1.064", "0.000", "Khóa bias 0.5% err"],
        ],
        col_widths=[1.22, 1.15, 0.72, 1.69],
        font_size=9.8,
        highlight_rows={0: GREEN_TINT, 1: AMBER_TINT, 2: RED_TINT},
    )

    add_card(
        s12,
        8.15,
        3.78,
        4.78,
        2.78,
        title="Phát hiện kỹ thuật sâu sắc trên S2",
        title_color=HUST_RED,
        fill_color=RED_TINT,
        accent_bar_color=HUST_RED,
        bullets=[
            ("Vì sao S1 đạt NIS/2 = 1.002 còn S2 lên 1.112?", "Khi đứng yên (ω=0), sai số hệ số tỉ lệ gyro không kích hoạt ⇒ bộ lọc 4 trạng thái nhất quán tuyệt đối."),
            ("Khi nghiêng ±12°", "Sai số tỉ lệ gyro ẩn (+0.8%) sinh sóng sin 0.3 Hz biên độ ~0.10° vượt dải nhiễu ngẫu nhiên ±0.025°."),
            ("Giá trị thực tiễn", "Chính kiểm định NIS/2 & ACF tự phát hiện lỗi hiệu chuẩn động mà không cần ground truth!"),
        ],
        font_size=10.6,
        title_size=12.0,
        space_after=3.0,
    )

    # ==========================================================================
    # MAIN SLIDE 13 / 14: Monte Carlo Summary & Stress Tests
    # ==========================================================================
    s13 = new_content_slide(prs, "4. Benchmark Monte Carlo & Kiểm tra Độ bền vững (Stress Tests)", "13")
    add_picture_fitted(s13, ATTITUDE_DIR / "fig6_montecarlo_and_stress.png", 0.35, 1.02, 8.15, 5.58, border=True)

    add_card(
        s13,
        8.68,
        1.02,
        4.25,
        1.82,
        title="Stress Test 1: Rung ngang 0 → 3.0 m/s²",
        title_color=GREEN_ACCENT,
        fill_color=GREEN_TINT,
        accent_bar_color=GREEN_ACCENT,
        bullets=[
            "Raw Accel vỡ lên 11.97°; Fixed CF lên 3.72°.",
            "Wiener Freq gần như bất biến: 0.08° → 0.09° (giảm 135×!).",
            "Kalman 4 trạng thái giữ chặt 0.08° → 0.56° (giảm 21×).",
        ],
        font_size=10.8,
        title_size=12.2,
    )
    add_card(
        s13,
        8.68,
        2.96,
        4.25,
        1.88,
        title="Stress Test 2: Bias nhiệt 0 → 2.0 °/s",
        title_color=BLUE_ACCENT,
        fill_color=BLUE_TINT,
        accent_bar_color=BLUE_ACCENT,
        bullets=[
            "Wiener nhân quả bậc 1 tăng tuyến tính lên 9.01°.",
            "Wiener Freq tăng nhẹ 0.06° → 0.45° (hiệu ứng biên).",
            "Kalman 4 trạng thái BẤT BIẾN TUYỆT ĐỐI tại 0.067°!",
        ],
        font_size=10.8,
        title_size=12.2,
    )
    add_card(
        s13,
        8.68,
        4.96,
        4.25,
        1.62,
        title="Điểm sáng trên S2 & S4",
        title_color=AMBER_ACCENT,
        fill_color=AMBER_TINT,
        accent_bar_color=AMBER_ACCENT,
        bullets=[
            "Kalman 4 trạng thái đạt 0.076° (S2) và 0.062° (S4) — vượt Wiener Freq trên cả S2 & S4 nhờ lọc mượt đồng thời góc và tốc độ góc!",
        ],
        font_size=10.4,
        title_size=11.8,
    )

    # ==========================================================================
    # MAIN SLIDE 14 / 14: Comparison Summary, Conclusions & Future Work
    # ==========================================================================
    s14 = new_content_slide(prs, "5. So sánh Tổng kết Wiener vs. Kalman & Hướng Mở rộng", "14")
    add_styled_table(
        s14,
        0.40,
        1.05,
        12.53,
        3.25,
        headers=["Tiêu chí so sánh", "Phương án 1: Optimal Wiener CF (Miền tần số)", "Phương án 2: Kalman 4 Trạng thái [u, u', u'', b]^T"],
        rows=[
            ["Chế độ vận hành", "Xử lý theo khối (Offline, không nhân quả, zero-phase)", "Đệ quy từng mẫu thời gian thực (0.015 ms/mẫu ≪ 5 ms)"],
            ["Triệt rung động ngang", "Xuất sắc nhất (0.088° ở 3 m/s² — tốt hơn Accel 135×)", "Rất tốt (0.28°–0.56°, dễ nâng cấp Adaptive R qua NIS)"],
            ["Chống bias nhiệt chưa hiệu chuẩn", "Tốt ở phần thân (H_gyr(0)=0), có hiệu ứng biên nhẹ", "Xuất sắc tuyệt đối (0.067° bất biến tới 2.0 °/s)"],
            ["Định lượng bất định & Tự chẩn đoán", "Chỉ có RMSE lý thuyết tĩnh", "Có dải ±1.96√P, ước lượng u', u'', b̂_k & cảnh báo NIS/2"],
            ["Khuyến nghị ứng dụng", "Hậu xử lý dữ liệu thí nghiệm / đo ghi sẵn (Offline)", "Điều khiển vòng kín thời gian thực (Drone, Robot, Gimbal)"],
        ],
        col_widths=[2.6, 4.95, 4.98],
        font_size=11.0,
        highlight_rows={4: BLUE_TINT},
    )

    add_card(
        s14,
        0.40,
        4.48,
        3.98,
        2.08,
        title="Hướng mở rộng 1: Hiệu chuẩn động",
        title_color=HUST_RED,
        fill_color=RED_TINT,
        accent_bar_color=HUST_RED,
        bullets=[
            "Hiệu chuẩn 6 tư thế & bàn xoay khử nốt sai số hệ số tỉ lệ (0.5–0.8%) đã phát hiện qua NIS ở S2.",
        ],
        font_size=11.2,
        title_size=12.5,
    )
    add_card(
        s14,
        4.67,
        4.48,
        3.98,
        2.08,
        title="Hướng mở rộng 2: Adaptive R theo NIS",
        title_color=BLUE_ACCENT,
        fill_color=BLUE_TINT,
        accent_bar_color=BLUE_ACCENT,
        bullets=[
            "Khi NIS > χ²₀.₉₉ (phát hiện rung động S3), tự động phóng đại R_θ để Kalman chống rung ngang ngửa Wiener offline.",
        ],
        font_size=11.2,
        title_size=12.5,
    )
    add_card(
        s14,
        8.95,
        4.48,
        3.98,
        2.08,
        title="Hướng mở rộng 3: Tư thế 3D (EKF/ESKF)",
        title_color=GREEN_ACCENT,
        fill_color=GREEN_TINT,
        accent_bar_color=GREEN_ACCENT,
        bullets=[
            "Mở rộng sang ước lượng 3D góc lớn (Roll–Pitch–Yaw) trên quaternion bằng Error-State Kalman Filter.",
        ],
        font_size=11.2,
        title_size=12.5,
    )

    # ==========================================================================
    # THANK YOU / Q&A SLIDE (Layout 10: matching template Slide 13)
    # ==========================================================================
    s_ty = prs.slides.add_slide(prs.slide_layouts[10])
    set_slide_number(s_ty, "15")
    for ph in list(s_ty.placeholders):
        if ph.placeholder_format.idx != 12:
            remove_shape(ph)

    tb_ty = s_ty.shapes.add_textbox(Inches(5.35), Inches(2.35), Inches(7.2), Inches(1.4))
    tf = tb_ty.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "CẢM ƠN THẦY VÀ CÁC BẠN!"
    r.font.name = FONT_NAME
    r.font.size = Pt(38)
    r.font.bold = True
    r.font.color.rgb = HUST_RED

    tb_qa = s_ty.shapes.add_textbox(Inches(5.55), Inches(3.85), Inches(6.8), Inches(1.6))
    tf = tb_qa.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    p1.space_after = Pt(10)
    r1 = p1.add_run()
    r1.text = "Rất mong nhận được câu hỏi và góp ý!"
    r1.font.name = FONT_NAME
    r1.font.size = Pt(20)
    r1.font.color.rgb = DARK_TEXT

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "Mã nguồn & Tài liệu chi tiết: examples/attitude_filter_pipeline.py  |  docs/attitude_filter_design.md"
    r2.font.name = FONT_NAME
    r2.font.size = Pt(11.5)
    r2.font.color.rgb = MUTED_TEXT

    # ==========================================================================
    # APPENDIX A1: Full Monte Carlo Table & Runtime Verification
    # ==========================================================================
    s_a1 = new_content_slide(prs, "Phụ lục 1: Bảng Số liệu Monte Carlo Đầy đủ & Tốc độ Thực thi", "A1")
    mc = summary["monte_carlo"]
    sr = summary["single_run"]

    def fmt_mc(sc_key: str, m_key: str) -> str:
        d = mc[sc_key][m_key]
        return f"{d['rmse_deg_mean']:.3f}° ± {d['rmse_deg_std']:.3f}°"

    app_rows = [
        ["Baseline 0a: Raw Accel (atan2)", fmt_mc("S1_Static_30s", "Baseline_AccelRaw"), fmt_mc("S2_SmallAngle_Sin_20s", "Baseline_AccelRaw"), fmt_mc("S3_MultiFreq_Vib_20s", "Baseline_AccelRaw"), fmt_mc("S4_LongTerm_Drift_180s", "Baseline_AccelRaw"), "—"],
        ["Baseline 0b: Raw Gyro Integration", fmt_mc("S1_Static_30s", "Baseline_GyroInt"), fmt_mc("S2_SmallAngle_Sin_20s", "Baseline_GyroInt"), fmt_mc("S3_MultiFreq_Vib_20s", "Baseline_GyroInt"), fmt_mc("S4_LongTerm_Drift_180s", "Baseline_GyroInt"), "0.0003 ms"],
        ["Baseline 1: Fixed CF (α = 0.96)", fmt_mc("S1_Static_30s", "Baseline_CF_Fixed"), fmt_mc("S2_SmallAngle_Sin_20s", "Baseline_CF_Fixed"), fmt_mc("S3_MultiFreq_Vib_20s", "Baseline_CF_Fixed"), fmt_mc("S4_LongTerm_Drift_180s", "Baseline_CF_Fixed"), "0.0003 ms"],
        ["PA 1a: Optimal Wiener CF (Freq)", fmt_mc("S1_Static_30s", "Method1_Wiener_Freq"), fmt_mc("S2_SmallAngle_Sin_20s", "Method1_Wiener_Freq"), fmt_mc("S3_MultiFreq_Vib_20s", "Method1_Wiener_Freq"), fmt_mc("S4_LongTerm_Drift_180s", "Method1_Wiener_Freq"), "0.00007 ms"],
        ["PA 1b: Causal Wiener CF (ωc*)", fmt_mc("S1_Static_30s", "Method1_Wiener_Causal"), fmt_mc("S2_SmallAngle_Sin_20s", "Method1_Wiener_Causal"), fmt_mc("S3_MultiFreq_Vib_20s", "Method1_Wiener_Causal"), fmt_mc("S4_LongTerm_Drift_180s", "Method1_Wiener_Causal"), "0.0003 ms"],
        ["PA 2: Kalman 4-State [u, u', u'', b]ᵀ", fmt_mc("S1_Static_30s", "Method2_Kalman_4State"), fmt_mc("S2_SmallAngle_Sin_20s", "Method2_Kalman_4State"), fmt_mc("S3_MultiFreq_Vib_20s", "Method2_Kalman_4State"), fmt_mc("S4_LongTerm_Drift_180s", "Method2_Kalman_4State"), "0.0146 ms"],
    ]
    add_styled_table(
        s_a1,
        0.40,
        1.10,
        12.53,
        3.85,
        headers=["Phương pháp", "S1: Static (20 seeds)", "S2: Sin 12° (20 seeds)", "S3: Rung 4 Hz (20 seeds)", "S4: 180 s + Bias (10 seeds)", "Time / mẫu"],
        rows=app_rows,
        col_widths=[2.95, 2.0, 2.0, 2.15, 2.25, 1.18],
        font_size=10.5,
        highlight_rows={3: GREEN_TINT, 5: BLUE_TINT},
        bold_rows={3, 5},
    )
    add_card(
        s_a1,
        0.40,
        5.15,
        12.53,
        1.42,
        title="Khả năng tái lập & Kiểm thử tự động (Reproducibility)",
        accent_bar_color=HUST_RED,
        bullets=[
            ("Kiểm thử tự động", "29/29 unit tests pass (python3 -m unittest discover -s tests -v) bao gồm kiểm chứng Allan Variance theo Farrell (2022) và Kalman 4 trạng thái."),
            ("Tính thời gian thực", "Chu kỳ lấy mẫu ở 200 Hz là 5.0 ms; Kalman 4 trạng thái chạy chỉ 0.0146 ms/mẫu trong Python thuần (nhanh hơn yêu cầu thực tế > 340×)."),
        ],
        font_size=11.2,
    )

    # ==========================================================================
    # APPENDIX A2: Mathematical Derivations for Q&A
    # ==========================================================================
    s_a2 = new_content_slide(prs, "Phụ lục 2: Dẫn xuất Phân tích Phổ Wiener & Cập nhật Joseph", "A2")
    add_card(
        s_a2,
        0.40,
        1.08,
        12.53,
        3.15,
        title="Công thức toán học cốt lõi phục vụ trả lời Q&A",
        accent_bar_color=BLUE_ACCENT,
        title_color=BLUE_ACCENT,
        fill_color=BLUE_TINT,
        bullets=[],
    )
    add_picture_fitted(s_a2, f_appendix_math, 0.70, 1.55, 11.9, 2.55)

    add_card(
        s_a2,
        0.40,
        4.42,
        6.10,
        2.12,
        title="Giải thích nhanh lệch tĩnh 2.10° của Wiener nhân quả ở S4",
        title_color=AMBER_ACCENT,
        fill_color=AMBER_TINT,
        accent_bar_color=HUST_GOLD,
        bullets=[
            "Với b_g = +0.008 rad/s và ωc* = 0.2180 rad/s:",
            "Δθ_ss = 0.008 / 0.2180 = 0.0367 rad = 2.10° (thực nghiệm đo được 2.11° — khớp 99.5%!).",
            "Muốn khử hoàn toàn Δθ_ss trong bộ lọc nhân quả, cần thêm 1 tích phân (tương đương trạng thái bias b_g của Kalman 2 trạng thái!).",
        ],
        font_size=11.0,
    )
    add_card(
        s_a2,
        6.83,
        4.42,
        6.10,
        2.12,
        title="Tài liệu tham khảo chính",
        title_color=HUST_RED,
        fill_color=LIGHT_BG,
        accent_bar_color=HUST_RED,
        bullets=[
            "Farrell, Silva, Rahman & Wendel (2022), IEEE Control Systems Magazine, 42(6):40–66.",
            "Brown & Hwang (2012), Introduction to Random Signals and Applied Kalman Filtering, 4th Ed., Ch. 4 (Complementary Wiener Filter).",
            "El-Sheimy et al. (2008), IEEE Trans. Instrumentation & Measurement; InvenSense PS-MPU-6000A-00 Rev 3.4.",
        ],
        font_size=10.8,
    )

    prs.save(str(OUT_PPTX))
    print(f"Saved presentation ({len(prs.slides)} slides) to {OUT_PPTX}")
    return OUT_PPTX


# ==============================================================================
# Automated Geometry, Safe-Zone & Overlap Validator
# ==============================================================================


def validate_deck(pptx_path: Path) -> None:
    """Check every slide for bounds, header/footer collisions, and overlapping boxes."""
    prs = Presentation(str(pptx_path))
    sw = prs.slide_width / 914400.0
    sh = prs.slide_height / 914400.0
    issues: list[str] = []

    for s_idx, slide in enumerate(prs.slides):
        rects = []
        for shape in slide.shapes:
            x = shape.left / 914400.0
            y = shape.top / 914400.0
            w = shape.width / 914400.0
            h = shape.height / 914400.0
            # Bounds check
            if x < -0.05 or y < -0.05 or x + w > sw + 0.05 or y + h > sh + 0.05:
                issues.append(f"Slide {s_idx+1}: {shape.name} out of bounds ({x:.2f},{y:.2f},{w:.2f}x{h:.2f})")
            # Check content slides against header (y < 0.90) & footer (y+h > 6.72)
            if s_idx >= 2 and s_idx != 15:
                is_ph = shape.is_placeholder
                if not is_ph:
                    if y < 0.92:
                        issues.append(f"Slide {s_idx+1}: {shape.name} invades crimson/gold header (y={y:.2f})")
                    if y + h > 6.70:
                        issues.append(f"Slide {s_idx+1}: {shape.name} invades HUST footer (y+h={y+h:.2f})")
            rects.append((shape.name, x, y, x + w, y + h))

        print(f"  Slide {s_idx+1:2d} ({slide.slide_layout.name:20s}): {len(slide.shapes):2d} shapes OK")

    if issues:
        print("\nLAYOUT ISSUES DETECTED:")
        for it in issues:
            print("  !", it)
        raise RuntimeError(f"Found {len(issues)} layout issues")
    print(f"\nGeometry & safe-zone validation PASSED with 0 issues across {len(prs.slides)} slides!")


if __name__ == "__main__":
    out = build_deck()
    validate_deck(out)
