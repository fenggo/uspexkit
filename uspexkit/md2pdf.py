#!/usr/bin/env python3
"""Convert a Markdown file to a nicely formatted PDF using reportlab.

Usage:
    from uspexkit.md2pdf import md2pdf
    md2pdf(input='CLAUDE')          # reads CLAUDE.md, writes CLAUDE.pdf
    md2pdf(input='README.md')       # reads README.md, writes README.pdf
    md2pdf(input='path/to/doc')     # reads path/to/doc.md, writes path/to/doc.pdf

CLI:
    uspexkit md2pdf --i=CLAUDE
"""

import re
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Preformatted,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── CJK Font Registration ─────────────────────────────────────────
_CJK_FONT_FILE = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
_CJK_FONT_NAME = 'WQYMicroHei'
_CJK_MONO_NAME = 'WQYMicroHeiMono'

for _subfont, _name in [(0, _CJK_FONT_NAME), (2, _CJK_MONO_NAME)]:
    try:
        pdfmetrics.registerFont(TTFont(_name, _CJK_FONT_FILE, subfontIndex=_subfont))
    except Exception:
        try:
            pdfmetrics.registerFont(TTFont(_name, _CJK_FONT_FILE, subfontIndex=0))
        except Exception:
            _alt = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
            pdfmetrics.registerFont(TTFont(_name, _alt, subfontIndex=0))


# ── Helpers ────────────────────────────────────────────────────────
def resolve_paths(raw_input):
    """Resolve input .md and output .pdf paths from the --i argument."""
    base, ext = os.path.splitext(raw_input)
    if ext == '.md':
        md_path = raw_input
        pdf_path = base + '.pdf'
    elif ext == '.pdf':
        md_path = base + '.md'
        pdf_path = raw_input
    else:
        md_path = raw_input + '.md'
        pdf_path = raw_input + '.pdf'
    return md_path, pdf_path


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='CodeBlock', fontName=_CJK_MONO_NAME, fontSize=8, leading=10,
        leftIndent=12, spaceAfter=6, spaceBefore=6,
        backColor=HexColor('#f4f4f4'), borderPadding=6,
    ))
    styles.add(ParagraphStyle(
        name='InlineCode', fontName=_CJK_MONO_NAME, fontSize=8.5,
        backColor=HexColor('#f0f0f0'),
    ))
    styles.add(ParagraphStyle(
        name='H1Custom', fontName=_CJK_FONT_NAME, fontSize=20, leading=26,
        spaceAfter=10, spaceBefore=20, textColor=HexColor('#1a1a2e'),
    ))
    styles.add(ParagraphStyle(
        name='H2Custom', fontName=_CJK_FONT_NAME, fontSize=14, leading=18,
        spaceAfter=8, spaceBefore=16, textColor=HexColor('#16213e'),
    ))
    styles.add(ParagraphStyle(
        name='H3Custom', fontName=_CJK_FONT_NAME, fontSize=11.5, leading=15,
        spaceAfter=6, spaceBefore=12, textColor=HexColor('#0f3460'),
    ))
    styles.add(ParagraphStyle(
        name='BodyCustom', fontName=_CJK_FONT_NAME, fontSize=9.5, leading=14,
        spaceAfter=4, spaceBefore=2,
    ))
    styles.add(ParagraphStyle(
        name='BulletCustom', fontName=_CJK_FONT_NAME, fontSize=9.5, leading=14,
        spaceAfter=2, spaceBefore=1, leftIndent=20, bulletIndent=10,
    ))
    styles.add(ParagraphStyle(
        name='tbl_header', fontName=_CJK_FONT_NAME, fontSize=8.5, leading=11,
        textColor=white, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name='tbl_cell', fontName=_CJK_FONT_NAME, fontSize=8.5, leading=11,
    ))
    return styles


def escape_xml(s):
    """Escape &, <, > for reportlab XML."""
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def code_span(text):
    """Convert inline `code` to styled XML."""
    return re.sub(
        r'`([^`]+)`',
        lambda m: f'<font face="{_CJK_MONO_NAME}" size="8" backColor="#f0f0f0"> {escape_xml(m.group(1))} </font>',
        text
    )


def bold_span(text):
    """Convert **bold** to <b> tags."""
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)


def render_md(text, styles):
    """Parse markdown text into a list of reportlab flowables."""
    flowables = []
    lines = text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = '\n'.join(code_lines)
            if code_text.strip():
                flowables.append(Preformatted(
                    escape_xml(code_text), styles['CodeBlock']
                ))
            continue

        # Table
        if line.startswith('|') and i + 1 < len(lines) and re.match(r'\|[\s\-:|]+\|', lines[i+1]):
            header = [c.strip() for c in line.split('|')[1:-1]]
            i += 2  # skip separator
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].split('|')[1:-1]])
                i += 1

            data = []
            data.append([Paragraph(escape_xml(h), styles['tbl_header']) for h in header])
            for row in rows:
                data.append([Paragraph(escape_xml(c), styles['tbl_cell']) for c in row])

            if len(header) > 1:
                w = (A4[0] - 40 * mm - 12 * mm) / len(header)
                col_widths = [w] * len(header)
            else:
                col_widths = [A4[0] - 40 * mm - 12 * mm]

            tbl = Table(data, colWidths=col_widths)
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            flowables.append(Spacer(1, 4))
            flowables.append(tbl)
            flowables.append(Spacer(1, 6))
            continue

        # H1
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            flowables.append(Paragraph(escape_xml(text), styles['H1Custom']))
            flowables.append(HRFlowable(
                width="100%", thickness=2, color=HexColor('#16213e'),
                spaceAfter=8, spaceBefore=2
            ))
            i += 1
            continue

        # H2
        if line.startswith('## '):
            text = line[3:].strip()
            flowables.append(Paragraph(escape_xml(text), styles['H2Custom']))
            i += 1
            continue

        # H3
        if line.startswith('### '):
            text = line[4:].strip()
            flowables.append(Paragraph(escape_xml(text), styles['H3Custom']))
            i += 1
            continue

        # Horizontal rule
        if line.strip() in ('---', '***', '___'):
            flowables.append(HRFlowable(
                width="100%", thickness=0.5, color=HexColor('#cccccc'),
                spaceAfter=8, spaceBefore=8
            ))
            i += 1
            continue

        # Bullet list item
        if re.match(r'^[\s]*[-*]\s+', line):
            text = re.sub(r'^[\s]*[-*]\s+', '', line)
            text = bold_span(code_span(escape_xml(text)))
            flowables.append(Paragraph(f'•  {text}', styles['BulletCustom']))
            i += 1
            continue

        # Numbered list item
        if re.match(r'^\d+\.\s+', line):
            text = re.sub(r'^\d+\.\s+', '', line)
            text = bold_span(code_span(escape_xml(text)))
            flowables.append(Paragraph(text, styles['BodyCustom']))
            i += 1
            continue

        # Empty line
        if not line.strip():
            flowables.append(Spacer(1, 3))
            i += 1
            continue

        # Regular paragraph
        text = bold_span(code_span(escape_xml(line.strip())))
        flowables.append(Paragraph(text, styles['BodyCustom']))
        i += 1

    return flowables


# ── Main ───────────────────────────────────────────────────────────
def md2pdf(input):
    """Convert a Markdown file to PDF.

    Args:
        input (str): Input file path (with or without .md extension).
                     Output will be <input>.pdf or <input_without_ext>.pdf.
    """
    md_path, pdf_path = resolve_paths(input)

    if not os.path.exists(md_path):
        print(f"Error: file not found: {md_path}")
        return

    with open(md_path, "r") as f:
        md_text = f.read()

    styles = build_styles()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=os.path.basename(pdf_path),
    )

    story = render_md(md_text, styles)
    doc.build(story)
    print(f"PDF written to {pdf_path}")