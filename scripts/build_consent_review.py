"""Generate a review PDF and a matching, non-enrolling HTML acknowledgement.

This is a local review artifact, not a replacement for production enrollment.
All formats share the same reviewed text. The source consent PDF is untouched.
"""
import json
import os
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
content = json.loads((ROOT / 'docs/consent_review_content.json').read_text())
output = ROOT / 'output/pdf'
output.mkdir(parents=True, exist_ok=True)
fonts = Path(os.environ.get('POLICYLENS_PDF_FONT_DIR', str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/standard_fonts')))
for name, suffix in [('Consent', 'Regular'), ('Consent-Bold', 'Bold'), ('Consent-Italic', 'Italic'), ('Consent-BoldItalic', 'BoldItalic')]:
    pdfmetrics.registerFont(TTFont(name, str(fonts / f'LiberationSans-{suffix}.ttf')))
pdfmetrics.registerFontFamily('Consent', normal='Consent', bold='Consent-Bold', italic='Consent-Italic', boldItalic='Consent-BoldItalic')
body = ParagraphStyle('Body', fontName='Consent', fontSize=11, leading=14.6, spaceAfter=12, alignment=TA_LEFT)
meta = ParagraphStyle('Meta', parent=body, spaceAfter=3)
note = ParagraphStyle('Note', parent=body, fontSize=9, leading=11.6, textColor=colors.HexColor('#73501d'), spaceAfter=12)

story = [Paragraph(escape(content['status']), note),
         Paragraph('<b>Project Title:</b> <i>' + escape(content['title']) + '</i>', meta),
         Paragraph('<b>NTU-IRB Ref No.:</b> <i>' + escape(content['irb_reference']) + '</i>', meta),
         Spacer(1, 8)]
for i, paragraph in enumerate(content['paragraphs']):
    if i == 5:
        story += [PageBreak(), Paragraph('<b>Participant information and consent</b> (continued)', meta), Spacer(1, 10)]
    story.append(Paragraph(paragraph, body))
story += [Paragraph('[  ] ' + escape(content['agreement']), body), Paragraph('[  ] ' + escape(content['decline']), body)]

def footer(canvas, doc):
    canvas.setFont('Consent', 8)
    canvas.setFillColor(colors.HexColor('#656565'))
    canvas.drawString(54, 26, 'PolicyLens | Consent review copy | 22 September 2026')
    canvas.drawRightString(558, 26, str(doc.page))

pdf_path = output / 'policylens_consent_review.pdf'
doc = SimpleDocTemplate(str(pdf_path), pagesize=(612, 792), rightMargin=54,
    leftMargin=54, topMargin=42, bottomMargin=44, title=content['title'],
    author='PolicyLens research team')
doc.build(story, onFirstPage=footer, onLaterPages=footer)

paragraphs = '\n'.join(f'<p>{p}</p>' for p in content['paragraphs'])
html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PolicyLens - consent review</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5f4;color:#171717;font:17px/1.55 Arial,sans-serif}
main{max-width:850px;margin:36px auto;padding:44px 54px;background:white;border:1px solid #ddd}
p{margin:0 0 22px}.metadata{margin:0 0 4px;line-height:1.45}.metadata:last-of-type{margin-bottom:22px}
.review{border:1px solid #e5c88d;background:#fff8e8;padding:16px 20px;margin-bottom:24px;font-size:14px;line-height:1.5}
.review p{margin:6px 0 0}a{color:#145ba8}fieldset{border:0;padding:0;margin:0}legend{font-weight:bold;margin-bottom:12px}
label{display:flex;align-items:flex-start;gap:10px;margin:12px 0}input{margin-top:7px;flex-shrink:0}
.actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}button{padding:12px 18px;font:inherit;border:1px solid #aaa;background:#eee;border-radius:4px}
button:disabled{color:#666;cursor:not-allowed}.note{font-size:14px;color:#555;margin-top:16px}
@media(max-width:600px){main{margin:0;padding:24px 20px;border:0}body{font-size:16px}}
@media print{body{background:white;font-size:11pt}main{border:0;margin:0;padding:0;max-width:none}.actions{display:none}p,label{break-inside:avoid}a{color:inherit;text-decoration:none}}
</style></head><body><main>
'''
html += f'<aside class="review"><b>{escape(content["status"])}</b><p>{escape(content["review_notice"])}</p></aside>'
html += f'<p class="metadata"><b>Project Title:</b> <i>{escape(content["title"])}</i></p>'
html += f'<p class="metadata"><b>NTU-IRB Ref No.:</b> <i>{escape(content["irb_reference"])}</i></p><br>'
html += paragraphs
html += f'<fieldset><legend>Your decision</legend><label><input type="radio" name="consent" value="yes">{escape(content["agreement"])}</label><label><input type="radio" name="consent" value="no">{escape(content["decline"])}</label></fieldset>'
html += '<div class="actions"><button disabled>Consent and Continue</button><button disabled>I do not wish to participate</button><button onclick="window.print()">Print / Save a copy</button></div><p class="note">Review preview only. These controls do not enrol you or collect a consent record.</p></main></body></html>'
(output / 'policylens_acknowledgement_review.html').write_text(html)

plain = f'# {content["status"]}\n\n{content["review_notice"]}\n\n**Project Title:** {content["title"]}\n\n**NTU-IRB Ref No.:** {content["irb_reference"]}\n\n'
for paragraph in content['paragraphs']:
    paragraph = re.sub(r'<a href="([^"]+)">([^<]+)</a>', r'[\2](\1)', paragraph)
    paragraph = paragraph.replace('<b>', '**').replace('</b>', '**')
    plain += paragraph + '\n\n'
plain += '- [ ] ' + content['agreement'] + '\n- [ ] ' + content['decline'] + '\n'
(ROOT / 'docs/consent_review.md').write_text(plain)
print(pdf_path)
print(output / 'policylens_acknowledgement_review.html')
