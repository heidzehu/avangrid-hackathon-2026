"""Build slides 4-10 (data, methodology, Part 1 classification) in a Lazard-inspired style.
Accessible: real title placeholders, alt text on charts/tables, decorative shapes marked, native tables/charts,
high-contrast text, slide-number fields, speaker notes."""
import json, copy
from pathlib import Path
from PIL import Image
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.oxml.ns import qn

SCR = Path(__file__).parent
OUT = SCR / "06_DNH Energy_Final Submission.pptx"
D = json.load(open(SCR / "slide_data.json"))

NAVY, BLUE, LBLUE, CAT, GREY, GOLD = "232A3E", "0A6FAF", "B5DAF3", "D6E9F8", "DCD8D2", "B8862F"
TXT, MUTED, RULE, PALE = "1F1F1F", "4A4A4A", "1F1F1F", "F2F7FC"
SERIF, SANS = "Garamond", "Arial"
rgb = RGBColor.from_string

# ---------- gradient header image (decorative) ----------
W, Hh = 3000, 190
img = Image.new("RGB", (W, Hh))
stops = [(0.00, (0x23, 0x2A, 0x3E)), (0.18, (0x23, 0x2A, 0x3E)), (0.40, (0x4C, 0x6E, 0x96)),
         (0.58, (0x8C, 0xC8, 0xF0)), (0.80, (0xC9, 0xE4, 0xF8)), (1.00, (0xF4, 0xF9, 0xFE))]
px = img.load()
for x in range(W):
    f = x / (W - 1)
    for i in range(len(stops) - 1):
        a, ca = stops[i]; b, cb = stops[i + 1]
        if a <= f <= b:
            t = (f - a) / (b - a) if b > a else 0
            c = tuple(int(ca[k] + (cb[k] - ca[k]) * t) for k in range(3)); break
    for y in range(Hh): px[x, y] = c
HEADER = SCR / "header_gradient.png"; img.save(HEADER)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
TITLE_ONLY = prs.slide_layouts[5]


# ---------- helpers ----------
def cnvpr(shape):
    el = shape._element
    for tag in ("p:nvSpPr", "p:nvPicPr", "p:nvGraphicFramePr", "p:nvCxnSpPr", "p:nvGrpSpPr"):
        n = el.find(qn(tag))
        if n is not None: return n.find(qn("p:cNvPr"))


def alt(shape, text, name=None):
    c = cnvpr(shape); c.set("descr", text)
    if name: c.set("name", name)


def decorative(shape, name="Decorative"):
    c = cnvpr(shape); c.set("name", name)
    ext = etree.SubElement(etree.SubElement(c, qn("a:extLst")), qn("a:ext"))
    ext.set("uri", "{C183D7F6-B498-43B3-948B-1728B52AA6E4}")
    d = etree.SubElement(ext, "{http://schemas.microsoft.com/office/drawing/2017/decorative}decorative")
    d.set("val", "1")


def style_run(r, size=12, bold=False, color=TXT, font=SANS, italic=False):
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = rgb(color); r.font.name = font


def text(slide, x, y, w, h, paras, name, anchor=MSO_ANCHOR.TOP, margin=0.0):
    """paras: list of dicts {runs:[(text, opts)], align, bullet, space_after, level}"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"): setattr(tf, f"margin_{side}", Inches(margin))
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        if p.get("space_after") is not None: para.space_after = Pt(p["space_after"])
        if p.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            lvl = p.get("level", 0)
            pPr.set("marL", str(int(Inches(0.2 + 0.2 * lvl)))); pPr.set("indent", str(int(-Inches(0.18))))
            bu = etree.SubElement(pPr, qn("a:buFont")); bu.set("typeface", "Arial")
            bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", "•" if lvl == 0 else "–")
        for t, o in p["runs"]:
            r = para.add_run(); r.text = t; style_run(r, **o)
    return tb


def box(slide, x, y, w, h, fill, name, line=None, shape=MSO_SHAPE.RECTANGLE, deco=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    if fill: s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    else: s.fill.background()
    if line: s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else: s.line.fill.background()
    s.shadow.inherit = False
    if deco: decorative(s, name)
    return s


def rule(slide, x, y, w, name="Rule", color=RULE, weight=0.75):
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.line.color.rgb = rgb(color); ln.line.width = Pt(weight); decorative(ln, name)
    return ln


def slide_number(slide):
    tb = slide.shapes.add_textbox(Inches(12.0), Inches(7.0), Inches(0.83), Inches(0.3)); tb.name = "Slide number"
    tf = tb.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, qn("a:fld")); fld.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}"); fld.set("type", "slidenum")
    rPr = etree.SubElement(fld, qn("a:rPr")); rPr.set("lang", "en-US"); rPr.set("sz", "1100")
    sf = etree.SubElement(rPr, qn("a:solidFill")); c = etree.SubElement(sf, qn("a:srgbClr")); c.set("val", NAVY)
    lat = etree.SubElement(rPr, qn("a:latin")); lat.set("typeface", SANS)
    t = etree.SubElement(fld, qn("a:t")); t.text = "‹#›"
    r = p.add_run(); r.text = "  |"; style_run(r, 11, color=NAVY)


def frame(section, title, subtitle, sources, notes):
    s = prs.slides.add_slide(TITLE_ONLY)
    # title placeholder first -> first in reading order
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.5), Inches(0.98), Inches(12.33), Inches(0.62)
    tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    for side in ("left", "right", "top", "bottom"): setattr(tf, f"margin_{side}", 0)
    tf.text = ""; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = title; style_run(r, 28, color=NAVY, font=SERIF)
    text(s, 0.5, 1.66, 12.33, 0.62, [{"runs": [(subtitle, dict(size=14, color=TXT))]}], "Subtitle")
    pic = s.shapes.add_picture(str(HEADER), 0, 0, prs.slide_width, Inches(0.78)); decorative(pic, "Header band")
    text(s, 0.5, 0.25, 4.5, 0.3, [{"runs": [("AMI CUSTOMER INSIGHTS", dict(size=10, color="FFFFFF", bold=True))]}],
         "Header project name").text_frame.paragraphs[0].runs[0].font._element.set("spc", "300")
    hdr = text(s, 6.3, 0.25, 6.53, 0.3, [{"runs": [(section.upper(), dict(size=10, color=NAVY))], "align": PP_ALIGN.RIGHT}],
               "Header section label")
    hdr.text_frame.paragraphs[0].runs[0].font._element.set("spc", "300")
    # footer
    text(s, 0.5, 6.98, 2.6, 0.32, [{"runs": [("Avangrid Hackathon 2026", dict(size=13, color=NAVY, font=SERIF))]}], "Footer")
    if sources:
        paras = [{"runs": [(lab + "  ", dict(size=9, color=MUTED, bold=True)), (txt, dict(size=9, color=MUTED))],
                  "space_after": 0} for lab, txt in sources]
        text(s, 3.2, 6.78, 8.6, 0.62, paras, "Source notes", anchor=MSO_ANCHOR.BOTTOM)
    slide_number(s)
    s.notes_slide.notes_text_frame.text = notes
    return s


def panel_header(s, x, y, w, label):
    text(s, x, y, w, 0.3, [{"runs": [(label, dict(size=14, color=TXT))]}], f"Panel heading: {label}", margin=0.05)
    rule(s, x, y + 0.33, w, f"Panel rule: {label}")


def no_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag)); ln.set("w", "0"); etree.SubElement(ln, qn("a:noFill"))


def border(cell, side, color, w_pt=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn(f"a:ln{side}"))
    if old is not None: tcPr.remove(old)
    ln = etree.Element(qn(f"a:ln{side}")); ln.set("w", str(int(Pt(w_pt))))
    sf = etree.SubElement(ln, qn("a:solidFill")); c = etree.SubElement(sf, qn("a:srgbClr")); c.set("val", color)
    # borders must precede fill elements in tcPr
    tcPr.insert(0, ln)


def table(s, x, y, w, col_w, rows, name, alt_text, size=11, row_h=0.3, header_fill=None, fills=None,
          bold_rows=(), aligns=None, header_color=TXT, rule_color="BFBFBF"):
    nr, nc = len(rows), len(rows[0])
    gf = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    gf.name = name; alt(gf, alt_text, name)
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "1"); tblPr.set("bandRow", "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None: sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # No Style, No Grid
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.01)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            val = rows[i][j]
            tf = c.text_frame; tf.word_wrap = True; tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT)
            r = p.add_run(); r.text = str(val)
            style_run(r, size, bold=(i == 0 or i in bold_rows), color=(header_color if i == 0 else TXT))
            end = etree.SubElement(p._p, qn("a:endParaRPr")); end.set("lang", "en-US"); end.set("sz", str(int(size * 100)))
            f = None
            if i == 0 and header_fill: f = header_fill
            elif fills and fills.get((i, j)): f = fills[(i, j)]
            elif fills and fills.get((i, None)): f = fills[(i, None)]
            if f: c.fill.solid(); c.fill.fore_color.rgb = rgb(f)
            else: c.fill.background()
            no_border(c)
            if i == 0: border(c, "B", RULE, 1.0)
            elif i == nr - 1: border(c, "B", RULE, 1.0)
            else: border(c, "B", rule_color, 0.5)
    return gf


def chart_style(ch, size=11):
    ch.font.size = Pt(size); ch.font.name = SANS; ch.font.color.rgb = rgb(TXT)


# =====================================================================================
# Slide 4 — Data preparation
# =====================================================================================
s = frame("Data & Methodology", "Data Preparation—Turning 298 Million Readings into One Clean Dataset",
          "Before modeling anything, we cleaned the meter and weather data, lined everything up hour by hour, and flagged "
          "problem meters instead of throwing them out",
          [("Source:", "Avangrid 2026 Hackathon AMI (9/1/2025–8/31/2026) and hourly weather data; team analysis."),
           ("Note:", "Reported findings use the original 34,238-customer snapshot (v2 delivery lacked customers 18,152–18,346 and "
                     "18,747–19,953). The later expanded dataset (35,236 customers) has not been fully recalculated.")],
          "Go through the four steps from top to bottom. Three things are worth saying out loud. The meters only record "
          "power coming in from the grid, so solar homes show up as zeros, not negative numbers. Every reading is a full hour. "
          "And one of the files we were sent was an exact copy of another one; we caught it, removed the duplicates and "
          "switched to the corrected file. The whole pipeline runs in under a minute on a laptop, which is a good sign it can "
          "scale to millions of customers.")
panel_header(s, 0.5, 2.4, 7.3, "How we prepared the data")
steps = [("Load", "We read all 199 data files straight out of the zips into a fast, compressed format. It takes about 12 seconds"),
         ("Remove duplicates", "One delivered file was an exact copy of another. We caught it, removed the repeats and "
                               "switched to the corrected v2 file"),
         ("Line up", "Each customer gets one row of 8,760 hours (only 0.63% missing), with weather matched to the same hours"),
         ("Flag problem meters", "Meters with patchy data (334), almost no use (306) or business-sized loads (68) are left "
                                 "out of training but kept in the results")]
for i, (h, d) in enumerate(steps):
    yy = 2.95 + i * 0.9
    c = box(s, 0.55, yy + 0.06, 0.46, 0.46, NAVY, f"Step {i+1} marker", shape=MSO_SHAPE.OVAL)
    c.text_frame.text = ""; r = c.text_frame.paragraphs[0].add_run(); r.text = str(i + 1)
    style_run(r, 14, bold=True, color="FFFFFF"); c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    alt(c, f"Step {i+1}")
    text(s, 1.2, yy, 6.5, 0.8, [{"runs": [(h, dict(size=14, bold=True, color=NAVY))], "space_after": 1},
                                {"runs": [(d, dict(size=12))]}], f"Step {i+1}: {h}")
arrow = box(s, 8.0, 2.85, 0.32, 3.75, "D9E3EC", "Flow arrow", shape=MSO_SHAPE.ISOSCELES_TRIANGLE, deco=True)
arrow.rotation = 90; arrow.left, arrow.top, arrow.width, arrow.height = Inches(6.28), Inches(4.57), Inches(3.75), Inches(0.32)
panel_header(s, 8.5, 2.4, 4.33, "Dataset at a glance")
table(s, 8.5, 2.88, 4.33, [2.2, 2.13],
      [["Metric", "Value"], ["Hourly readings (cleaned)", "298.0 M"], ["Customers analyzed", "34,238"],
       ["Hours per customer", "8,760"], ["Missing hours", "0.63%"], ["Outdoor temperature range", "−5°F to 92°F"],
       ["Busiest hour (all meters)", "79.2 MW (7/2/26, 5–6 pm)"]],
      "Dataset at a glance table", "Table: dataset statistics. 298.0 million customer-hours after de-duplication, 34,238 "
      "customers, 8,760 hours each, 0.63% missing hours, outdoor temperature −5°F to 92°F, system peak 79.2 MW on "
      "July 2, 2026 at 5–6 pm.", size=11, row_h=0.29, aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT])
text(s, 8.5, 5.12, 4.33, 1.5,
     [{"runs": [("Grid power only: ", dict(size=12, bold=True)), ("solar homes show up as 0, never negative", dict(size=12))],
       "bullet": True, "space_after": 3},
      {"runs": [("Full hours: ", dict(size=12, bold=True)), ("gaps are missing rows, not half-recorded hours", dict(size=12))],
       "bullet": True, "space_after": 3},
      {"runs": [("Summer peak: ", dict(size=12, bold=True)),
                ("all 100 busiest hours fall in June–August, mostly between 3 and 8 pm", dict(size=12))], "bullet": True}],
     "Data findings")

# =====================================================================================
# Slide 5 — Feature engineering
# =====================================================================================
s = frame("Data & Methodology", "Feature Engineering—Describing Each Home by How It Uses Power",
          "We describe every home with 65 measurements that blend meter data with outdoor temperature, and each one maps to "
          "something a utility engineer would recognize",
          [("Source:", "Team analysis of Avangrid 2026 Hackathon AMI and weather data."),
           ("Note:", "T_d = daily mean outdoor temperature; balance points T_H and T_C chosen per customer by grid search; "
                     "all customers fitted in one vectorized least-squares pass.")],
          "This is where weather comes in. The first equation splits each home's daily use into a base amount, extra for "
          "heating and extra for cooling. The second spots EV charging: a charger pulls a steady block of power for a few "
          "hours, which heating and cooking don't do. The third tells us how much each home adds during the system's 100 "
          "busiest hours, and we use it again in the battery section.")
panel_header(s, 0.5, 2.4, 6.85, "What we measure")
table(s, 0.5, 2.88, 6.85, [1.55, 3.0, 2.3],
      [["Group", "Examples", "What it tells us"],
       ["Size & shape", "Yearly kWh, peak kW, load factor, always-on load", "How big the home is"],
       ["Weather response", "Extra use per degree of cold or heat", "Electric heat, AC"],
       ["Timing", "Use by time of day, peak hour, weekends", "When people are home"],
       ["EV sessions", "Charging power, how often, how long", "Level 1 or Level 2 charging"],
       ["Seasons", "Winter vs. summer use, empty months", "Pools, hot tubs, vacation homes"],
       ["Peak timing", "Use during the system's 100 busiest hours", "Share of the system peak"]],
      "Feature families table", "Table: six feature families. Size and shape (annual kWh, p99 kW, load factor, baseload) "
      "detects home size and always-on load. Weather response (heating and cooling slopes, balance points, R squared) "
      "detects electric heat and AC. Timing detects occupancy and evening peaks. EV sessions detect Level 1 and Level 2 "
      "charging. Season and occupancy detects pools, hot tubs and seasonal homes. Grid coincidence measures peak "
      "contribution.", size=11, row_h=0.5)
panel_header(s, 7.75, 2.4, 5.08, "Key equations")
eqs = [("How each home responds to temperature",
        "kWh_d = β₀ + β_H·max(0, T_H − T_d) + β_C·max(0, T_d − T_C)",
        "β_H, β_C = extra kWh per day for each degree colder or hotter"),
       ("Spotting EV charging sessions",
        "E_h = kWh_h − P20(day);  session = 1–10 h at E_h ≥ 3 kW, flat",
        "Flat = steady, like a charger; Level 1 looks for 0.9–3 kW overnight"),
       ("How much each home adds to the peak",
        "CP_i = mean kW of customer i over the system's top-100 hours",
        "Coincidence factor = CP_i ÷ the home's own peak kW")]
for i, (h, e, d) in enumerate(eqs):
    yy = 2.92 + i * 1.22
    text(s, 7.75, yy, 5.08, 0.28, [{"runs": [(h, dict(size=12, bold=True, color=NAVY))]}], f"Equation {i+1} heading")
    b = box(s, 7.75, yy + 0.32, 5.08, 0.46, PALE, f"Equation {i+1}")
    b.text_frame.text = ""; b.text_frame.margin_left = Inches(0.12); b.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = b.text_frame.paragraphs[0].add_run(); r.text = e; style_run(r, 13, color=TXT, font="Cambria")
    alt(b, f"Equation: {e}")
    text(s, 7.75, yy + 0.82, 5.08, 0.3, [{"runs": [(d, dict(size=11, color=MUTED))]}], f"Equation {i+1} note")

# =====================================================================================
# Slide 6 — Classification method
# =====================================================================================
s = frame("Part 1 · Customer Classification",
          "Classification Method—Models Where We Have Labels, Rules Where We Don't",
          "Only EVs and heat pumps came with (partial) labels, so a model learns those; for everything else we wrote clear, "
          "physics-based rules that anyone can check",
          [("Source:", "Elkan & Noto (2008), “Learning classifiers from only positive and unlabeled data”; team analysis."),
           ("Note:", "c = 0.25 is the organizers' stated labeling rate (≈25% of likely EV and heat-pump owners are listed). "
                     "LightGBM: 150 trees × 4 leaves (EV); 217 × 8 (heat pump, from nested-CV search).")],
          "Here's the main idea. Only about a quarter of the real EV and heat-pump owners are on Avangrid's list, so the "
          "unlabeled customers include plenty of hidden owners. A normal model would learn that those people don't own one, "
          "which is wrong. So instead we train the model to spot customers who look like the labeled ones, then scale that "
          "up by the known 25% rate to get a real probability. For appliances with no labels at all, we wrote simple rules, "
          "so an engineer can check exactly why any home was flagged.")
flow = [("65 features", "for each home"), ("LightGBM model", "labeled vs. the rest"), ("Fair score s(x)", "from homes it never saw"),
        ("P(owner) = s(x) ÷ 0.25", "a real probability"), ("Ranked list", "top K = labeled ÷ 0.25")]
cw, x0 = 2.52, 0.5
for i, (a, b) in enumerate(flow):
    shp = MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON
    c = box(s, x0 + i * (cw - 0.06), 2.42, cw, 0.8, NAVY if i % 2 == 0 else BLUE, f"Flow step {i+1}", shape=shp)
    tf = c.text_frame; tf.text = ""; tf.margin_left = Inches(0.32 if i else 0.15); tf.margin_right = Inches(0.25)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = a; style_run(r, 12, True, "FFFFFF")
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER; r = p2.add_run(); r.text = b; style_run(r, 10, color="FFFFFF")
    alt(c, f"Step {i+1}: {a}, {b}")
panel_header(s, 0.5, 3.5, 6.0, "EVs & heat pumps: learning from partial labels")
text(s, 0.5, 3.98, 6.0, 1.55,
     [{"runs": [("The catch: ", dict(size=12, bold=True)), ("only about 1 in 4 owners is on the list, so the "
                 "“unlabeled” group is hiding the other 3", dict(size=12))], "bullet": True, "space_after": 4},
      {"runs": [("Our fix: ", dict(size=12, bold=True)), ("learn who looks like a labeled owner, then scale up by the known "
                 "25% rate", dict(size=12))], "bullet": True, "space_after": 4},
      {"runs": [("Sanity check: ", dict(size=12, bold=True)), ("the probabilities add up to about the expected number of "
                 "owners (EV 703 vs. 724; heat pump 303 vs. 340)", dict(size=12))], "bullet": True}],
     "PU learning bullets")
b = box(s, 0.5, 5.68, 6.0, 0.62, PALE, "PU equation")
b.text_frame.text = ""; b.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE; b.text_frame.margin_left = Inches(0.12)
r = b.text_frame.paragraphs[0].add_run(); r.text = "P(owner | x) = P(labeled | x) ÷ c ,   c = 0.25"
style_run(r, 14, color=TXT, font="Cambria"); alt(b, "Equation: P(owner given x) equals P(labeled given x) divided by c, with c = 0.25")
panel_header(s, 6.83, 3.5, 6.0, "Everything else: rules based on physics")
text(s, 6.83, 3.98, 6.0, 1.55,
     [{"runs": [("Each rule ", dict(size=12, bold=True)), ("checks 1–3 things you can see in the data. A pool pump, for "
                 "example, adds 0.6+ kW in summer daytime on a timer, whatever the weather", dict(size=12))], "bullet": True, "space_after": 4},
      {"runs": [("Soft cut-offs: ", dict(size=12, bold=True)), ("each check scores 0 to 1 instead of a hard yes or no; "
                 "a home is flagged at 0.5", dict(size=12))], "bullet": True, "space_after": 4},
      {"runs": [("Easy to audit: ", dict(size=12, bold=True)), ("every flag traces back to a threshold you can read", dict(size=12))],
       "bullet": True}], "Rules bullets")
b = box(s, 6.83, 5.68, 6.0, 0.62, PALE, "Ramp equation")
b.text_frame.text = ""; b.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE; b.text_frame.margin_left = Inches(0.12)
r = b.text_frame.paragraphs[0].add_run(); r.text = "conf = Π r(x_k; x₀, w) ,   r = 1 ÷ (1 + e^(−(x − x₀)/w))"
style_run(r, 14, color=TXT, font="Cambria")
alt(b, "Equation: confidence is the product of logistic ramps r of x with threshold x0 and width w")

# =====================================================================================
# Slide 7 — Validation
# =====================================================================================
s = frame("Part 1 · Customer Classification", "Model Validation—Our EV Model Finds Owners 17× Better Than Random",
          "We test every home with a model that never saw it. Contacting just the top 2% of homes reaches 37% of known EV "
          "owners; heat pumps are much harder to pick out",
          [("Source:", "Team analysis; out-of-fold scores from 5-fold stratified CV repeated over 5 seeds."),
           ("Note:", "K = labeled owners ÷ 0.25 (expected owner population). Estimated precision = recall@K × labeled ÷ 0.25 ÷ K, "
                     "unbiased under random labeling. Lift = recall@K ÷ share of customers flagged.")],
          "Here's how to read the chart: if Avangrid contacted the top X% of homes on our list, what share of real owners "
          "would it reach? The dashed line is picking homes at random. For EVs, the top 2% of homes include 37% of the "
          "owners the model hadn't seen, which is 17 times better than random, and about 1 in 3 homes on the list should "
          "really have an EV. Heat pumps are harder, and the next few slides explain why. The features doing the most work "
          "are the charging-session ones we built, which tells us the model picked up real charging behavior, not noise.")
cd = CategoryChartData(); cats = [str(i) for i in range(31)]; cd.categories = cats
cd.add_series("EV (AUC 0.80)", D["gains_ev"]); cd.add_series("Heat pump (AUC 0.77)", D["gains_hp"])
cd.add_series("Random", [float(i) for i in range(31)])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.5), Inches(2.42), Inches(7.4), Inches(4.25), cd)
gf.name = "Cumulative gains chart"
alt(gf, "Line chart: cumulative share of held-out labeled owners found versus share of customers flagged, 0 to 30%. "
        "EV reaches 49% at 5% flagged, 57% at 10% and 72% at 30%. Heat pump reaches 32% at 5%, 48% at 10% and 74% at 30%. "
        "Random selection is a straight diagonal line.", "Cumulative gains chart")
ch = gf.chart; chart_style(ch)
ch.has_title = True; ch.chart_title.text_frame.text = "Share of known owners found (%) vs. share of homes contacted (%)"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 12, color=TXT)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
for ser, col, dash in zip(ch.plots[0].series, [BLUE, GOLD, "7F7F7F"], [None, None, "dash"]):
    ser.format.line.color.rgb = rgb(col); ser.format.line.width = Pt(2.5 if dash is None else 1.5); ser.smooth = False
    ser.marker.style = None
    if dash: ser.format.line.dash_style = 4
    from pptx.enum.chart import XL_MARKER_STYLE
    ser.marker.style = XL_MARKER_STYLE.NONE
va = ch.value_axis; va.maximum_scale = 80; va.minimum_scale = 0; va.major_unit = 20
va.major_gridlines.format.line.color.rgb = rgb("D9D9D9"); va.format.line.fill.background()
ca = ch.category_axis; ca.tick_labels.font.size = Pt(10)
from pptx.enum.chart import XL_TICK_LABEL_POSITION
ca_el = ca._element
for tag, val in (("c:tickLblSkip", "5"), ("c:tickMarkSkip", "5")):
    e = etree.SubElement(ca_el, qn(tag)); e.set("val", val)
panel_header(s, 8.3, 2.4, 4.53, "How well it works (on unseen homes)")
table(s, 8.3, 2.88, 4.53, [2.25, 1.14, 1.14],
      [["Metric", "EV", "Heat pump"], ["Labeled owners used", "181", "85"], ["AUC", "0.80", "0.77"],
       ["Expected owners (K)", "724", "340"], ["Owners found in top K", "37%", "9%"], ["Better than random", "17×", "9×"],
       ["Share of list that's right", "36%", "9%"], ["Probabilities vs. K", "703 / 724", "303 / 340"]],
      "Validation metrics table", "Table: validation metrics. EV versus heat pump: labeled owners used 181 and 85; AUC 0.80 "
      "and 0.77; expected owners K 724 and 340; recall in top K 37% and 9%; lift 17 and 9 times; estimated precision 36% "
      "and 9%; sum of probabilities 703 versus 724 and 303 versus 340.", size=11, row_h=0.3,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
text(s, 8.3, 5.4, 4.53, 1.25,
     [{"runs": [("What gives an EV away: ", dict(size=12, bold=True, color=NAVY)),
                ("how much power each charge pulls, plus how often and how long Level 1 sessions run", dict(size=12))], "space_after": 6},
      {"runs": [("What gives a heat pump away: ", dict(size=12, bold=True, color=NAVY)),
                ("overnight heating, cooling use and always-on load", dict(size=12))]}], "Top drivers")

# =====================================================================================
# Slide 8 — Robustness
# =====================================================================================
s = frame("Part 1 · Customer Classification", "Robustness—Different Algorithms and Tuning All Give the Same Answer",
          "Changing the algorithm or tuning its settings moves accuracy by only about ±0.01, so the gains come from the "
          "features we built, not the model we picked",
          [("Source:", "Team analysis; identical customers, features and 5-fold × 5-seed CV for every model."),
           ("Note:", "Random forest A: min leaf 20; B: min leaf 50, class-balanced. Nested CV: 40-configuration random search inside each outer-training fold (5 folds × 2 repeats), "
                     "winner scored on the untouched fold. Rule fixed in advance: adopt if gain > 1 SE and recall@K not worse.")],
          "We ran two checks to make sure we weren't fooling ourselves. On the left, four different model types all land "
          "within about one hundredth of each other, so we kept the fastest, LightGBM. On the right, we tried 40 different "
          "settings, and we tested them on customers the search never saw, so a lucky result can't sneak through. We set the "
          "rule for switching before we looked at the results. Tuning didn't help the EV model, so we kept it. It helped "
          "the heat pump model in 7 of 10 tests, so we switched that one.")
cd = CategoryChartData()
cd.categories = ["LightGBM (production)", "Random forest A", "Random forest B", "Logistic regression"]
cd.add_series("EV", [0.797, 0.791, 0.804, 0.800]); cd.add_series("Heat pump", [0.760, 0.746, 0.773, 0.763])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(2.42), Inches(7.0), Inches(4.25), cd)
gf.name = "Model comparison chart"
alt(gf, "Clustered column chart of out-of-fold AUC by model. EV: LightGBM 0.797, random forest A (min leaf 20) 0.791, random forest B "
        "(min leaf 50, balanced) 0.804, logistic regression 0.800. Heat pump: 0.760, 0.746, 0.773, 0.763.", "Model comparison chart")
ch = gf.chart; chart_style(ch)
ch.has_title = True; ch.chart_title.text_frame.text = "Accuracy (AUC) on unseen homes, by algorithm"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 12, color=TXT)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
pl = ch.plots[0]; pl.gap_width = 80; pl.overlap = -10
pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = "0.000"; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(10)
for ser, col in zip(pl.series, [BLUE, GOLD]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
va = ch.value_axis; va.minimum_scale = 0.6; va.maximum_scale = 0.85; va.major_unit = 0.05
va.tick_labels.number_format = "0.00"; va.tick_labels.number_format_is_linked = False
va.major_gridlines.format.line.color.rgb = rgb("D9D9D9"); va.format.line.fill.background()
ch.category_axis.tick_labels.font.size = Pt(10)
panel_header(s, 7.9, 2.4, 4.93, "Tuning test (nested cross-validation)")
table(s, 7.9, 2.88, 4.93, [1.05, 1.45, 0.88, 1.55],
      [["Model", "AUC gain (± SE)", "Tests won", "Decision"], ["EV", "+0.001 ± 0.006", "5 of 10", "Keep original"],
       ["Heat pump", "+0.013 ± 0.008", "7 of 10", "Switch to tuned"]],
      "Nested search table", "Table: nested cross-validation search results. EV: AUC gain plus 0.001 plus or minus 0.006, "
      "tuned model won 5 of 10 folds, keep hand-tuned model. Heat pump: gain plus 0.013 plus or minus 0.008, won 7 of 10 "
      "folds, adopt tuned model.", size=11, row_h=0.36)
text(s, 7.9, 4.12, 4.93, 1.2,
     [{"runs": [("What we tuned: ", dict(size=12, bold=True)), ("number and size of trees, learning speed, minimum group "
                 "size, sampling and regularization", dict(size=12))], "bullet": True, "space_after": 4},
      {"runs": [("What we found: ", dict(size=12, bold=True)), ("AUC ranged 0.71–0.80 (EV) and 0.67–0.77 (heat pump); "
                 "bigger trees didn't help", dict(size=12))], "bullet": True}], "Search notes")
b = box(s, 7.9, 5.55, 4.93, 0.95, None, "Takeaway callout", line=NAVY)
tf = b.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.12); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.text = ""; r = tf.paragraphs[0].add_run()
r.text = "Hourly data limits accuracy more than the algorithm does, so we put our effort into building better features"
style_run(r, 12, color=NAVY, italic=True); tf.paragraphs[0].alignment = PP_ALIGN.CENTER; alt(b, "Takeaway: " + r.text)

# =====================================================================================
# Slide 9 — Inventory
# =====================================================================================
s = frame("Part 1 · Customer Classification", "Appliance Inventory—Two-Thirds Run AC; About 2–2.5% Charge an EV",
          "Adding up the flags gives an inventory for the whole territory. EV and pool homes draw the most at the summer peak, "
          "while vacation homes barely register",
          [("Source:", "Team analysis of 34,238 customers; customer-level flags and probabilities in outputs/customer_classification.csv."),
           ("Note:", "¹ Two different estimates: 844 homes flagged by the classifier (2.5%) vs. ≈724 estimated owners from labeled ÷ 0.25 "
                     "(2.1%); flagged EV list ≈36% precise, heat-pump candidates ≈9%. Peak kW = mean demand in the system's top-100 hours.")],
          "This is the inventory the challenge asked for: how many homes have each appliance and what their usage looks "
          "like. The column to point at is kW at system peak, meaning how much a home draws during the 100 busiest hours of "
          "the year. EV and pool homes draw about 3 to 3.5 kW then, compared with 1.9 kW for a typical home, so they're the "
          "first place to look for demand flexibility. Heat pumps show as 88 confirmed plus a list of likely ones, because "
          "hourly data can't reliably tell them apart from baseboard heat.")
L, R_ = PP_ALIGN.LEFT, PP_ALIGN.RIGHT
rows = [["Group", "Technology", "Customers", "% of homes", "Est. population", "Median kWh/yr", "Median peak kW",
         "kW at system peak", "Method"],
        ["EV charging", "EV (all)", "844", "2.5%", "724¹", "12,554", "11.3", "3.01", "PU model"],
        ["", "   Level 2", "499", "1.5%", "–", "15,253", "13.9", "3.36", "Session rule"],
        ["", "   Level 1 / occasional", "345", "1.0%", "–", "10,367", "6.8", "2.51", "Session rule"],
        ["Space & water", "Air conditioning", "23,102", "67.5%", "23,102", "7,214", "5.5", "2.37", "Rule"],
        ["", "Electric space heat (any)", "6,102", "17.8%", "6,102", "9,512", "6.3", "1.65", "Rule"],
        ["", "   Heat pump (labeled)", "88", "0.3%", "340¹", "11,158", "7.3", "2.14", "Labels"],
        ["", "   Heat-pump candidates", "332", "1.0%", "–", "16,280", "8.9", "2.02", "PU model"],
        ["", "   Resistance heat", "2,800", "8.2%", "2,800", "14,436", "8.6", "1.90", "Rule"],
        ["", "Electric water heater", "1,332", "3.9%", "1,332", "13,052", "9.1", "2.53", "Rule"],
        ["Other loads & DER", "Pool pump", "957", "2.8%", "957", "11,985", "7.5", "3.53", "Rule"],
        ["", "Hot tub / spa", "1,930", "5.6%", "1,930", "12,052", "7.5", "2.64", "Rule"],
        ["", "Solar PV (known)", "397", "1.2%", "397", "8,280", "6.8", "1.34", "Labels"],
        ["", "Solar + battery", "4", "<0.1%", "4", "7,235", "11.1", "0.49", "Rule"],
        ["Discovered", "Seasonal / vacant home", "2,973", "8.7%", "2,973", "1,768", "2.5", "0.49", "Rule"],
        ["", "Large non-residential-like", "68", "0.2%", "68", "44,236", "22.9", "8.06", "Rule"],
        ["", "All customers", "34,238", "100%", "–", "6,699", "5.4", "1.89", ""]]
fills = {}
for i in range(1, len(rows)): fills[(i, 0)] = CAT
for j in range(len(rows[0])): fills[(len(rows) - 1, j)] = LBLUE
table(s, 0.5, 2.38, 12.33, [1.55, 2.35, 1.0, 0.95, 1.2, 1.25, 1.15, 1.35, 1.53], rows, "Appliance inventory table",
      "Table: appliance inventory by technology with customers, share of homes, estimated population, median annual kWh, "
      "median peak kW, mean kW at system peak and method. EV all 844 customers, 2.5%, estimated population 724, 3.01 kW at "
      "system peak. Level 2 499, Level 1 345. Air conditioning 23,102, 67.5%. Electric space heat 6,102, 17.8%. Heat pump "
      "labeled 88, estimated population 340; heat-pump candidates 332; resistance heat 2,800. Electric water heater 1,332. "
      "Pool pump 957, 2.8%, 3.53 kW at peak. Hot tub 1,930. Solar PV 397. Solar plus battery 4. Seasonal or vacant homes "
      "2,973, 8.7%, 0.49 kW at peak. Large non-residential-like 68. All customers 34,238, median 6,699 kWh per year, 1.89 kW "
      "at system peak.", size=11, row_h=0.24, fills=fills, bold_rows=(len(rows) - 1,),
      aligns=[L, L, R_, R_, R_, R_, R_, R_, L])

# =====================================================================================
# Slide 10 — Evidence
# =====================================================================================
s = frame("Part 1 · Customer Classification",
          "Classification Evidence—Every Appliance Leaves a Clear Mark in the Data",
          "Each flag rests on something you can see in the data, and the homes our model newly flagged as EVs charge "
          "overnight just like the known owners do",
          [("Source:", "Team analysis; chart shows mean hourly kWh in mild months (Apr, May, Oct) to exclude heating and cooling."),
           ("Note:", "“Other homes” excludes EV, heat-pump, solar, seasonal, non-residential-like and electric-heat homes. "
                     "Full thresholds in the GitHub README.")],
          "Each row answers the question: what does this appliance look like in the data, and how do we flag it? The chart "
          "is our reality check. The 658 homes the model newly flagged as EVs rise in the evening and stay high overnight, "
          "just like the 186 known owners, and they look nothing like other homes. The box at the bottom is where we're "
          "upfront about heat pumps: in hourly data they look a lot like baseboard heat.")
rows = [["Appliance", "What it looks like in the data", "How we flag it (score ≥ 0.5)"],
        ["EV Level 2", "Steady 1–10 h blocks of +3 to 12 kW", "Block ≥ 4.5 kW and ≥ 2 sessions / 30 days"],
        ["EV Level 1", "Long +0.9 to 3 kW overnight runs", "4–14 h runs starting 6 pm–2 am"],
        ["Air conditioning", "Use climbs with summer heat", "Cooling slope ≥ 0.35, temp. corr. ≥ 0.25"],
        ["Electric heat", "Use climbs as it gets colder", "Heating is ≥ 12% of yearly use"],
        ["Resistance heat", "Steep rise on cold nights", "Electric heat + ≥ 1 kW cold-night rise"],
        ["Water heater", "Short 2–5.5 kW one-hour bursts", "≥ 0.35 bursts/day outside heating season"],
        ["Pool pump", "Higher summer daytime use on a timer", "+0.6 kW vs. spring, not weather-driven"],
        ["Hot tub / spa", "Higher overnight use in winter", "+0.4 to 2 kW, without space heating"],
        ["Solar + battery", "No grid use midday or evening", "Known solar; ≥ 30% of summer evenings ~0"]]
fills = {(i, 0): CAT for i in range(1, len(rows))}
table(s, 0.5, 2.38, 7.55, [1.45, 2.75, 3.35], rows, "Technology evidence table",
      "Table: fingerprint and flag rule per technology. EV Level 2: flat 1 to 10 hour blocks of 3 to 12 kW; flagged if "
      "block at least 4.5 kW and at least 2 sessions per 30 days. EV Level 1: long 0.9 to 3 kW overnight runs. Air "
      "conditioning: load climbs with summer heat. Electric heat: heating at least 12% of annual kWh. Resistance heat: "
      "steep rise on cold nights. Water heater: isolated 2 to 5.5 kW one-hour bursts. Pool pump: timer-driven summer "
      "daytime floor. Hot tub: raised winter overnight floor without space heat. Solar plus battery: known solar home "
      "with near-zero evening imports.", size=11, row_h=0.42, fills=fills)
cd = CategoryChartData(); cd.categories = [str(h) for h in range(1, 25)]
n = D["ev_shape"]["n"]
cd.add_series(f"Newly flagged EV (n={n[1]})", D["ev_shape"]["new"])
cd.add_series(f"Labeled EV (n={n[0]})", D["ev_shape"]["labeled"])
cd.add_series(f"Other homes (n={n[2]:,})", D["ev_shape"]["other"])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(8.35), Inches(2.38), Inches(4.48), Inches(2.95), cd)
gf.name = "EV load shape chart"
ev = D["ev_shape"]
alt(gf, f"Line chart: mean kWh per hour by hour ending in mild months. Newly flagged EVs peak at {max(ev['new']):.2f} kWh "
        f"around 9 pm with high overnight load; labeled EVs follow the same shape peaking at {max(ev['labeled']):.2f}; other "
        f"homes stay below {max(ev['other']):.2f} kWh with no overnight rise.", "EV load shape chart")
ch = gf.chart; chart_style(ch, 10)
ch.has_title = True; ch.chart_title.text_frame.text = "Average kWh per hour in spring and fall, by hour"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 11, color=TXT)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(10)
from pptx.enum.chart import XL_MARKER_STYLE
for ser, col, dash in zip(ch.plots[0].series, [BLUE, NAVY, "7F7F7F"], [4, None, None]):
    ser.format.line.color.rgb = rgb(col); ser.format.line.width = Pt(2.25); ser.smooth = False
    ser.marker.style = XL_MARKER_STYLE.NONE
    if dash: ser.format.line.dash_style = dash
va = ch.value_axis; va.minimum_scale = 0; va.major_gridlines.format.line.color.rgb = rgb("D9D9D9")
va.format.line.fill.background(); va.tick_labels.number_format = "0.0"; va.tick_labels.number_format_is_linked = False
for tag, val in (("c:tickLblSkip", "3"), ("c:tickMarkSkip", "3")):
    e = etree.SubElement(ch.category_axis._element, qn(tag)); e.set("val", val)
b = box(s, 8.35, 5.45, 4.48, 1.2, None, "Heat pump limitation callout", line=NAVY)
tf = b.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.12); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.text = ""; p = tf.paragraphs[0]
r = p.add_run(); r.text = "Limitation: "; style_run(r, 11, bold=True, color=NAVY, italic=True)
r = p.add_run(); r.text = ("in hourly data a whole-home heat pump looks a lot like baseboard heat, so we report 88 confirmed "
                          "heat pumps plus a ranked list of likely ones (about 1 in 10 will be right), not confirmed new finds")
style_run(r, 11, color=NAVY, italic=True)
alt(b, "Limitation: in hourly data a whole-home heat pump looks a lot like baseboard heat, so we report 88 confirmed heat "
       "pumps plus a ranked list of likely ones, about 1 in 10 of which will be right, not confirmed new finds")

from pptx.enum.chart import XL_MARKER_STYLE
exec(open(SCR / "build_extra.py").read())

# reading order: title, subtitle, content, then source notes / footer / slide number; header and decor last
LATE = ("Source notes", "Footer", "Slide number", "Header project name", "Header section label", "Header band")
for sl in prs.slides:
    tree = sl.shapes._spTree
    for nm in LATE:
        for shp in list(sl.shapes):
            if shp.name == nm:
                el = shp._element; tree.remove(el); tree.append(el)
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.core_properties.title = "Reading the Grid, One Home at a Time"
prs.core_properties.language = "en-US"
prs.save(OUT)
print("saved", OUT)
