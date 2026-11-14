SNAP = ('Reported findings use the original 34,238-customer analytical snapshot. The later expanded dataset (35,236 customers, after a supplemental delivery) has not been fully recalculated.')
# Executed inside build_deck.py (shares its namespace): title, executive summary, approach, section dividers,
# Part 2 segmentation, Part 3 battery & flexibility, recommendations, limitations, appendix.
import numpy as np
from PIL import Image as _Img

B3 = D["bess"]; T = B3["tiers"]; EL = B3["eligible"]
SEG = {s["segment"]: s for s in D["segments"]}
fmt = lambda x: f"{x:,.0f}"

# ---------- decorative images: cover and divider panel ----------
def _streak_img(w, h, path, light=1.0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u = x / w; v = y / h
    base = np.stack([0x23 + 0 * u, 0x2A + 0 * u, 0x3E + 0 * u], -1)
    t = np.clip((u - v * 0.9 + 0.15), 0, 1) ** 1.6 * light
    hi = np.array([0x8C, 0xC8, 0xF0], np.float32)
    img = base * (1 - t[..., None]) + hi * t[..., None]
    for k, (a, wdt, s) in enumerate([(0.18, 0.035, 0.55), (0.28, 0.02, 0.45), (0.10, 0.05, 0.35)]):
        band = np.exp(-((v - (a + 0.55 * u)) / wdt) ** 2) * s * light
        img = img * (1 - band[..., None]) + np.array([0xE6, 0xF2, 0xFC]) * band[..., None]
    _Img.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
COVER = SCR / "cover.png"; _streak_img(2400, 1350, COVER)
PANEL = SCR / "divider_panel.png"; _streak_img(760, 1350, PANEL, light=0.9)


def title_slide():
    s = prs.slides.add_slide(TITLE_ONLY)
    pic = s.shapes.add_picture(str(COVER), 0, 0, prs.slide_width, prs.slide_height); decorative(pic, "Cover image")
    tree = s.shapes._spTree; tree.remove(pic._element); tree.insert(2, pic._element)   # send image behind text
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(1.0), Inches(2.6), Inches(11.33), Inches(1.6)
    tf = t.text_frame; tf.word_wrap = True; tf.text = ""; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "Reading the Grid, One Home at a Time"; style_run(r, 44, color="FFFFFF", font=SERIF)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r = p2.add_run(); r.text = "Appliance detection, customer segments and a battery & demand-response plan from hourly AMI data"
    style_run(r, 16, color="FFFFFF")
    tb = text(s, 1.0, 4.55, 11.33, 0.4, [{"runs": [("AVANGRID 2026 ENERGY INNOVATION HACKATHON", dict(size=12, color="FFFFFF"))],
                                          "align": PP_ALIGN.CENTER}], "Event name")
    tb.text_frame.paragraphs[0].runs[0].font._element.set("spc", "400")
    text(s, 1.0, 5.0, 11.33, 0.4, [{"runs": [("Team 6  ·  DNH Energy  ·  November 2026", dict(size=13, color="FFFFFF"))],
                                    "align": PP_ALIGN.CENTER}], "Team line")
    s.notes_slide.notes_text_frame.text = ("Introduce the team and the one-line pitch: we used a year of hourly meter data to "
        "figure out what's inside each home, group homes by how they use power, and pick where batteries and demand response "
        "would help the grid most.")
    return s


def divider(numeral, title_txt, notes):
    s = prs.slides.add_slide(TITLE_ONLY)
    pic = s.shapes.add_picture(str(PANEL), 0, 0, Inches(4.2), prs.slide_height); decorative(pic, "Divider panel")
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(4.55), Inches(3.05), Inches(8.3), Inches(1.4)
    tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.text = ""
    for side in ("left", "right", "top", "bottom"): setattr(tf, f"margin_{side}", 0)
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    r = tf.paragraphs[0].add_run(); r.text = title_txt; style_run(r, 40, color=NAVY, font=SERIF)
    n = text(s, 0.3, 2.85, 3.65, 1.8, [{"runs": [(numeral, dict(size=96, color="FFFFFF", font=SERIF))], "align": PP_ALIGN.RIGHT}],
             "Section number", anchor=MSO_ANCHOR.MIDDLE)
    hdr = text(s, 4.55, 0.3, 4.0, 0.3, [{"runs": [("AMI CUSTOMER INSIGHTS", dict(size=10, color=NAVY))]}], "Header project name")
    hdr.text_frame.paragraphs[0].runs[0].font._element.set("spc", "300")
    slide_number(s)
    s.notes_slide.notes_text_frame.text = notes
    return s


def kpi(s, x, y, w, big, label, name, color=NAVY):
    b = box(s, x, y, w, 1.25, PALE, name)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.text = ""
    tf.margin_left = tf.margin_right = Inches(0.1)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = big; style_run(r, 30, color=color, font=SERIF)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER; r = p2.add_run(); r.text = label; style_run(r, 11, color=TXT)
    alt(b, f"{big}: {label}")


new = []
# ===================================================================== 1. Title
new.append(title_slide())

# ===================================================================== 2. Executive summary
t12_mw = T["Tier 1"]["cp_red_mw"] + T["Tier 2"]["cp_red_mw"]
s = frame("Overview", "Executive Summary—What We Found and What Avangrid Should Do",
          f"We turned a year of hourly meter data for 34,238 homes into an appliance inventory, nine customer segments and a "
          f"ranked battery and demand-response program",
          [("Source:", "Team analysis of Avangrid 2026 Hackathon AMI, weather and reference-customer data."),
           ("Note:", "Battery economics use illustrative assumptions (capacity value $200/kW-yr; $14,000 per 2-unit system; "
                     "TOU $0.28/$0.11 per kWh); see the battery slides for details.")],
          "If the judges only read one slide, it's this one. Walk down the five rows: what's in the homes, who drives the "
          "peak, what batteries can and can't do, why demand response comes first, and how batteries help electrification. "
          "The honest headline on batteries is that they work physically but need an incentive or extra value to pay off.")
rows = [["Topic", "What we found"],
        ["Who has what", "844 homes flagged as EV chargers (2.5%), about 724 estimated EV owners (2.1%); two-thirds run AC, "
                         "18% use electric heat. The EV model finds owners 17× better than random; heat pumps come as a ranked list"],
        ["Who drives the peak", f"The system peaks on summer evenings ({B3['system_peak_mw']:.0f} MW). Three segments, 48% of "
                                "homes, cause 66% of it: high-use year-round homes, heavy-AC homes and steady moderate users"],
        ["Batteries", f"A home battery can only cut what that home uses, about 2 kW at peak. Our top 3,000 homes give "
                      f"{t12_mw:.1f} MW of modeled potential, but at about $3,400+ per kW they need an incentive or added value to pay off"],
        ["Demand response", f"Thermostat events in {B3['dr_homes']:,} AC homes show about {B3['dr_total_mw']:.0f} MW of modeled "
                            "potential before enrollment, at far lower cost, so demand response should come first"],
        ["Electrification", f"{EL['l2_ready']:,} preliminary Level 2 candidates pass an energy-headroom screen with a 2-unit "
                            "battery, pending power, service-capacity and feeder checks"]]
table(s, 0.5, 2.42, 12.33, [2.3, 10.03], rows, "Executive summary table",
      "Table: executive summary. " + " ".join(f"{r[0]}: {r[1]}." for r in rows[1:]), size=12, row_h=0.62,
      fills={(i, 0): CAT for i in range(1, len(rows))}, bold_rows=())
b = box(s, 0.5, 6.25, 12.33, 0.45, PALE, "Data snapshot note")
tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Inches(0.12); tf.text = ""
r = tf.paragraphs[0].add_run(); r.text = "Data snapshot: "; style_run(r, 11, bold=True, color=NAVY)
r = tf.paragraphs[0].add_run(); r.text = SNAP; style_run(r, 11, color=NAVY); alt(b, "Data snapshot: " + SNAP)
new.append(s)

# ===================================================================== 3. Approach
s = frame("Overview", "Our Approach—One Pipeline from Raw Meter Data to a Program List",
          "Every step is scripted and repeatable: the same code that built these slides can rerun on a new year of data or a "
          "million more customers",
          [("Source:", "Team GitHub repository (scripts src/01–14; README explains setup and run order).")],
          "This slide is the map for the rest of the talk. Five steps, each with its own section. Mention that everything "
          "runs from scripts, so Avangrid could rerun it next year, and that it's built to scale.")
flow = [("1 · Clean", "298M readings → 34,238 homes"), ("2 · Describe", "65 features per home"),
        ("3 · Classify", "ML + physics rules"), ("4 · Segment", "6 clusters + 3 groups"), ("5 · Prioritize", "battery & DR simulation")]
cw = 2.52
for i, (a, b2) in enumerate(flow):
    c = box(s, 0.5 + i * (cw - 0.06), 2.45, cw, 0.85, NAVY if i % 2 == 0 else BLUE, f"Approach step {i+1}",
            shape=MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON)
    tf = c.text_frame; tf.text = ""; tf.margin_left = Inches(0.32 if i else 0.15); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = a; style_run(r, 13, True, "FFFFFF")
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER; r = p2.add_run(); r.text = b2; style_run(r, 10, color="FFFFFF")
    alt(c, f"Step {a}: {b2}")
cols = [("What goes in", ["Hourly AMI readings for 35,725 meter IDs (Sep 2025–Aug 2026)",
                          "Hourly temperature, humidity and wind", "Reference lists: 200 EV, 100 heat-pump, 434 solar homes"]),
        ("What comes out", ["Appliance flags and probabilities for every home", "A load segment for every home",
                            "Battery size, peak relief, tier and DR potential per home"]),
        ("Built to scale", ["Full run takes minutes on a laptop", "Processes data in chunks, never all in memory",
                            "Same steps map onto Spark or Dask for millions of meters"])]
for j, (h, items) in enumerate(cols):
    x = 0.5 + j * 4.22
    panel_header(s, x, 3.7, 3.9, h)
    text(s, x, 4.18, 3.9, 2.3, [{"runs": [(it, dict(size=13))], "bullet": True, "space_after": 6} for it in items], f"{h} list")
new.append(s)

# ===================================================================== 4. Divider I
new.append(divider("I", "Data, Methodology & Customer Classification",
                   "Section one: how we cleaned the data, what we measured and how we figured out what's inside each home."))

# ===================================================================== Divider II + segmentation
new.append(divider("II", "Customer Segmentation & Load Insights",
                   "Section two: grouping homes by how they use power, and what each group means for Avangrid."))

order6 = ["High-use, year-round", "Summer-dominant heavy AC", "Steady moderate use", "Small homes, evening-peaky AC",
          "Winter-peaking electric heat", "Low-use, little AC"]
s = frame("Part 2 · Segmentation", "Customer Segments—Six Everyday Load Patterns Plus Three Special Groups",
          "We grouped homes by the shape, seasonality and size of their load with k-means clustering; the curves show each "
          "segment's typical summer day",
          [("Source:", "Team analysis; k-means on daily shape (6 PCA components), monthly shape (3) and six size/behavior features."),
           ("Note:", "Silhouette was low for every k from 4 to 9 (home load is a continuum), so k = 6 was chosen for clear, actionable "
                     "groups. Seasonal homes, large non-residential-like meters and short-data meters are assigned by rule first.")],
          "Each line is a segment's average summer day. The two big ones at the top, high-use and heavy-AC homes, climb hard "
          "into the evening right when the grid peaks. The flat bottom line is the low-use group. We pulled seasonal homes, "
          "business-sized meters and short-data meters out first so they wouldn't distort the clusters.")
cd = CategoryChartData(); cd.categories = [str(h) for h in range(1, 25)]
for sg in order6: cd.add_series(sg, D["seg_summer"][sg])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.5), Inches(2.42), Inches(7.3), Inches(4.25), cd)
gf.name = "Segment summer profiles chart"
alt(gf, "Line chart of the average summer day (kW by hour) for six segments. High-use year-round and summer-dominant heavy AC "
        "rise to about 2 to 2.5 kW in the early evening; steady moderate and evening-peaky AC homes rise to about 1.3 to 1.5 kW; "
        "winter-peaking electric heat is moderate; low-use homes stay under about 0.7 kW.", "Segment summer profiles chart")
ch = gf.chart; chart_style(ch, 10)
ch.has_title = True; ch.chart_title.text_frame.text = "Average summer day, kW by hour"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 12, color=TXT)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10)
palette = [(NAVY, None), (BLUE, None), (GOLD, None), ("6FA8DC", 4), ("7F7F7F", 4), ("B7B7B7", 4)]
for ser, (col, dash) in zip(ch.plots[0].series, palette):
    ser.format.line.color.rgb = rgb(col); ser.format.line.width = Pt(2.25); ser.smooth = False
    ser.marker.style = XL_MARKER_STYLE.NONE
    if dash: ser.format.line.dash_style = dash
va = ch.value_axis; va.minimum_scale = 0; va.major_gridlines.format.line.color.rgb = rgb("D9D9D9"); va.format.line.fill.background()
va.tick_labels.number_format = "0.0"; va.tick_labels.number_format_is_linked = False
for tag, val in (("c:tickLblSkip", "3"), ("c:tickMarkSkip", "3")):
    e = etree.SubElement(ch.category_axis._element, qn(tag)); e.set("val", val)
panel_header(s, 8.15, 2.4, 4.68, "Segment size and peak draw")
rows = [["Segment", "Homes", "kWh/yr", "kW at peak"]]
for sg in order6 + ["Seasonal / vacant home", "Large non-residential-like", "Insufficient data"]:
    v = SEG[sg]; rows.append([sg, f"{v['n']:,}", fmt(v["annual_kwh"]), f"{v['coinc_peak_kw']:.1f}"])
fl = {(i, 0): CAT for i in range(1, len(rows))}
table(s, 8.15, 2.88, 4.68, [2.48, 0.7, 0.75, 0.75], rows, "Segment table",
      "Table: segment size and draw at system peak. " + "; ".join(f"{r[0]} {r[1]} homes, {r[2]} kWh per year, {r[3]} kW at peak" for r in rows[1:]),
      size=10.5, row_h=0.37, fills=fl, aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
new.append(s)

s = frame("Part 2 · Segmentation", "Peak Drivers—Three Segments Cause Two-Thirds of the Summer Peak",
          "High-use, heavy-AC and steady moderate homes are 48% of customers but 66% of the load during the system's 100 "
          "busiest hours; seasonal and low-use homes barely matter at peak",
          [("Source:", "Team analysis; share of peak = segment's total kW during the system's top-100 hours ÷ all meters' total."),
           ("Note:", "Technology shares use the Part 1 flags; electric heat includes heat pumps and resistance heat.")],
          "The gray bars are each segment's share of customers; the blue bars are its share of the peak. Wherever blue is "
          "longer than gray, that segment punches above its weight at peak, and that's where programs pay off. The table on "
          "the right shows what's inside each segment, which tells us which program fits.")
segs9 = sorted([s_ for s_ in SEG.values()], key=lambda v: v["share_of_system_peak_pct"])
cd = CategoryChartData(); cd.categories = [v["segment"] for v in segs9]
cd.add_series("% of customers", [round(v["share_pct"], 1) for v in segs9])
cd.add_series("% of peak load", [round(v["share_of_system_peak_pct"], 1) for v in segs9])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.5), Inches(2.42), Inches(6.9), Inches(4.25), cd)
gf.name = "Peak share chart"
alt(gf, "Clustered bar chart comparing each segment's share of customers with its share of system-peak load. " +
    "; ".join(f"{v['segment']}: {v['share_pct']:.0f}% of customers, {v['share_of_system_peak_pct']:.0f}% of peak" for v in segs9[::-1]),
    "Peak share chart")
ch = gf.chart; chart_style(ch, 10)
ch.has_title = True; ch.chart_title.text_frame.text = "Share of customers vs. share of system-peak load (%)"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 12, color=TXT)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -5
pl.has_data_labels = True; pl.data_labels.number_format = "0"; pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; pl.data_labels.font.size = Pt(9)
for ser, col in zip(pl.series, ["A6A6A6", BLUE]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False; ch.category_axis.tick_labels.font.size = Pt(10)
panel_header(s, 7.75, 2.4, 5.08, "What's inside each segment (% of homes)")
rows = [["Segment", "EV", "AC", "Elec. heat", "Pool"]]
for sg in order6:
    v = SEG[sg]; rows.append([sg, f"{v['ev']*100:.0f}%", f"{v['ac']*100:.0f}%", f"{v['elec_heat']*100:.0f}%", f"{v['pool']*100:.0f}%"])
table(s, 7.75, 2.88, 5.08, [2.48, 0.6, 0.6, 0.8, 0.6], rows, "Segment technology table",
      "Table: technology share by segment. " + "; ".join(f"{r[0]}: EV {r[1]}, AC {r[2]}, electric heat {r[3]}, pool {r[4]}" for r in rows[1:]),
      size=10.5, row_h=0.36, fills={(i, 0): CAT for i in range(1, len(rows))},
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4)
b = box(s, 7.75, 5.6, 5.08, 0.95, None, "Peak insight callout", line=NAVY)
tf = b.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.12); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.text = ""; r = tf.paragraphs[0].add_run()
r.text = ("Electric-heat homes add little to today's summer peak (9%), but they are where a future winter peak will "
          "grow as heating electrifies")
style_run(r, 12, color=NAVY, italic=True); tf.paragraphs[0].alignment = PP_ALIGN.CENTER; alt(b, r.text)
new.append(s)

s = frame("Part 2 · Segmentation", "Segment Playbook—Matching Each Group to the Right Utility Program",
          "Each segment points to a different tool, from batteries and thermostat events to managed EV charging and "
          "heat-pump outreach",
          [("Source:", "Team analysis; segment statistics from slides above."),
           ("Note:", "Recommendations are starting points for program design; enrollment and costs would be tested in a pilot.")],
          "This is the 'so what' of segmentation. Go row by row and tie each segment to one program. The big two for the "
          "summer peak are high-use homes, which are our battery targets, and heavy-AC homes, which are perfect for "
          "thermostat events. Steady moderate homes have the most EVs, so managed charging fits them.")
rows = [["Segment", "What they look like", "Best-fit program", "Why"],
        ["High-use, year-round", "Big, steady evening load; 20% electric heat, 15% hot tubs", "Battery program (main Tier 1 target)",
         "Highest kW at peak, every day"],
        ["Summer-dominant heavy AC", "96% AC; 3.0 kW at peak; 10% pools", "Smart-thermostat events, pool-pump timing",
         "Cheap kW on hot evenings"],
        ["Steady moderate use", "Flat across seasons; most EVs (6%)", "Managed EV charging, time-of-use rates",
         "Shift charging past 9 pm"],
        ["Small homes, evening-peaky AC", "Sharp 5–8 pm rise; 92% AC", "Thermostat events, peak-time alerts",
         "Many homes, small kW each"],
        ["Winter-peaking electric heat", "3× more use in winter; 76% electric heat", "Heat-pump upgrades with winter flexibility",
         "Future winter peak driver"],
        ["Low-use, little AC", "0.8 kW at peak", "Efficiency and outreach only", "Little peak to save"],
        ["Seasonal / vacant", "Empty two months or more", "Leave out of programs; forecast separately", "Load isn't there"],
        ["Large non-residential-like", "Business-sized meters (64)", "Review rate class", "Not a home battery fit"]]
table(s, 0.5, 2.42, 12.33, [2.55, 3.65, 3.55, 2.58], rows, "Segment playbook table",
      "Table: segment playbook. " + " ".join(f"{r[0]}: {r[1]}; program: {r[2]}; why: {r[3]}." for r in rows[1:]),
      size=11, row_h=0.44, fills={(i, 0): CAT for i in range(1, len(rows))})
new.append(s)

# ===================================================================== Divider III + battery
new.append(divider("III", "Demand Flexibility & Battery Opportunity",
                   "Section three: which homes should get a battery or a thermostat program, how big, and what it's worth."))

s = frame("Part 3 · Battery & Flexibility", "Battery Prioritization—Rank Homes by the Cost of Each kW of Peak Relief",
          "We simulated a battery in every eligible home, then ranked homes by how cheaply each one cuts load during the "
          "system's 100 busiest hours",
          [("Source:", "Team simulation (src/14_bess_opportunity.py) on 34,238 customers; GridFlex optimizer in the repo can refine "
                       "dispatch for short-listed homes."),
           ("Note:", "Persistent peak = home draws ≥ 2 kW in at least 40% of the system's top-100 hours. Battery cost $7,500 "
                     "(1 unit) / $14,000 (2 units); 15 years at 7%.")],
          "The funnel on the left is how we narrow down. We start with every meter, remove businesses, empty homes and bad "
          "data, keep homes that are reliably busy at peak, and rank them. The ranking metric is simple: how much incentive "
          "the utility would need to pay per kW of peak relief, after the customer keeps their bill savings. Lowest first.")
funnel = [(f"{34238:,} meters", "all customers in the data", 6.6), (f"{EL['homes']:,} eligible homes",
           f"minus {B3['excluded']:,} business-sized, seasonal or short-data meters", 5.9),
          (f"{T['Tier 1']['homes'] + T['Tier 2']['homes'] + T['Tier 3']['homes']:,} persistent-peak homes",
           "busy at the system peak, not just once in a while", 5.2),
          (f"{T['Tier 1']['homes'] + T['Tier 2']['homes']:,} Tiers 1–2", "lowest cost per kW of relief", 4.5),
          (f"{T['Tier 1']['homes']:,} Tier 1", "staged pilot: start with ~100 monitored homes", 3.8)]
for i, (a, b2, w) in enumerate(funnel):
    x = 0.5 + (6.6 - w) / 2; y = 2.45 + i * 0.84
    c = box(s, x, y, w, 0.74, NAVY if i % 2 == 0 else BLUE, f"Funnel step {i+1}")
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.text = ""
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = a; style_run(r, 13, True, "FFFFFF")
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER; r = p2.add_run(); r.text = b2; style_run(r, 10, color="FFFFFF")
    alt(c, f"{a}: {b2}")
panel_header(s, 7.5, 2.4, 5.33, "How we rank")
b = box(s, 7.5, 2.9, 5.33, 0.9, PALE, "Ranking equation")
tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Inches(0.12); tf.text = ""
r = tf.paragraphs[0].add_run(); r.text = "Cost per kW = (battery cost − PV of bill savings) ÷ kW cut at system peak"
style_run(r, 13, color=TXT, font="Cambria"); alt(b, "Equation: " + r.text)
text(s, 7.5, 3.95, 5.33, 2.6,
     [{"runs": [("Eligible: ", dict(size=12, bold=True)), ("homes only; business-sized, seasonal and short-data meters "
                 "are excluded (17 of GridFlex's 30 first picks were business-sized meters)", dict(size=12))], "bullet": True, "space_after": 5},
      {"runs": [("Persistent: ", dict(size=12, bold=True)), ("reliably busy at peak, so the battery's kW show up when the "
                 "grid needs them", dict(size=12))], "bullet": True, "space_after": 5},
      {"runs": [("Ranked: ", dict(size=12, bold=True)), ("lowest cost per kW first; Tier 1 = best 1,000, Tier 2 = next 2,000, "
                 "Tier 3 = other persistent-peak homes", dict(size=12))], "bullet": True, "space_after": 5},
      {"runs": [("Benchmark: ", dict(size=12, bold=True)), (f"capacity worth ${B3['cap_pv_per_kw']:,.0f} per kW over 15 years "
                 "at the assumed $200/kW-yr", dict(size=12))], "bullet": True}], "Ranking rules")
new.append(s)

pk = B3["peak_day"]
# recharge by valley-filling midnight-7 am (charge brings those hours up to one flat level)
_need = sum(pk["discharge"]) / 0.894; _night = pk["load"][:7]
_lo, _hi = min(_night), max(_night) + _need
for _ in range(60):
    _lv = (_lo + _hi) / 2
    if sum(max(0, _lv - x) for x in _night) > _need: _hi = _lv
    else: _lo = _lv
chg = [max(0, _lv - x) if h < 7 else 0 for h, x in enumerate(pk["load"])]
net = [round(l - d_ + c_, 3) for l, d_, c_ in zip(pk["load"], pk["discharge"], chg)]
s = frame("Part 3 · Battery & Flexibility", "Sizing and Dispatch—What a Battery Does on the Hottest Evening",
          "On the system's peak day, the average Tier 1 home drops from about 5.8 kW to about 2.4 kW in the early evening, "
          "and refills its battery after midnight when the grid is quiet",
          [("Source:", f"Team simulation; average of the 1,000 Tier 1 homes on {pk['date']}, the system's peak day."),
           ("Note:", "Battery keeps a 10% backup reserve; round-trip efficiency 89.4% split evenly between charge and discharge; "
                     "no export to the grid.")],
          "The dark line is what the average Tier 1 home normally pulls from the grid on the hottest day. The blue dashed line "
          "is what it pulls with a battery: the evening hump is cut by more than half, and the battery refills after "
          "midnight, filling in the overnight dip. On the right is how we size: if a home's evening use on peak days fits in one battery unit, we recommend "
          "one; otherwise two.")
cd = CategoryChartData(); cd.categories = [str(h) for h in range(1, 25)]
cd.add_series("Home load without battery (kW)", pk["load"]); cd.add_series("Grid draw with battery (kW)", net)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.5), Inches(2.42), Inches(7.3), Inches(4.25), cd)
gf.name = "Peak day dispatch chart"
alt(gf, f"Line chart for the average Tier 1 home on {pk['date']}. Without a battery, load rises from about 2.7 kW at dawn to "
        f"{max(pk['load']):.1f} kW around 5 pm. With the battery, grid draw drops to about {min(net[15:21]):.1f} to "
        f"{max(net[15:21]):.1f} kW between 3 and 9 pm and rises slightly after midnight while the battery recharges.",
    "Peak day dispatch chart")
ch = gf.chart; chart_style(ch, 10)
ch.has_title = True; ch.chart_title.text_frame.text = "Average Tier 1 home on the system peak day, by hour"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 12, color=TXT)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
for ser, col, dash in zip(ch.plots[0].series, [NAVY, BLUE], [None, 4]):
    ser.format.line.color.rgb = rgb(col); ser.format.line.width = Pt(2.5); ser.smooth = False
    ser.marker.style = XL_MARKER_STYLE.NONE
    if dash: ser.format.line.dash_style = dash
va = ch.value_axis; va.minimum_scale = 0; va.major_gridlines.format.line.color.rgb = rgb("D9D9D9"); va.format.line.fill.background()
va.tick_labels.number_format = "0"; va.tick_labels.number_format_is_linked = False
for tag, val in (("c:tickLblSkip", "3"), ("c:tickMarkSkip", "3")):
    e = etree.SubElement(ch.category_axis._element, qn(tag)); e.set("val", val)
panel_header(s, 8.15, 2.4, 4.68, "How we size each battery")
table(s, 8.15, 2.88, 4.68, [1.6, 1.5, 1.58],
      [["Size", "1 unit", "2 units"], ["Energy / power", "10.08 kWh / 3.84 kW", "20.16 kWh / 7.68 kW"],
       ["Usable per day", "8.6 kWh", "17.2 kWh"], ["Recommended for", f"{B3['units_sizing']['1']:,} homes", f"{B3['units_sizing']['2']:,} homes"],
       ["Tier 1 homes", f"{T['Tier 1']['one_unit']:,}", f"{T['Tier 1']['two_unit']:,}"]],
      "Battery sizing table", f"Table: battery sizes. One unit is 10.08 kWh and 3.84 kW with 8.6 kWh usable per day, recommended "
      f"for {B3['units_sizing']['1']:,} homes. Two units are 20.16 kWh and 7.68 kW with 17.2 kWh usable per day, recommended for "
      f"{B3['units_sizing']['2']:,} homes, including all {T['Tier 1']['two_unit']:,} Tier 1 homes.", size=11, row_h=0.36,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
text(s, 8.15, 4.85, 4.68, 1.75,
     [{"runs": [("Sizing rule: ", dict(size=12, bold=True)), ("one unit if the home's 3–9 pm use on peak days fits in "
                 "8.6 kWh, otherwise two", dict(size=12))], "bullet": True, "space_after": 5},
      {"runs": [("Discharge: ", dict(size=12, bold=True)), ("3–9 pm, biggest hours first, never more than the home uses", dict(size=12))],
       "bullet": True, "space_after": 5},
      {"runs": [("Recharge: ", dict(size=12, bold=True)), ("midnight to 7 am, filling the overnight dip, or from rooftop solar", dict(size=12))],
       "bullet": True}], "Dispatch rules")
new.append(s)

t1, t2, t3 = T["Tier 1"], T["Tier 2"], T["Tier 3"]
mwh12 = t1["mwh_shifted"] + t2["mwh_shifted"]
bill12 = (t1["bill_avg"] * t1["homes"] + t2["bill_avg"] * t2["homes"]) / (t1["homes"] + t2["homes"])
s = frame("Part 3 · Battery & Flexibility", f"Program Impact—Top 3,000 Homes Show {t12_mw:.1f} MW of Modeled Peak Relief",
          "Modeled technical potential, before enrollment, participation and feeder checks: about 2 kW per home at peak and "
          "$670–840 a year in bill savings, but each kW costs more than today's capacity value",
          [("Source:", "Team simulation on 34,238 customers (src/14_bess_opportunity.py)."),
           ("Note:", "Illustrative: capacity value $200/kW-yr; TOU $0.28 on-peak (3–9 pm) / $0.11 off-peak per kWh; battery "
                     "$7,500 / $14,000; O&M $100/yr; 15 years at 7%. Incentive = battery cost − PV of net bill savings. MW are modeled "
                     "technical potential, not firm capacity.")],
          f"Start with the four big numbers, and be clear they're modeled potential, before we know who would enroll. The top 3,000 homes could cut {t12_mw:.1f} megawatts at peak and move about "
          f"{mwh12/1000:.0f} gigawatt-hours a year out of the evening. Customers save around $700 a year on a time-of-use "
          "rate. Now the honest part: even the best homes cost about $3,400 per kW, while the capacity is worth about $1,800 "
          "per kW over 15 years at our assumed value. Break-even battery cost for Tier 1 is about $10,600, so either costs "
          "come down, Avangrid pays an incentive, or we count extra value like backup power or a deferred upgrade.")
kpis = [(f"{t12_mw:.1f} MW", "modeled peak relief, Tiers 1–2"), (f"{mwh12/1000:.1f} GWh", "shifted out of 3–9 pm each year"),
        (f"${bill12:,.0f}", "average yearly bill savings"), (f"${t1['cost_per_kw_med']:,.0f}", "per kW for Tier 1 (median)")]
for i, (big, lab) in enumerate(kpis):
    kpi(s, 0.5 + i * 3.1, 2.42, 2.93, big, lab, f"KPI {i+1}")
rows = [["Tier", "Homes", "kW cut per home", "Peak relief", "MWh shifted / yr", "Bill savings / yr", "Incentive needed",
         "Cost per kW", "Break-even battery cost"]]
for nm, v in [("Tier 1 (pilot)", t1), ("Tier 2", t2), ("Tier 3", t3)]:
    rows.append([nm, f"{v['homes']:,}", f"{v['cp_red_kw_avg']:.1f} of {v['cp_before_kw_avg']:.1f}", f"{v['cp_red_mw']:.1f} MW",
                 fmt(v["mwh_shifted"]), f"${v['bill_avg']:,.0f}", f"${v['incentive_avg']:,.0f}", f"${v['cost_per_kw_med']:,.0f}",
                 f"${v['breakeven_med']:,.0f}"])
all3 = {k: t1[k] + t2[k] + t3[k] for k in ["homes", "cp_red_mw", "mwh_shifted"]}
rows.append(["Tiers 1–3", f"{all3['homes']:,}", "–", f"{all3['cp_red_mw']:.1f} MW", fmt(all3["mwh_shifted"]), "–", "–", "–", "–"])
fills = {(i, 0): CAT for i in range(1, len(rows))}
for j in range(len(rows[0])): fills[(len(rows) - 1, j)] = LBLUE
table(s, 0.5, 3.95, 12.33, [1.6, 0.95, 1.45, 1.15, 1.5, 1.4, 1.45, 1.2, 1.63], rows, "Program impact table",
      "Table: program impact by tier. " + " ".join(f"{r[0]}: {r[1]} homes, {r[2]} kW cut per home, {r[3]} peak relief, {r[4]} MWh "
      f"shifted per year, bill savings {r[5]}, incentive {r[6]}, cost per kW {r[7]}, break-even battery cost {r[8]}." for r in rows[1:]),
      size=11, row_h=0.42, fills=fills, bold_rows=(len(rows) - 1,), aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 8)
text(s, 0.5, 6.18, 12.33, 0.4, [{"runs": [("Capacity benchmark: ", dict(size=12, bold=True, color=NAVY)),
     (f"about ${B3['cap_pv_per_kw']:,.0f} per kW over 15 years. Batteries close that gap through lower costs, incentives, "
      "backup-power value or targeting constrained feeders.", dict(size=12))]}], "Benchmark line")
new.append(s)

s = frame("Part 3 · Battery & Flexibility", "Beyond Batteries—Thermostats, Electrification Headroom and Solar",
          "The cheapest flexibility is already installed in most homes; batteries matter most where they unlock new "
          "electric load or pair with rooftop solar",
          [("Source:", "Team simulation on 34,238 customers."),
           ("Note:", "Thermostat event = 30% of each AC home's cooling load during the system's top-100 hours. Level 2 check: a "
                     "7.2 kW charger exceeds the 3 kVA guideline by 4.2 kW; the battery must cover that for 2 hours (8.4 kWh) "
                     "after its own peak duty. Solar day = 3+ midday hours with zero grid imports.")],
          "Three opportunities beyond the battery program. First, smart-thermostat events: tiny per home, but there are 22,000 "
          "AC homes, so it adds up to about 10 megawatts of modeled potential with no new hardware, before we know who enrolls. "
          "Second, electrification: the 3 kVA rule limits what a home can add. These 12,445 homes pass a first energy screen, "
          "but each would still need a check of battery power, service capacity and the feeder. Third, solar homes "
          "could charge their battery from their own panels on about half of days.")
cards = [(f"{B3['dr_total_mw']:.1f} MW", "modeled thermostat potential",
          [f"{B3['dr_homes']:,} AC homes × about {B3['dr_avg_kw']:.2f} kW each", "Before enrollment and participation rates",
           "No new hardware; best fit: heavy-AC and evening-peaky homes"]),
         (f"{EL['l2_ready']:,}", "preliminary Level 2 candidates",
          ["Pass an energy-headroom screen with a 2-unit battery", "Still need checks on battery power, service capacity, "
           "charging schedule and feeder", "Pair EV rebates with battery or managed charging"]),
         (f"{B3['solar_days_avg']*100:.0f}%", "of days solar homes could self-charge",
          [f"{B3['solar_pv_homes']:,} eligible solar homes; 4 already have batteries", "Zero midday imports signal spare solar",
           "Solar + battery cuts evening peak without grid charging"])]
for j, (big, lab, items) in enumerate(cards):
    x = 0.5 + j * 4.22
    kpi(s, x, 2.42, 3.9, big, lab, f"Opportunity {j+1}")
    text(s, x, 3.85, 3.9, 2.7, [{"runs": [(it, dict(size=12))], "bullet": True, "space_after": 6} for it in items], f"Opportunity {j+1} detail")
new.append(s)

# ===================================================================== Divider IV + recommendations
new.append(divider("IV", "Recommendations & Next Steps",
                   "Section four: what Avangrid should do, in what order, and how this could get even better."))

s = frame("Recommendations", "Recommendations—Start with Thermostats, Then Pilot Batteries",
          "A phased plan that captures the cheap megawatts first, proves battery value in a targeted pilot, and uses the "
          "inventory to plan for electrification",
          [("Source:", "Team analysis."), ("Note:", "MW figures are modeled technical potential during the system's top-100 hours, "
                                                    "before enrollment, participation and feeder adjustments.")],
          "Four recommendations, in order. One, launch thermostat events, the cheapest source of flexibility. Two, stage a battery "
          "pilot: start with about 100 monitored Tier 1 homes, check the real kW against our simulation, then scale toward "
          "1,000 and test how big an incentive it takes. Three, use the "
          "EV and heat-pump lists to pair electrification with batteries or managed charging. Four, feed the segments into "
          "load forecasting, especially the growing winter peak from electric heat.")
recs = [("Launch smart-thermostat events", f"Heavy-AC and evening-peaky homes; about {B3['dr_total_mw']:.0f} MW of modeled potential"),
        ("Stage a battery pilot", f"Start with ~100 monitored Tier 1 homes, then scale toward 1,000 ({t1['cp_red_mw']:.1f} MW "
                                  f"modeled) if measured kW match"),
        ("Pair electrification with flexibility", f"Offer batteries or managed charging to the {EL['l2_ready']:,} preliminary "
                                                  f"Level 2 candidates after site checks"),
        ("Feed segments into forecasting", "Track EV growth and the future winter peak from electric heat by segment")]
panel_header(s, 0.5, 2.4, 6.6, "What we recommend")
for i, (h, d_) in enumerate(recs):
    yy = 2.95 + i * 0.9
    c = box(s, 0.55, yy + 0.06, 0.46, 0.46, NAVY, f"Recommendation {i+1} marker", shape=MSO_SHAPE.OVAL)
    c.text_frame.text = ""; r = c.text_frame.paragraphs[0].add_run(); r.text = str(i + 1)
    style_run(r, 14, bold=True, color="FFFFFF"); c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER; alt(c, f"Recommendation {i+1}")
    text(s, 1.2, yy, 5.9, 0.82, [{"runs": [(h, dict(size=14, bold=True, color=NAVY))], "space_after": 1},
                                 {"runs": [(d_, dict(size=12))]}], f"Recommendation {i+1}: {h}")
panel_header(s, 7.5, 2.4, 5.33, "Roadmap")
phases = [("Next 6 months", ["Enroll thermostat customers in heavy-AC segments", "Install ~100 monitored Tier 1 batteries",
                             "Confirm heat pumps with rebate records"]),
          ("Year 1", ["Compare measured kW at peak with the simulation", "Scale toward 1,000 batteries if it matches",
                      "Test incentive levels and backup-power offers"]),
          ("Year 2+", ["Scale to Tier 2 if the pilot hits its cost target", "Add feeder data to target constrained areas",
                       "Rerun the pipeline each year as homes electrify"])]
for i, (h, items) in enumerate(phases):
    yy = 2.9 + i * 1.22
    c = box(s, 7.5, yy, 1.45, 1.1, NAVY if i % 2 == 0 else BLUE, f"Phase {i+1}")
    c.text_frame.text = ""; c.text_frame.word_wrap = True; c.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = c.text_frame.paragraphs[0].add_run(); r.text = h; style_run(r, 13, True, "FFFFFF")
    c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER; alt(c, h)
    text(s, 9.1, yy, 3.73, 1.1, [{"runs": [(it, dict(size=11))], "bullet": True, "space_after": 2} for it in items],
         f"Phase {i+1} actions", anchor=MSO_ANCHOR.MIDDLE)
new.append(s)

s = frame("Recommendations", "Limitations and Next Steps—What Would Make This Even Better",
          "We were upfront about what hourly data can't tell us; each limitation has a clear fix with more data",
          [("Source:", "Team analysis.")],
          "Be honest here; judges like it. The biggest limits are hourly data, which blurs short appliance cycles; partial "
          "labels; one weather station; and illustrative costs. Each has a fix on the right. End with scalability: the "
          "pipeline is built in chunks, so it can run on millions of meters.")
panel_header(s, 0.5, 2.4, 6.0, "Limitations")
lims = ["Hourly data blurs short cycles, so heat pumps look like baseboard heat",
        "Only ~25% of EV and heat-pump owners were labeled",
        "One weather station for the whole territory",
        "Meters show grid imports only, so solar output is inferred from zeros",
        "Battery costs and capacity value are illustrative, not tariffs",
        "Results use the 34,238-customer snapshot; the expanded 35,236 set isn't fully rerun",
        "MW results are modeled potential, not enrolled or firm capacity"]
text(s, 0.5, 2.9, 6.0, 3.6, [{"runs": [(l, dict(size=13))], "bullet": True, "space_after": 6} for l in lims], "Limitations list")
panel_header(s, 6.83, 2.4, 6.0, "What would improve it")
fixes = ["15-minute or faster data to separate appliance cycles",
         "Rebate, permit and program records to confirm heat pumps and EVs",
         "Feeder and transformer mapping to value relief by location",
         "Run the GridFlex optimizer on the full Tier 1–2 list",
         "Actual tariffs, incentive levels and installed battery costs"]
text(s, 6.83, 2.9, 6.0, 2.6, [{"runs": [(f, dict(size=13))], "bullet": True, "space_after": 6} for f in fixes], "Improvements list")
b = box(s, 6.83, 5.6, 6.0, 0.95, None, "Scalability callout", line=NAVY)
tf = b.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.12); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.text = ""; r = tf.paragraphs[0].add_run(); r.text = "Scales to millions: "; style_run(r, 12, bold=True, color=NAVY, italic=True)
r2 = tf.paragraphs[0].add_run(); r2.text = ("the pipeline runs in chunks with constant memory per customer and maps "
                                           "directly onto Spark or Dask")
style_run(r2, 12, color=NAVY, italic=True); alt(b, "Scales to millions: " + r2.text)
new.append(s)

s = frame("Appendix", "Appendix—Reproducibility and Where to Find Everything",
          "All code, documentation and the requirements file are in the team's GitHub repository; the hackathon data "
          "itself is never committed",
          [("Source:", "github.com/heidzehu/avangrid-hackathon-2026 (private to the team and judges).")],
          "Point judges to the repository and the README. Every number in this deck comes from these scripts.")
rows = [["Step", "Script(s)", "What it produces"],
        ["Load and clean", "src/01_ingest.py, 03_matrix.py", "Clean customer × hour matrix, weather on the same grid"],
        ["Describe homes", "src/04_features.py, 04b_session_features.py", "65 features per home"],
        ["Classify", "src/06_classify_ml.py, 07_classify_rules.py", "Appliance flags and probabilities per home"],
        ["Check the models", "src/06b_model_comparison.py, 06c_hyperparameter_search.py", "Model comparison, nested-CV tuning"],
        ["Segment", "src/09_segment.py", "A segment for every home"],
        ["Battery & DR screening", "src/14_bess_opportunity.py", "Battery size, peak relief, tier, DR potential per home"],
        ["Detailed dispatch", "src/10–13 (GridFlex)", "Day-ahead battery optimization, sizing and sensitivity"],
        ["Setup", "README.md, requirements.txt", "How to install and run, in order"]]
table(s, 0.5, 2.42, 12.33, [2.3, 4.6, 5.43], rows, "Reproducibility table",
      "Table: scripts and outputs. " + " ".join(f"{r[0]}: {r[1]} produces {r[2]}." for r in rows[1:]), size=12, row_h=0.45,
      fills={(i, 0): CAT for i in range(1, len(rows))})
new.append(s)

# ---------- final slide order ----------
sld = prs.slides._sldIdLst
ids = list(sld)
orig, extra = ids[:7], ids[7:]
title_, exec_, appr, d1, d2, sg1, sg2, sg3, d3, b1, b2, b3, b4, d4, rec, lim, apx = extra
final = [title_, exec_, appr, d1] + orig + [d2, sg1, sg2, sg3, d3, b1, b2, b3, b4, d4, rec, lim, apx]
for e in ids: sld.remove(e)
for e in final: sld.append(e)
