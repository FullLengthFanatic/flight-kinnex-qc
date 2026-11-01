"""The two explanatory drawings of the Supplementary Note, as inline SVG: how a TSO-TSO molecule forms a mirror
array (schematic_svg) and how one skera false cut breaks two molecules (false_cut_svg). Both are drawn to explain,
not to scale. report.html (reader_report) and the study's report use these same functions."""

# Molecule fates in the slot budget: lost molecules in greys, molecules skera never writes cleanly in amber.
FATE = {"unfilled": "#E9EDF1", "artifact": "#B9C2CB", "downstream": "#7A8691", "unwritten": "#E69F00"}

SVG_STYLE = ("<style>.svglabel{font:600 13px 'IBM Plex Sans',Arial,sans-serif;fill:#1B1F24}.svgsmall{font:12px 'IBM Plex Sans',Arial,sans-serif;fill:#5B636E}"
             ".svgadapter{font:600 11px 'IBM Plex Sans',Arial,sans-serif;fill:#FFFFFF}.svgpanel{font:600 14px 'IBM Plex Sans',Arial,sans-serif;fill:#1B1F24}</style>")


def schematic_svg() -> str:
    """Panel 3a: how a TSO-TSO molecule forms a mirror array (drawn to explain, not to scale)."""
    def mol(x, y, left, right, kind, w=120):
        col = {"proper": "#E2E7EC", "tt": "#F6D3BF"}[kind]
        lab = {"proper": "Read1 ... insert ... TSO", "tt": "TSO ... insert ... TSO"}[kind]
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="3" fill="{col}" stroke="#A7B1BB" stroke-width="1"/>',
               f'<text x="{x + w / 2}" y="{y + 15}" text-anchor="middle" class="svgsmall">{lab}</text>']
        for ax, a in ((x - 16, left), (x + w + 2, right)):
            out.append(f'<rect x="{ax}" y="{y - 2}" width="14" height="26" rx="2" fill="#2B333B"/>'
                       f'<text x="{ax + 7}" y="{y + 15}" text-anchor="middle" class="svgadapter">{a}</text>')
        return "".join(out)
    rows = []
    rows.append('<text x="10" y="22" class="svglabel">Each position PCR puts the lower-index adapter on the Read1 end and the higher-index adapter on the TSO end:</text>')
    rows.append(mol(40, 34, "G", "H", "proper"))
    rows.append('<text x="200" y="50" class="svgsmall">a proper molecule of slot G to H; adapters join only to the same letter, so arrays grow A, B, ..., H, Q</text>')
    rows.append('<text x="10" y="96" class="svglabel">A TSO-TSO molecule is primed by the TSO primer at both ends and gets the higher-index adapter twice:</text>')
    rows.append(mol(40, 108, "H", "H", "tt"))
    rows.append('<text x="200" y="124" class="svgsmall">each H end then takes a slot H to Q molecule, so both halves run to Q</text>')
    y = 168
    x = 40
    seq = [("Q'", "proper"), ("H'", "tt"), ("H", "proper"), ("Q", None)]
    rows.append(f'<text x="10" y="{y - 10}" class="svglabel">The read: a mirror junction (H\' followed by H) with the TSO-TSO molecule at its centre</text>')
    parts = []
    cx = x
    for i, (a, kind) in enumerate(seq):
        parts.append(f'<rect x="{cx}" y="{y}" width="14" height="26" rx="2" fill="#2B333B"/><text x="{cx + 7}" y="{y + 17}" text-anchor="middle" class="svgadapter">{a}</text>')
        cx += 16
        if kind:
            col = "#F6D3BF" if kind == "tt" else "#E2E7EC"
            lab = "TSO-TSO" if kind == "tt" else "proper"
            parts.append(f'<rect x="{cx}" y="{y + 2}" width="150" height="22" rx="3" fill="{col}" stroke="#A7B1BB"/><text x="{cx + 75}" y="{y + 17}" text-anchor="middle" class="svgsmall">{lab}</text>')
            cx += 152
    rows.append("".join(parts))
    head = '<text x="10" y="17" class="svgpanel">a  Mechanism (schematic)</text>'
    return (f'<svg class="schematic" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 240" role="img" aria-label="How a TSO-TSO molecule forms a mirror array">{SVG_STYLE}{head}'
            f'<g transform="translate(0,34)">{"".join(rows)}</g></svg>')


def false_cut_svg(fa: dict) -> str:
    """Supplementary Fig. 1: one skera false cut on a read whose adapters run A to Q (drawn to explain, not to scale).

    fa: the false-cut summary. The lima labels (the heading's "lima rejects both" and the shares lima passes,
    fa["fates"]["fused_lima"] and fa["fates"]["frag_lima"]) are drawn only when fa has "fates": the study's
    false-cut anatomy has them, `kinnex-qc run` does not run lima, so report.html's drawing leaves them out."""
    amber, dark, grey, light = FATE["unwritten"], "#2B333B", "#A7B1BB", "#E2E7EC"
    x0, w, y = 40, 300, 40
    m1, aD, m2, aE = x0 + 16, x0 + 16 + w + 2, x0 + 16 + w + 18, x0 + 16 + 2 * w + 20
    cut = m2 + 0.47 * w

    def adapter(x, letter, yy):
        return (f'<rect x="{x}" y="{yy - 2}" width="14" height="26" rx="2" fill="{dark}"/>'
                f'<text x="{x + 7}" y="{yy + 15}" text-anchor="middle" class="svgadapter">{letter}</text>')

    def molecule(x, slot, yy, at):
        return (f'<rect x="{x}" y="{yy}" width="{w}" height="22" rx="3" fill="{light}" stroke="{grey}"/>'
                f'<rect x="{x + 2}" y="{yy + 2}" width="34" height="18" rx="2" fill="#5E6A76"/><text x="{x + 19}" y="{yy + 15}" text-anchor="middle" class="svgadapter">R1</text>'
                f'<rect x="{x + w - 38}" y="{yy + 2}" width="36" height="18" rx="2" fill="#7A8691"/><text x="{x + w - 20}" y="{yy + 15}" text-anchor="middle" class="svgadapter">TSO</text>'
                f'<text x="{x + at * w}" y="{yy + 15}" text-anchor="middle" class="svgsmall">slot {slot}</text>')

    out = ['<text x="10" y="16" class="svglabel">The read with our adapters and primers; vertical lines: where skera cuts</text>',
           adapter(x0, "C", y), molecule(m1, "C to D", y, 0.5), adapter(aD, "D", y), molecule(m2, "D to E", y, 0.68), adapter(aE, "E", y),
           f'<text x="{aE + 24}" y="{y + 7}" class="svgsmall">R1: Read1 with the cell label and UMI</text>',
           f'<text x="{aE + 24}" y="{y + 23}" class="svgsmall">TSO: template-switching oligo</text>']
    for x, col, dash in ((x0 + 14, dark, ""), (cut, amber, ' stroke-dasharray="4 3"'), (aE, dark, "")):
        out.append(f'<line x1="{x}" y1="{y - 8}" x2="{x}" y2="{y + 32}" stroke="{col}" stroke-width="{2 if col == amber else 1.4}"{dash}/>')
    yl = y + 48
    out += [f'<text x="{x0 + 14}" y="{yl}" text-anchor="middle" class="svgsmall">cut at C</text>',
            f'<text x="{aD + 7}" y="{yl}" text-anchor="middle" class="svgsmall">no cut at the real D</text>',
            f'<text x="{cut}" y="{yl}" text-anchor="middle" class="svgsmall" style="fill:#9A6A00">false D, at a D-like sequence</text>',
            f'<text x="{aE}" y="{yl}" text-anchor="middle" class="svgsmall">cut at E</text>']
    y3 = 132
    heading = "The two S-reads skera writes; lima rejects both" if "fates" in fa else "The two S-reads skera writes"
    out.append(f'<text x="10" y="{y3 - 12}" class="svglabel">{heading}</text>')
    out.append(f'<rect x="{x0 + 14}" y="{y3}" width="{cut - x0 - 15}" height="24" rx="3" fill="#FFFFFF" stroke="{amber}" stroke-width="1.5"/>'
               f'<text x="{(x0 + 14 + cut) / 2}" y="{y3 + 16}" text-anchor="middle" class="svgsmall">S-read 1, chimera: whole C to D molecule, real D, first part of D to E</text>'
               f'<rect x="{cut + 1}" y="{y3}" width="{aE - cut - 1}" height="24" rx="3" fill="#FFFFFF" stroke="{amber}" stroke-width="1.5"/>'
               f'<text x="{(cut + aE) / 2}" y="{y3 + 16}" text-anchor="middle" class="svgsmall">S-read 2, fragment</text>')
    if "fates" in fa:
        out.append(f'<text x="{aE + 24}" y="{y3 + 9}" class="svgsmall">lima passes {fa["fates"]["fused_lima"]:.1%} of chimeras</text>'
                   f'<text x="{aE + 24}" y="{y3 + 25}" class="svgsmall">and {fa["fates"]["frag_lima"]:.1%} of fragments</text>')
    y4 = 196
    out.append(f'<text x="10" y="{y4 - 12}" class="svglabel">Our segmentation keeps both molecules</text>'
               f'<rect x="{m1}" y="{y4}" width="{w}" height="22" rx="3" fill="{light}" stroke="{dark}"/><text x="{m1 + w / 2}" y="{y4 + 15}" text-anchor="middle" class="svgsmall">proper molecule</text>'
               f'<rect x="{m2}" y="{y4}" width="{w}" height="22" rx="3" fill="{light}" stroke="{dark}"/><text x="{m2 + w / 2}" y="{y4 + 15}" text-anchor="middle" class="svgsmall">proper molecule</text>')
    out.append(f'<text x="10" y="{y4 + 48}" class="svgsmall">On reads from the other strand the picture is mirrored: the cut falls in the molecule after the skipped adapter in the direction of the read,'
               f' and the fragment keeps the Read1 end.</text>')
    return (f'<svg class="schematic" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 256" role="img" aria-label="How one skera false cut breaks two molecules">'
            f'{SVG_STYLE}{"".join(out)}</svg>')
