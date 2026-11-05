"""A coloured view of one read: adapters, TSO, Read1 and the site where skera cut, as HTML for a <pre> block.
Shared by report.html and the study's example reads."""

LABEL = {"adapter": "adapter", "tso": "TSO", "r1": "Read1", "false": "site skera called an adapter"}

CSS = """pre{font:13px/1.55 Menlo,Consolas,monospace;overflow-x:auto;background:#FAFBFC;border:1px solid #E2E7EC;padding:12px;border-radius:4px}
.pos{color:#8A939C} .adapter{background:#1B1F24;color:#FFFFFF;font-weight:600}
.tso{background:#9CC3E6;text-decoration:underline} .r1{background:#F4D2DF} .false{background:#F6C445;outline:2px dashed #9A6A00}
.key span{display:inline-block;padding:0 6px;margin-right:10px;font:13px Menlo,Consolas,monospace}"""


def html_sequence(seq: str, features: list[tuple[str, int, int, str]], wrap: int = 100, labels: dict[str, str] | None = None) -> str:
    """seq in lines of `wrap` bases, each led by its 1-based position, with every feature (name, start, end, kind),
    0-based half-open on seq, kind one of LABEL's keys, as a span of that class whose hover title gives the name,
    the kind (worded by labels, default LABEL) and its 1-based inclusive coordinates. Where features overlap, adapters are drawn last, on top."""
    labels = labels or LABEL
    owner = [None] * len(seq)
    for m in sorted(features, key=lambda m: ("r1", "tso", "false", "adapter").index(m[3])):      # adapters drawn last, on top
        for i in range(m[1], m[2]):
            owner[i] = m
    lines = []
    for i in range(0, len(seq), wrap):
        out, j, stop = [], i, min(i + wrap, len(seq))
        while j < stop:
            k = j
            while k < stop and owner[k] is owner[j]:
                k += 1
            m = owner[j]
            out.append(seq[j:k] if m is None else
                       f'<span class="{m[3]}" title="{m[0]} ({labels[m[3]]}), {m[1] + 1}-{m[2]}">{seq[j:k]}</span>')
            j = k
        lines.append(f'<span class="pos">{i + 1:>6}</span>  ' + "".join(out))
    return "\n".join(lines)
