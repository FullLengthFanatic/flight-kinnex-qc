"""Numbers as a reader sees them in the Supplementary Note, for the golden regression."""
import re
from html import unescape

NUM = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


def visible_text(html: str) -> str:
    """Report text as a reader sees it: inlined plotly bundle, plot divs, scripts and styles removed, tags stripped."""
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    html = re.sub(r'<div class="plotwrap">.*?</div>\s*</div>', " ", html, flags=re.S)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))


def numbers(text: str) -> list[str]:
    return NUM.findall(text)
