"""
Build DECK.html and DECK.pdf.  Figures are inlined as base64 so the deck is a
single portable file.

    python3 make_deck.py

The slide content lives in build() below.  Numbers here must match the JSON
artifacts; REPORT.md is the canonical source and fact-checks against them.
"""
import base64
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIGS = ("fig1_funnel", "fig2_null", "fig3_replication",
        "fig4_ladder", "fig5_weather", "fig6_twonulls")


def main():
    html = HERE / "DECK.html"
    if not html.exists():
        sys.exit("DECK.html missing -- it is generated; see git history for the "
                 "inline builder, or re-run the slide definitions.")
    out = HERE / "DECK.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", f"file://{html}"], check=True)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
