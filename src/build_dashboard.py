"""
Injects output/metrics.json into src/template.html and writes the finished
dashboard to output/dashboard.html.

The dashboard holds no numbers of its own — it is a view over the metrics file.
Re-run analysis.py, re-run this, and the page updates.

Run:  python src/build_dashboard.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
metrics = json.loads((ROOT / "output" / "metrics.json").read_text())
html = (ROOT / "src" / "template.html").read_text()

out = ROOT / "output" / "dashboard.html"
out.write_text(html.replace("__METRICS__", json.dumps(metrics, separators=(",", ":"))))
print(f"Wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
