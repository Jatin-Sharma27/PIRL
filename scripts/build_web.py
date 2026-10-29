"""Assemble the self-contained interactive page: web/sim_template.html + results/web_data.json
+ web/findings.html  ->  results/PIRL_sim.html

    python scripts/build_web.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tpl = open(os.path.join(ROOT, "web", "sim_template.html"), encoding="utf-8").read()
    data = open(os.path.join(ROOT, "results", "web_data.json"), encoding="utf-8").read()
    fpath = os.path.join(ROOT, "web", "findings.html")
    findings = open(fpath, encoding="utf-8").read() if os.path.exists(fpath) else ""
    html = tpl.replace("__DATA__", data.replace("</", "<\\/")).replace("__FINDINGS__", findings)
    out = os.path.join(ROOT, "results", "PIRL_sim.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
