"""Build a deterministic workspace and tasks.json (with answers computed from the data; tasks go beside the workspace, not inside it).

usage: python make_workspace.py <new-empty-output-dir>
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    rnd = random.Random(7)

    # --- logs ---
    services = ["auth", "billing", "search", "gateway", "notify", "cache"]
    codes = {"E101": 5, "E102": 9, "E103": 14, "E104": 3, "E105": 7}
    msgs = ["request handled", "cache miss", "token refreshed", "queue drained", "health ok"]
    lines, err = [], Counter()
    for i in range(1500):
        ts = f"2026-10-01 {i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d}"
        svc = rnd.choice(services)
        r = rnd.random()
        if r < 0.12:
            code = rnd.choices(list(codes), weights=list(codes.values()))[0]
            err[code] += 1
            lines.append(f"{ts} [ERROR] {svc}: operation failed code={code}")
        elif r < 0.25:
            lines.append(f"{ts} [WARN] {svc}: slow response {rnd.randint(300, 900)}ms")
        else:
            lines.append(f"{ts} [INFO] {svc}: {rnd.choice(msgs)}")
    (out / "app.log").write_text("\n".join(lines) + "\n")
    top_code, top_n = err.most_common(1)[0]

    # --- sales ---
    regions = ["North", "South", "East", "West", "Central"]
    rows, total = ["region,product,units,price"], defaultdict(float)
    for _ in range(400):
        reg = rnd.choice(regions)
        units, price = rnd.randint(1, 20), round(rnd.uniform(5, 120), 2)
        rows.append(f"{reg},P{rnd.randint(1, 30)},{units},{price}")
        total[reg] += units * price
    (out / "sales.csv").write_text("\n".join(rows) + "\n")
    top_region = max(total, key=total.get)

    # --- code ---
    pkgs = {"billing": ["invoice", "tax", "refund"], "auth": ["login", "tokens"], "search": ["index", "query"],
            "util": ["dates", "money", "text"], "notify": ["email", "sms"]}
    for pkg, mods in pkgs.items():
        d = out / "src" / pkg
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("")
        for m in mods:
            body = [f'"""{pkg}.{m} module."""', ""]
            for j in range(6):
                body += [f"def {m}_helper_{j}(x, factor={j + 2}):", f"    return x * factor + {j}", ""]
            if (pkg, m) == ("billing", "tax"):
                body += ["def compute_tax(amount, rate=0.18):", "    return round(amount * rate, 2)", ""]
            if (pkg, m) == ("billing", "invoice"):
                body += ["def compute_discount(amount, rate=0.10):", "    return round(amount * rate, 2)", ""]
            (d / f"{m}.py").write_text("\n".join(body))

    # --- config ---
    default = ["# default settings"]
    for k in range(80):
        default.append(f"setting_{k}: {rnd.randint(1, 999)}")
    default += ["max_connections: 50", "timeout_seconds: 30"]
    (out / "config").mkdir()
    (out / "config" / "default.yaml").write_text("\n".join(default) + "\n")
    (out / "config" / "prod.yaml").write_text("# production overrides\nmax_connections: 200\nregion: eu\n")

    tasks = [
        {"id": "log_top_error", "prompt": "In app.log, which error code appears most often on ERROR lines, and how many times?",
         "expect": [top_code, str(top_n)]},
        {"id": "log_services", "prompt": "How many distinct services appear in app.log?", "expect": [str(len(services))]},
        {"id": "csv_top_region", "prompt": "In sales.csv, which region has the highest total revenue (units times price)? Reply with the region name.",
         "expect": [top_region]},
        {"id": "code_find", "prompt": "Which file defines compute_tax, and what is its default rate?", "expect": ["tax.py", "0.18"]},
        {"id": "config_effective", "prompt": "config/prod.yaml overrides config/default.yaml. What is the effective max_connections in prod?",
         "expect": ["200"]},
    ]
    out.with_name(out.name + ".tasks.json").write_text(json.dumps(tasks, indent=2))  # kept outside the workspace: it holds the answers
    print(f"workspace at {out}: top error {top_code} x{top_n}, top region {top_region}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
