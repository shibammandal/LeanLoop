"""FIXED (do not edit during research). Builds the workspace and the dev/test task splits.

Like autoresearch's prepare.py: one-time data prep, never touched by the research loop.
dev  = tasks the research loop optimises on.
test = held-out tasks of the same kinds with different questions, only run at the end (and for pi).
Answers are computed from the generated data and written beside the workspace, never inside it.

usage: python research/prepare.py <new-empty-output-dir>
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def top(counter, largest=True):
    """Return the unique arg-max (or arg-min); fail loudly on a tie so no task is ambiguous."""
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=largest)
    assert items[0][1] != items[1][1], f"tie: {items[:2]}"
    return items[0][0]


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    rnd = random.Random(11)

    # --- app.log: 1500 lines ---
    services = ["auth", "billing", "search", "gateway", "notify", "cache"]
    codes = {"E101": 5, "E102": 9, "E103": 14, "E104": 3, "E105": 7}
    msgs = ["request handled", "cache miss", "token refreshed", "queue drained", "health ok"]
    lines, err_code, err_svc, warn_svc, slow = [], Counter(), Counter(), Counter(), defaultdict(int)
    errors = []
    for i in range(1500):
        ts = f"2026-10-01 {i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d}"
        svc = rnd.choice(services)
        r = rnd.random()
        if r < 0.12:
            code = rnd.choices(list(codes), weights=list(codes.values()))[0]
            err_code[code] += 1
            err_svc[svc] += 1
            errors.append((svc, code))
            lines.append(f"{ts} [ERROR] {svc}: operation failed code={code}")
        elif r < 0.25:
            ms = rnd.randint(300, 999)
            warn_svc[svc] += 1
            slow[svc] = max(slow[svc], ms)
            lines.append(f"{ts} [WARN] {svc}: slow response {ms}ms")
        else:
            lines.append(f"{ts} [INFO] {svc}: {rnd.choice(msgs)}")
    (out / "app.log").write_text("\n".join(lines) + "\n")

    # --- sales.csv: 400 rows ---
    regions = ["North", "South", "East", "West", "Central"]
    rows, revenue, units_reg = ["region,product,units,price"], Counter(), Counter()
    prod_rows, prod_units = Counter(), Counter()
    for _ in range(400):
        reg, prod = rnd.choice(regions), f"P{rnd.randint(1, 30)}"
        units, price = rnd.randint(1, 20), round(rnd.uniform(5, 120), 2)
        rows.append(f"{reg},{prod},{units},{price}")
        revenue[reg] += units * price
        units_reg[reg] += units
        prod_rows[prod] += 1
        prod_units[prod] += units
    (out / "sales.csv").write_text("\n".join(rows) + "\n")

    # --- users.csv: 3000 rows (a big file) ---
    countries = ["India", "Japan", "Brazil", "Germany", "Kenya", "Canada", "France", "Mexico"]
    plans = ["free", "basic", "pro", "team"]
    users, by_plan, by_country = ["id,name,country,plan,signup"], Counter(), Counter()
    user_info = {}
    for uid in range(1, 3001):
        c, p = rnd.choice(countries), rnd.choices(plans, weights=[50, 25, 15, 10])[0]
        name = "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7))
        users.append(f"{uid},{name},{c},{p},2025-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}")
        by_plan[p] += 1
        by_country[c] += 1
        user_info[uid] = (c, p)
    (out / "users.csv").write_text("\n".join(users) + "\n")

    # --- src/: python packages, random default factors ---
    pkgs = {"billing": ["invoice", "tax", "refund"], "auth": ["login", "tokens"], "search": ["index", "query"],
            "util": ["dates", "money", "text"], "notify": ["email", "sms"]}
    factor = {}
    for pkg, mods in pkgs.items():
        d = out / "src" / pkg
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("")
        for m in mods:
            body = [f'"""{pkg}.{m} module."""', ""]
            for j in range(6):
                f = rnd.randint(2, 19)
                factor[f"{m}_helper_{j}"] = f
                body += [f"def {m}_helper_{j}(x, factor={f}):", f"    return x * factor + {j}", ""]
            if (pkg, m) == ("billing", "tax"):
                body += ["def compute_tax(amount, rate=0.18):", "    return round(amount * rate, 2)", ""]
            if (pkg, m) == ("billing", "invoice"):
                body += ["def compute_discount(amount, rate=0.12):", "    return round(amount * rate, 2)", ""]
            (d / f"{m}.py").write_text("\n".join(body))

    # --- config/ ---
    default, setting = ["# default settings"], {}
    for k in range(80):
        setting[k] = rnd.randint(1, 999)
        default.append(f"setting_{k}: {setting[k]}")
    default += ["max_connections: 50", "timeout_seconds: 30"]
    (out / "config").mkdir()
    (out / "config" / "default.yaml").write_text("\n".join(default) + "\n")
    (out / "config" / "prod.yaml").write_text("# production overrides\nmax_connections: 200\nregion: eu\n")

    def t(tid, prompt, *expect):
        return {"id": tid, "prompt": prompt, "expect": [str(e) for e in expect]}

    dev = [
        t("log_code_count", "How many ERROR lines in app.log have code=E102?", err_code["E102"]),
        t("log_top_error_svc", "Which service has the most ERROR lines in app.log?", top(err_svc)),
        t("log_max_slow", "What is the largest slow response time (in ms) on WARN lines for the gateway service in app.log?", slow["gateway"]),
        t("log_last_error", "What error code is on the last ERROR line of app.log?", errors[-1][1]),
        t("csv_top_revenue", "In sales.csv, which region has the highest total revenue (units times price)?", top(revenue)),
        t("csv_product_rows", "How many rows in sales.csv are for product P7?", prod_rows["P7"]),
        t("code_tax", "Which file defines compute_tax, and what is its default rate?", "tax.py", "0.18"),
        t("code_factor", "What is the default value of the factor argument of query_helper_4?", factor["query_helper_4"]),
        t("cfg_effective", "config/prod.yaml overrides config/default.yaml. What is the effective max_connections in prod?", 200),
        t("cfg_setting", "In config/default.yaml, what is the value of setting_42?", setting[42]),
        t("users_lookup", "In users.csv, which country is the user with id 2471 from?", user_info[2471][0]),
        t("users_plan_count", "How many users in users.csv are on the pro plan?", by_plan["pro"]),
    ]
    test = [
        t("log_code_count", "How many ERROR lines in app.log have code=E105?", err_code["E105"]),
        t("log_top_warn_svc", "Which service has the most WARN lines in app.log?", top(warn_svc)),
        t("log_max_slow", "What is the largest slow response time (in ms) on WARN lines for the auth service in app.log?", slow["auth"]),
        t("log_first_error", "Which service logged the first ERROR line of app.log?", errors[0][0]),
        t("csv_low_units", "In sales.csv, which region has the lowest total units sold?", top(units_reg, largest=False)),
        t("csv_product_units", "What is the total number of units sold for product P12 in sales.csv?", prod_units["P12"]),
        t("code_discount", "Which file defines compute_discount, and what is its default rate?", "invoice.py", "0.12"),
        t("code_factor", "What is the default value of the factor argument of dates_helper_2?", factor["dates_helper_2"]),
        t("cfg_effective", "config/prod.yaml overrides config/default.yaml. What is the effective timeout_seconds in prod?", 30),
        t("cfg_setting", "In config/default.yaml, what is the value of setting_67?", setting[67]),
        t("users_lookup", "In users.csv, which plan is the user with id 1388 on?", user_info[1388][1]),
        t("users_country_count", "How many users in users.csv are from Japan?", by_country["Japan"]),
    ]
    out.with_name(out.name + ".dev.json").write_text(json.dumps(dev, indent=2))
    out.with_name(out.name + ".test.json").write_text(json.dumps(test, indent=2))
    print(f"workspace at {out}; {len(dev)} dev tasks, {len(test)} test tasks")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
