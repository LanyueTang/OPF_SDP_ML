import re
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("experiments/protest_analysis_0204/data/case2746wop")
PATTERN = re.compile(r"rep_(\d+)_perturbation\.csv$")

def load_all_reps(data_dir):
    dfs = []
    for f in sorted(data_dir.glob("*_rep_*_perturbation.csv")):
        m = PATTERN.search(f.name)
        rep = int(m.group(1))
        df = pd.read_csv(f)
        df["rep"] = rep
        df["source_file"] = f.name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)

    out["SolveTime"] = pd.to_numeric(out["SolveTime"], errors="coerce")
    out["Merge"] = out["Merge"].astype(str).str.lower().map({"true": True, "false": False})
    return out

df = load_all_reps(DATA_DIR)
TIME_COL = "SolveTime"

group_keys = [
    "Case",
    "Perturbation",      # ★ load 维度
    "Formulation",
    "Merge",
    "A_parameter",
]

g = df.groupby(group_keys)[TIME_COL]

summary = g.agg(
    n="count",
    mean="mean",
    std="std",
    median="median",
    max="max",
).reset_index()

summary["cv"] = summary["std"] / summary["mean"]
summary["tail_ratio"] = summary["max"] / summary["median"]

# 诊断 flag
summary["cv_ge_5pct"] = summary["cv"] >= 0.05
summary["tail_ge_2x"] = summary["tail_ratio"] >= 2.0
# 只分析成功的
df_ok = df[df["Status"].astype(str).str.upper() == "OPTIMAL"].copy()

rank_keys = ["Case", "Perturbation", "rep"]

def rank_within_rep(x):
    x = x.sort_values(TIME_COL)
    x["rank"] = np.arange(1, len(x) + 1)
    return x

ranks = (
    df_ok.groupby(rank_keys, group_keys=False)
         .apply(rank_within_rep)
         .reset_index(drop=True)
)

rank_changes = (
    ranks.groupby(
        ["Case", "Perturbation", "Formulation", "Merge", "A_parameter"]
    )["rank"]
    .nunique()
    .reset_index(name="rank_unique_count")
)

rank_changes["rank_swaps"] = rank_changes["rank_unique_count"] > 1
from itertools import combinations

def pairwise_winprob(df_in):
    out = []
    for (case, pert, rep), grp in df_in.groupby(["Case", "Perturbation", "rep"]):
        pv = grp.pivot_table(
            index="rep",
            columns=["Formulation", "Merge", "A_parameter"],
            values=TIME_COL,
            aggfunc="mean",
        )
        cols = pv.columns
        for a, b in combinations(cols, 2):
            d = pv[[a, b]].dropna()
            if len(d) == 0:
                continue
            p = (d[a] < d[b]).mean()
            out.append((*a, *b, case, pert, len(d), p))

    return pd.DataFrame(
        out,
        columns=[
            "FormA", "MergeA", "AA",
            "FormB", "MergeB", "AB",
            "Case", "Perturbation",
            "n_rep", "P(A_faster_than_B)"
        ],
    )

pw = pairwise_winprob(df_ok)
from pathlib import Path

OUT_DIR = Path("experiments/protest_analysis_0204/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) 保存三张核心表
summary_path = OUT_DIR / "summary.csv"
rank_path = OUT_DIR / "rank_changes.csv"
pw_path = OUT_DIR / "pairwise_winprob.csv"

summary.to_csv(summary_path, index=False)
rank_changes.to_csv(rank_path, index=False)
pw.to_csv(pw_path, index=False)

print(f"[OK] wrote {summary_path}")
print(f"[OK] wrote {rank_path}")
print(f"[OK] wrote {pw_path}")

# 2) 额外：生成一个 quick report（按 Perturbation 汇总）
#    - 每个 load 下，CV 最大的若干策略
#    - rank swap 最严重的若干策略
#    - 胜率最极端的若干 pair

def _topk(df, by, k=10, ascending=False):
    return df.sort_values(by, ascending=ascending).head(k)

report_lines = []
report_lines.append("# Quick runtime-stochasticity report\n")

for (case, pert), sub in summary.groupby(["Case", "Perturbation"]):
    report_lines.append(f"## Case={case}, Perturbation={pert}\n")

    # A) CV top
    report_lines.append("### Highest CV (std/mean)\n")
    top_cv = _topk(sub, "cv", k=10, ascending=False)[
        ["Formulation", "Merge", "A_parameter", "n", "mean", "std", "cv", "tail_ratio"]
    ]
    report_lines.append(top_cv.to_markdown(index=False))
    report_lines.append("\n")

    # B) Tail top
    report_lines.append("### Highest tail ratio (max/median)\n")
    top_tail = _topk(sub, "tail_ratio", k=10, ascending=False)[
        ["Formulation", "Merge", "A_parameter", "n", "median", "max", "tail_ratio", "cv"]
    ]
    report_lines.append(top_tail.to_markdown(index=False))
    report_lines.append("\n")

    # C) Rank swaps top (if exists)
    rc_sub = rank_changes[(rank_changes["Case"] == case) & (rank_changes["Perturbation"] == pert)].copy()
    if len(rc_sub) > 0:
        report_lines.append("### Most rank-unstable strategies (rank_unique_count)\n")
        top_swaps = _topk(rc_sub, "rank_unique_count", k=10, ascending=False)[
            ["Formulation", "Merge", "A_parameter", "rank_unique_count", "rank_swaps"]
        ]
        report_lines.append(top_swaps.to_markdown(index=False))
        report_lines.append("\n")

    # D) Pairwise extremes (if exists)
    pw_sub = pw[(pw["Case"] == case) & (pw["Perturbation"] == pert)].copy()
    if len(pw_sub) > 0:
        report_lines.append("### Pairwise win-probability extremes\n")
        # 最接近 1 的
        top_hi = _topk(pw_sub, "P(A_faster_than_B)", k=5, ascending=False)[
            ["FormA","MergeA","AA","FormB","MergeB","AB","n_rep","P(A_faster_than_B)"]
        ]
        report_lines.append("**Most decisive (A much faster than B):**\n")
        report_lines.append(top_hi.to_markdown(index=False))
        report_lines.append("\n")

        # 最接近 0.5 的（最不确定）
        pw_sub["dist_to_0p5"] = (pw_sub["P(A_faster_than_B)"] - 0.5).abs()
        top_unc = pw_sub.sort_values("dist_to_0p5").head(5)[
            ["FormA","MergeA","AA","FormB","MergeB","AB","n_rep","P(A_faster_than_B)"]
        ]
        report_lines.append("**Most uncertain (close to 0.5):**\n")
        report_lines.append(top_unc.to_markdown(index=False))
        report_lines.append("\n")

report_path = OUT_DIR / "quick_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"[OK] wrote {report_path}")

# 3) 终端里顺手给你看一眼（可删）
print("\n=== preview: summary (top 10 by cv) ===")
print(summary.sort_values("cv", ascending=False).head(10).to_string(index=False))
# ============================================================
# WINNER ANALYSIS: For each load (Perturbation), which strategies
# became rank-1 across the 20 repetitions?
# ============================================================

# 1) 每次 (Case, Perturbation, rep) 选出 SolveTime 最小的策略 = winner
winner_rows = (
    df_ok.sort_values(["Case", "Perturbation", "rep", TIME_COL], ascending=True)
         .groupby(["Case", "Perturbation", "rep"], as_index=False)
         .head(1)  # ties: take the first after sorting
         .copy()
)

# 2) 给 winner 一个可读的策略名
winner_rows["strategy"] = (
    winner_rows["Formulation"].astype(str)
    + "; Merge=" + winner_rows["Merge"].astype(str)
    + "; A=" + winner_rows["A_parameter"].astype(str)
)

# 3) 对每个 load 汇总：哪些策略当过第一名、各赢了几次
winner_counts = (
    winner_rows.groupby(["Case", "Perturbation", "strategy"])
              .size()
              .reset_index(name="wins")
              .sort_values(["Case", "Perturbation", "wins"], ascending=[True, True, False])
)

# 4) 打印：每个 load 的 winner 列表 + “是否换过第一名”
print("\n================ WINNER SUMMARY (Rank-1 across reps) ================\n")
for (case, pert), sub in winner_counts.groupby(["Case", "Perturbation"]):
    strategies = sub["strategy"].tolist()
    wins = sub["wins"].tolist()
    unique_winners = len(strategies)

    # 是否换人：unique_winners > 1 即换过第一名
    swapped = (unique_winners > 1)

    print(f"Case={case}, Perturbation={pert}")
    print(f"  Unique winners = {unique_winners}  -->  {'SWAPPED' if swapped else 'STABLE'}")
    print("  Winners (strategy : wins out of 20 OPTIMAL reps):")
    for s, w in zip(strategies, wins):
        print(f"    - {s} : {w}")
    print("")

# 5)（可选但强烈推荐）保存成文件，方便你放进论文表格
winner_path = OUT_DIR / "winners_by_load.csv"
winner_counts.to_csv(winner_path, index=False)
print(f"[OK] wrote {winner_path}")

# 6)（可选）也保存每一次 rep 的赢家明细，方便 debug
winner_detail_path = OUT_DIR / "winners_per_rep.csv"
winner_rows[["Case","Perturbation","rep","Formulation","Merge","A_parameter",TIME_COL,"strategy"]].to_csv(
    winner_detail_path, index=False
)
print(f"[OK] wrote {winner_detail_path}")
