# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""三段式证据门控校准。

在 ch04 全部可答桶(A/B/C/E)和库外桶(D)上统计 Reranker Top1 原始分：
低线取“可答保留率至少 95%”条件下的最高阈值；高线取“库外直接放行率为 0”时的
最低阈值。低于低线拒答，高于高线直接生成，中间区交 LLM 做证据充分性自检。
运行:make calibrate-confidence(需 milvus + 上游可调通 + 知识库已建)。
产物:data/ch09/reports/confidence_calibration.{txt,json};「观测与成本」页只读这份 json。
"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime

from app.core import query_understanding, retrieval

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch09/reports"
_OUT = _OUT_DIR / "confidence_calibration.txt"
_OUT_JSON = _OUT_DIR / "confidence_calibration.json"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial", "E_multi"}
_SEM = asyncio.Semaphore(8)
_LINES: list[str] = []
HIGH_SAFETY_MARGIN = 0.02  # 防止直接放行线紧贴本轮最难库外题而过拟合


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


async def _top1(sample) -> tuple[str, float]:
    async with _SEM:
        # 与线上 retrieve_knowledge 完全同口径：先标准化/同义词扩展，再混合检索和精排。
        u = await query_understanding.understand(sample["query"])
        query = u["standard"]
        bm25_text = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")
        hits = await retrieval.search_knowledge(
            query, strategy="hybrid_rerank", bm25_text=bm25_text)
    return sample["bucket"], float(hits[0].get("rerank_score", 0.0)) if hits else 0.0


def _dist(name, xs) -> dict | None:
    """打印一行分布,同时把这行的数原样返回给 json 产物——两边同源,不各算一遍。"""
    xs = sorted(xs)
    if not xs:
        return None
    def p(q):
        return xs[min(len(xs) - 1, int(q * len(xs)))]
    stat = {"n": len(xs), "min": round(xs[0], 3), "p25": round(p(.25), 3),
            "p50": round(p(.5), 3), "p75": round(p(.75), 3), "max": round(xs[-1], 3)}
    _log(f"{name:12s} n={stat['n']:3d} min={stat['min']:.3f} p25={stat['p25']:.3f} "
         f"p50={stat['p50']:.3f} p75={stat['p75']:.3f} max={stat['max']:.3f}")
    return stat


async def main():
    with open(_ROOT / "tests/data/eval_ch04.jsonl", encoding="utf-8") as f:
        samples = [json.loads(ln) for ln in f]
    results = await asyncio.gather(*(_top1(s) for s in samples))
    answerable = [c for b, c in results if b in ANSWERABLE]
    absent = [c for b, c in results if b == "D_absent"]

    _log("=== Reranker Top1 分布(hybrid_rerank)===")
    dist = {"answerable": _dist("可答(ABCE)", answerable), "absent": _dist("应拒(D)", absent)}

    _log("\n=== 阈值扫描(通过率=conf>=t 的可答占比;放行率=conf>=t 的应拒占比)===")
    _log(f"{'t':>6s} {'可答通过率':>10s} {'应拒放行率':>10s} {'YoudenJ':>8s}")
    scan, best_t, best_j = [], 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        scan.append({"t": round(t, 2), "pass_rate": round(tpr, 3),
                     "leak_rate": round(fpr, 3), "youden_j": round(j, 3)})
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:10.3f} {j:8.3f}")
        if j > best_j:
            best_t, best_j = t, j
    # 低线优先控制误拒:保留至少 95% 可答题；高线优先控制误放:库外直接放行为 0。
    low_pick = max((r for r in scan if r["pass_rate"] >= 0.95),
                   key=lambda r: r["t"], default=scan[0])
    high_floor = max(absent) + HIGH_SAFETY_MARGIN
    high_pick = min((r for r in scan
                     if r["leak_rate"] == 0.0 and r["t"] >= high_floor),
                    key=lambda r: r["t"], default=scan[-1])
    if high_pick["t"] < low_pick["t"]:
        high_pick = low_pick
    recommended = {
        # threshold 保留给旧页面/外部消费者；语义等于 reject_below。
        "threshold": low_pick["t"],
        "accept_threshold": high_pick["t"],
        "reject_below": low_pick["t"],
        "direct_accept_at_or_above": high_pick["t"],
        "pass_rate": low_pick["pass_rate"],
        "leak_rate": low_pick["leak_rate"],
        "direct_accept_rate": high_pick["pass_rate"],
        "direct_accept_leak_rate": high_pick["leak_rate"],
        "high_safety_margin": HIGH_SAFETY_MARGIN,
        "single_threshold_youden": {"threshold": round(best_t, 2),
                                     "youden_j": round(best_j, 3)},
    }
    _log(f"\n推荐三段式门控:top1 < {low_pick['t']:.2f} 直接拒答;"
         f"top1 >= {high_pick['t']:.2f} 直接放行;中间区 LLM 自检")
    _log(f"低线可答保留率={low_pick['pass_rate']:.3f},库外进入复核/放行区比例={low_pick['leak_rate']:.3f};"
         f"高线可答直接放行率={high_pick['pass_rate']:.3f},库外直接放行率={high_pick['leak_rate']:.3f}")
    _log("请回填 app/config.py 的 rerank_reject_threshold / rerank_accept_threshold。")
    # 三段式有两条线，确定性拼注释比让 LLM 从单阈值扫描图中猜语义更可靠。
    note = (f"Top1 低于 {recommended['reject_below']:.2f} 直接拒答,"
            f"达到 {recommended['direct_accept_at_or_above']:.2f} 直接放行,中间区交 LLM 自检。"
            f"低线保留 {recommended['pass_rate']:.1%} 可答题,"
            f"高线下库外题直接放行率为 {recommended['direct_accept_leak_rate']:.1%}。")
    _log("\n读图小注:" + (note if note else "本轮没有(页面用兜底句)"))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _OUT_JSON.write_text(json.dumps({
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "strategy": "hybrid_rerank", "dataset": "tests/data/eval_ch04.jsonl",
                 "score": "reranker_top1", "n_answerable": len(answerable),
                 "n_absent": len(absent)},
        "distribution": dist,
        "scan": scan,
        "recommended": recommended,
        "read_notes": {"confidence_calibration": note} if note else {},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"报告已落 {_OUT.relative_to(_ROOT)} 与 {_OUT_JSON.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
