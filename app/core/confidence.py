# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""生成前证据门控。

只使用 Reranker Top1 原始相关分，不再把证据数、分差和关键词命中按人工权重揉成一个
难解释的分数。两个阈值把请求分成三段：

* reject：低于拒答线，直接兜底；
* review：处于临界区，交给 LLM 判断证据是否完整；
* accept：高于直接放行线，跳过额外自检。

阈值必须在同一模型、同一服务商、同一知识库切分方式下用评估集校准。
"""
from dataclasses import dataclass
from typing import Literal


GateDecision = Literal["reject", "review", "accept"]


@dataclass(frozen=True)
class EvidenceGate:
    top1_score: float
    decision: GateDecision

    @property
    def trace(self) -> dict:
        return {"top1_score": self.top1_score, "decision": self.decision}


def evaluate_evidence_gate(
    hits: list[dict], reject_threshold: float, accept_threshold: float,
) -> EvidenceGate:
    """按 Top1 相关分返回三段式决策；无召回永远拒答。"""
    if not 0.0 <= reject_threshold <= accept_threshold <= 1.0:
        raise ValueError("证据门控阈值必须满足 0 <= reject <= accept <= 1")
    top1 = float(hits[0].get("rerank_score", 0.0)) if hits else 0.0
    if not hits or top1 < reject_threshold:
        decision: GateDecision = "reject"
    elif top1 < accept_threshold:
        decision = "review"
    else:
        decision = "accept"
    return EvidenceGate(round(top1, 6), decision)


def snapshot_from_hits(hits: list[dict], top_n: int = 3) -> list[dict]:
    """保存 Top N 原文与得分，供低置信度复盘和用户差评回捞。"""
    return [{"question": h.get("question", ""), "answer": h.get("answer", ""),
             "rerank_score": float(h.get("rerank_score", 0.0)),
             "section_path": h.get("section_path", "")}
            for h in hits[:top_n]]


def snapshot_from_hits(hits: list[dict], top_n: int = 3) -> list[dict]:
    """落池召回快照的统一出口:Top N 的原文与得分(审核页给人看的那份)。"""
    return [{"question": h.get("question", ""), "answer": h.get("answer", ""),
             "rerank_score": float(h.get("rerank_score", 0.0)),
             "section_path": h.get("section_path", "")}
            for h in hits[:top_n]]
