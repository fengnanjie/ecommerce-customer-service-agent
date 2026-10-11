from app.core import long_term_memory as ltm


def _candidate(**kwargs):
    data = {
        "memory_type": "preference", "subject_key": "reply_style",
        "content": "用户偏好简短回复", "confidence": 0.9,
        "confirmed": True, "source_message_ids": [1],
    }
    data.update(kwargs)
    return ltm.MemoryCandidate(**data)


def test_trigger_every_six_turns_or_explicit_signal(monkeypatch):
    monkeypatch.setattr(ltm.settings, "long_term_memory_extract_every_turns", 6)
    assert not ltm.should_trigger(5, "继续查一下")
    assert ltm.should_trigger(6, "继续查一下")
    assert ltm.should_trigger(1, "请记住我偏好简短回复")
    assert ltm.should_trigger(1, "已经解决了", confirmed_case=True)


def test_candidate_rejects_sensitive_and_unconfirmed_case():
    assert ltm.is_safe_candidate(_candidate(), {1})
    assert not ltm.is_safe_candidate(_candidate(content="手机号13800138000"), {1})
    assert not ltm.is_safe_candidate(_candidate(
        memory_type="case", confirmed=False, content="重启解决故障"), {1})
    assert not ltm.is_safe_candidate(_candidate(source_message_ids=[99]), {1})


async def test_retrieve_filters_threshold_and_mysql_active(monkeypatch):
    class Row:
        def __init__(self, mid, content):
            self.id, self.memory_type, self.subject_key = mid, "fact", "device"
            self.content, self.confidence = content, 0.9

    async def embed(_): return [0.1]
    async def search(*_):
        return [{"id": 1, "score": 0.80}, {"id": 2, "score": 0.50},
                {"id": 3, "score": 0.75}]
    async def active(_uid, ids):
        assert ids == [1, 3]
        return [Row(1, "用户使用A型号")]  # 3 已被 superseded，不得注入
    async def touch(ids): assert ids == [1]
    monkeypatch.setattr(ltm.embeddings, "embed_query", embed)
    monkeypatch.setattr(ltm.memory_index, "asearch", search)
    monkeypatch.setattr(ltm.repository, "get_active_memories", active)
    monkeypatch.setattr(ltm.repository, "touch_memories", touch)
    monkeypatch.setattr(ltm.settings, "long_term_memory_threshold", 0.62)
    out = await ltm.retrieve("u1", "我的设备")
    assert [m["id"] for m in out] == [1]
