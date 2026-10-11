"""长期记忆的 Milvus 用户级向量索引；MySQL 才是事实源。"""
from pymilvus import DataType

from app.config import settings
from app.kb.milvus_client import DIM, acall, get_client


def ensure_collection(client=None) -> None:
    client = client or get_client()
    name = settings.long_term_memory_collection
    if client.has_collection(name):
        client.load_collection(name)
        return
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("user_id", DataType.VARCHAR, max_length=64)
    schema.add_field("memory_type", DataType.VARCHAR, max_length=32)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    index = client.prepare_index_params()
    index.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(name, schema=schema, index_params=index)
    client.load_collection(name)


def upsert(memory_id: int, user_id: str, memory_type: str, vector: list[float]) -> None:
    client = get_client()
    ensure_collection(client)
    client.upsert(settings.long_term_memory_collection, [{
        "id": memory_id, "user_id": user_id,
        "memory_type": memory_type, "dense": vector,
    }])
    client.flush(settings.long_term_memory_collection)


def search(user_id: str, vector: list[float], limit: int) -> list[dict]:
    client = get_client()
    ensure_collection(client)
    safe_user = user_id.replace('"', '\\"')
    result = client.search(
        settings.long_term_memory_collection,
        data=[vector], anns_field="dense", limit=limit,
        filter=f'user_id == "{safe_user}"',
        output_fields=["memory_type"],
        search_params={"metric_type": "COSINE"},
    )
    return [{"id": int(h["id"]), "score": float(h["distance"]),
             "memory_type": h.get("entity", {}).get("memory_type", "")}
            for h in result[0]]


async def aupsert(memory_id: int, user_id: str, memory_type: str,
                  vector: list[float]) -> None:
    await acall(upsert, memory_id, user_id, memory_type, vector)


async def asearch(user_id: str, vector: list[float], limit: int) -> list[dict]:
    return await acall(search, user_id, vector, limit)
