#!/usr/bin/env python3
"""
TriGraphX 向量功能测试脚本
使用本地缓存的 all-MiniLM-L6-v2 模型
"""

import sys
sys.path.insert(0, "/home/jj/docker/src/novaos/trigraphx")

from sentence_transformers import SentenceTransformer
from trigraphx import MetricSpace, MetricType, SemanticEmbedding, config

# 模型路径
MODEL_PATH = "/home/jj/docker/src/novaos/junjaxos/model_cache/all-MiniLM-L6-v2"

print("=" * 60)
print("TriGraphX 向量功能测试")
print("=" * 60)

# 1. 加载模型
print("\n[1] 加载 Sentence-BERT 模型...")
model = SentenceTransformer(MODEL_PATH)
print(f"    ✓ 模型加载成功: {MODEL_PATH}")
print(f"    ✓ 向量维度: {model.get_sentence_embedding_dimension()}")

# 2. 创建 MetricSpace
print("\n[2] 创建 MetricSpace...")
space = MetricSpace(max_entities=1000)
print(f"    ✓ 最大实体数: {space.max_entities}")

# 3. 准备测试数据
print("\n[3] 准备测试数据...")
test_data = [
    {"name": "张三", "text": "张三是红杉资本的投资人，专注于AI和科技领域投资"},
    {"name": "李四", "text": "李四是一位软件工程师，在字节跳动工作"},
    {"name": "王五", "text": "王五是腾讯的产品经理，负责社交产品"},
    {"name": "赵六", "text": "赵六是投资人，主要投资人工智能初创公司"},
    {"name": "钱七", "text": "钱七是AI研究员，在清华大学从事深度学习研究"},
]

# 4. 生成向量并摄入实体
print("\n[4] 生成向量并摄入实体...")
for item in test_data:
    # 生成语义向量
    vector = model.encode(item["text"]).tolist()
    
    # 摄入实体
    entity, created = space.ingest(
        data={"name": item["name"], "text": item["text"]},
        embeddings={MetricType.SEMANTIC: SemanticEmbedding(vector=vector)}
    )
    status = "新建" if created else "合并"
    print(f"    ✓ {item['name']}: {status}, ID={entity.id[:20]}...")

print(f"\n    总实体数: {space.entity_count()}")

# 5. 测试 KNN 查询
print("\n[5] 测试 KNN 语义查询...")
query_text = "投资人"
query_vector = model.encode(query_text).tolist()

# 创建临时查询实体
query_entity, _ = space.ingest(
    {"name": "query_temp"},
    embeddings={MetricType.SEMANTIC: SemanticEmbedding(vector=query_vector)}
)
query_entity_id = query_entity.id

result = space.knn_query(query_entity_id, k=5, metric_type=MetricType.SEMANTIC)

print(f"    查询: '{query_text}'")
print(f"    结果 (Top 5 相似实体):")
for i, (eid, dist, score) in enumerate(zip(result.entity_ids, result.distances, result.scores), 1):
    entity = space.get_entity(eid)
    if entity and entity.data.get("name") != query_entity_id:
        print(f"      {i}. {entity.data.get('name', eid[:20])} - 相似度: {score:.4f}, 距离: {dist:.4f}")

# 清理查询实体
space.hard_delete_entity(query_entity_id)

# 6. 测试实体间相似度
print("\n[6] 测试实体间相似度...")
zhangsan_id = None
zhaoliu_id = None
for eid, entity in space.entities.items():
    if entity.data.get("name") == "张三":
        zhangsan_id = eid
    if entity.data.get("name") == "赵六":
        zhaoliu_id = eid

if zhangsan_id and zhaoliu_id:
    result = space.knn_query(zhangsan_id, k=5, metric_type=MetricType.SEMANTIC)
    zhaoliu_rank = None
    for i, eid in enumerate(result.entity_ids):
        if eid == zhaoliu_id:
            zhaoliu_rank = i + 1
            break
    
    print(f"    '张三' vs '赵六':")
    print(f"      - 两人都是投资人，语义相似")
    if zhaoliu_rank:
        print(f"      - '赵六' 在 '张三' 的 KNN 结果中排名第 {zhaoliu_rank}")

# 7. 测试数据持久化
print("\n[7] 测试数据持久化...")
from trigraphx.persistence import PersistenceLayer
import tempfile
import shutil

tmp_dir = tempfile.mkdtemp()
persist = PersistenceLayer(tmp_dir)

# 保存
entities_list = list(space.entities.values())
persist.save_entities_batch(entities_list, batch_id=0)
print(f"    ✓ 保存 {len(entities_list)} 个实体到 {tmp_dir}")

# 加载
loaded = persist.load_all_entities()
print(f"    ✓ 加载 {len(loaded)} 个实体")

# 验证
if len(loaded) == len(entities_list):
    print("    ✓ 数据完整性验证通过")
else:
    print(f"    ✗ 数据丢失: 期望 {len(entities_list)}, 实际 {len(loaded)}")

# 清理
shutil.rmtree(tmp_dir)

print("\n" + "=" * 60)
print("✅ 所有测试完成!")
print("=" * 60)
