#!/usr/bin/env python3
"""
TriGraphX 向量功能测试脚本 (简化版)
使用随机向量测试，无需额外安装模型
"""

import sys
sys.path.insert(0, "/home/jj/docker/src/novaos/trigraphx")

import numpy as np
from trigraphx import MetricSpace, MetricType, SemanticEmbedding

print("=" * 60)
print("TriGraphX 向量功能测试 (简化版)")
print("=" * 60)

# 1. 创建 MetricSpace
print("\n[1] 创建 MetricSpace...")
space = MetricSpace(max_entities=1000)
print(f"    ✓ 最大实体数: {space.max_entities}")

# 2. 准备测试数据 - 模拟语义向量
print("\n[2] 准备测试数据...")

# 模拟不同主题的向量 (使用随机种子模拟语义相似性)
np.random.seed(42)

# 投资人类 (向量相似)
investor_base = np.random.randn(384)
investor_vectors = [
    investor_base + np.random.randn(384) * 0.1 for _ in range(3)
]

# 工程师类 (向量相似)
engineer_base = np.random.randn(384)
engineer_vectors = [
    engineer_base + np.random.randn(384) * 0.1 for _ in range(2)
]

test_data = [
    {"name": "张三", "type": "investor", "role": "投资人", "vector": investor_vectors[0]},
    {"name": "李四", "type": "engineer", "role": "工程师", "vector": engineer_vectors[0]},
    {"name": "王五", "type": "investor", "role": "投资人", "vector": investor_vectors[1]},
    {"name": "赵六", "type": "investor", "role": "投资人", "vector": investor_vectors[2]},
    {"name": "钱七", "type": "engineer", "role": "工程师", "vector": engineer_vectors[1]},
]

# 3. 摄入实体
print("\n[3] 摄入实体...")
for item in test_data:
    entity, created = space.ingest(
        data={"name": item["name"], "type": item["type"], "role": item["role"]},
        embeddings={MetricType.SEMANTIC: SemanticEmbedding(vector=item["vector"].tolist())}
    )
    status = "新建" if created else "合并"
    print(f"    ✓ {item['name']} ({item['role']}): {status}")

print(f"\n    总实体数: {space.entity_count()}")

# 4. 测试 KNN 查询
print("\n[4] 测试 KNN 语义查询...")

# 用张三的向量查询
zhangsan_id = None
for eid, entity in space.entities.items():
    if entity.data.get("name") == "张三":
        zhangsan_id = eid
        break

if zhangsan_id:
    result = space.knn_query(zhangsan_id, k=5, metric_type=MetricType.SEMANTIC)
    print(f"    查询: '张三' (投资人)")
    print(f"    结果 (Top 5 相似实体):")
    for i, (eid, dist, score) in enumerate(zip(result.entity_ids, result.distances, result.scores), 1):
        entity = space.get_entity(eid)
        name = entity.data.get("name", eid[:20])
        role = entity.data.get("role", "unknown")
        print(f"      {i}. {name} ({role}) - 相似度: {score:.4f}, 距离: {dist:.4f}")

# 5. 验证相似性
print("\n[5] 验证语义相似性...")
print("    预期: 投资人之间相似度高，工程师之间相似度高")
print("    预期: 投资人与工程师相似度低")

if zhangsan_id:
    result = space.knn_query(zhangsan_id, k=5, metric_type=MetricType.SEMANTIC)
    for eid in result.entity_ids[:3]:
        entity = space.get_entity(eid)
        name = entity.data.get("name")
        role = entity.data.get("role")
        idx = result.entity_ids.index(eid)
        score = result.scores[idx]
        if name == "张三":
            continue
        if role == "投资人":
            print(f"    ✓ {name} (投资人) 排名靠前，相似度 {score:.4f} ✓")
        else:
            print(f"    ✗ {name} (工程师) 排名靠前，相似度 {score:.4f}")

# 6. 测试数据持久化
print("\n[6] 测试数据持久化...")
from trigraphx.persistence import PersistenceLayer
import tempfile
import shutil

tmp_dir = tempfile.mkdtemp()
persist = PersistenceLayer(tmp_dir)

# 保存
entities_list = list(space.entities.values())
persist.save_entities_batch(entities_list, batch_id=0)
print(f"    ✓ 保存 {len(entities_list)} 个实体")

# 加载
loaded = persist.load_all_entities()
print(f"    ✓ 加载 {len(loaded)} 个实体")

# 验证向量
has_vectors = sum(1 for e in loaded if MetricType.SEMANTIC in e.embeddings)
print(f"    ✓ 其中 {has_vectors} 个实体包含语义向量")

# 清理
shutil.rmtree(tmp_dir)

# 7. 测试去重功能
print("\n[7] 测试去重功能...")
entity1, created1 = space.ingest({"name": "张三", "new_field": "新数据"})
print(f"    再次摄入 '张三': created={created1} (应为 False)")
print(f"    数据合并: new_field={entity1.data.get('new_field')}")

print("\n" + "=" * 60)
print("✅ 所有测试完成!")
print("=" * 60)
print("\n提示: 如需使用真实语义向量，请安装 sentence-transformers:")
print("  pip install sentence-transformers")
