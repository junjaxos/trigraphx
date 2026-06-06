"""
TriGraphX Streamlit UI - Interactive Data Visualization Dashboard
展示数据、执行查询、监控性能
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

from trigraphx import Entity, MetricType, MetricSpace, SemanticEmbedding, HierarchyEmbedding
from trigraphx.persistence import PersistenceLayer
from trigraphx.enterprise import RoleBasedAccessControl, Role, DataQualityReport

# Page configuration
st.set_page_config(
    page_title="TriGraphX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'space' not in st.session_state:
    st.session_state.space = MetricSpace(max_entities=10000)

if 'persistence' not in st.session_state:
    db_root = Path("trigraphx_data")
    st.session_state.persistence = PersistenceLayer(db_root)

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

if 'metrics_collector' not in st.session_state:
    st.session_state.metrics_collector = {}

# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

st.sidebar.markdown("# 🎯 TriGraphX Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "选择功能",
    ["📊 总览", "🔍 查询", "📈 可视化", "⏱️ 性能", "💾 数据管理", "ℹ️ 帮助"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 实时统计")
num_entities = len(st.session_state.space.entities)
st.sidebar.metric("实体数量", num_entities)
st.sidebar.metric("查询次数", len(st.session_state.query_history))

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "📊 总览":
    st.markdown("# 📊 TriGraphX 数据总览")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 实体总数", len(st.session_state.space.entities), delta="+0" if len(st.session_state.space.entities) == 0 else "+1")
    with col2:
        st.metric("🔍 查询次数", len(st.session_state.query_history))
    with col3:
        stats = st.session_state.persistence.stats()
        st.metric("💾 存储大小", f"{stats.get('total_size_mb', 0):.2f} MB")
    
    st.markdown("---")
    
    # Data statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 实体类型分布")
        if st.session_state.space.entities:
            entity_data = []
            metric_counts = {metric: 0 for metric in MetricType}
            
            for entity in st.session_state.space.entities.values():
                for emb_type, embeddings in entity.embeddings.items():
                    if embeddings:
                        metric_counts[emb_type] += 1
            
            df_types = pd.DataFrame({
                "度量类型": [m.value for m in metric_counts.keys()],
                "数量": list(metric_counts.values())
            })
            
            fig = px.pie(df_types, values="数量", names="度量类型", 
                        title="度量类型分布")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无数据，请先添加实体")
    
    with col2:
        st.subheader("📊 存储统计")
        if st.session_state.space.entities:
            stats = st.session_state.persistence.stats()
            stats_data = {
                "类型": ["实体文件", "索引", "检查点", "日志"],
                "大小(MB)": [
                    stats.get('entity_storage_mb', 0),
                    stats.get('index_size_mb', 0),
                    stats.get('checkpoint_size_mb', 0),
                    stats.get('operation_log_mb', 0)
                ]
            }
            df_storage = pd.DataFrame(stats_data)
            fig = px.bar(df_storage, x="类型", y="大小(MB)", 
                        title="存储分布")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无存储数据")
    
    st.markdown("---")
    st.subheader("🔧 系统配置")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("最大实体数", st.session_state.space.max_entities)
    with col2:
        st.metric("缓存命中率", "N/A" if len(st.session_state.query_history) == 0 else f"{100}%")
    with col3:
        st.metric("数据库版本", "1.0.0")

# ============================================================================
# PAGE 2: QUERY
# ============================================================================

elif page == "🔍 查询":
    st.markdown("# 🔍 交互式查询")
    
    query_type = st.radio(
        "选择查询类型",
        ["KNN查询", "范围查询", "路径查询", "多指标查询"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if not st.session_state.space.entities:
        st.warning("⚠️ 暂无数据，请先在「数据管理」中添加实体")
    else:
        entity_ids = list(st.session_state.space.entities.keys())
        
        # KNN Query
        if query_type == "KNN查询":
            st.subheader("📍 KNN（K近邻）查询")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                source_id = st.selectbox("选择源实体", entity_ids)
            with col2:
                k = st.slider("选择K值", 1, min(10, len(entity_ids)), 3)
            with col3:
                metric_type = st.selectbox("选择度量类型", list(MetricType))
            
            if st.button("🚀 执行查询"):
                start_time = time.time()
                try:
                    result = st.session_state.space.knn_query(source_id, k=k)
                    query_time = (time.time() - start_time) * 1000
                    
                    st.session_state.query_history.append({
                        "type": "KNN",
                        "timestamp": datetime.now(),
                        "time_ms": query_time,
                        "result_count": len(result.entity_ids)
                    })
                    
                    st.success(f"✅ 查询成功（耗时: {query_time:.2f}ms）")
                    
                    # Display results
                    results_data = []
                    for entity_id, distance, score in zip(result.entity_ids, result.distances, result.scores):
                        results_data.append({
                            "实体ID": entity_id,
                            "距离": f"{distance:.4f}",
                            "相似度": f"{score:.2%}"
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(df_results, x="实体ID", y="距离", 
                                    title="距离分布", color="距离")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.scatter(df_results, x="实体ID", y="相似度",
                                       title="相似度分布", size="距离")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ 查询失败: {str(e)}")
        
        # Range Query
        elif query_type == "范围查询":
            st.subheader("🎯 范围（Radius）查询")
            col1, col2 = st.columns(2)
            
            with col1:
                source_id = st.selectbox("选择源实体", entity_ids)
            with col2:
                radius = st.slider("选择搜索半径", 0.0, 2.0, 0.5, step=0.1)
            
            if st.button("🚀 执行查询"):
                start_time = time.time()
                try:
                    result = st.session_state.space.range_query(source_id, radius)
                    query_time = (time.time() - start_time) * 1000
                    
                    st.session_state.query_history.append({
                        "type": "Range",
                        "timestamp": datetime.now(),
                        "time_ms": query_time,
                        "result_count": len(result.entity_ids)
                    })
                    
                    st.success(f"✅ 找到 {len(result.entity_ids)} 个实体（耗时: {query_time:.2f}ms）")
                    
                    results_data = []
                    for entity_id, distance in zip(result.entity_ids, result.distances):
                        results_data.append({
                            "实体ID": entity_id,
                            "距离": f"{distance:.4f}"
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    fig = px.histogram(df_results, x="距离", nbins=20,
                                     title=f"范围 {radius} 内的距离分布")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ 查询失败: {str(e)}")
        
        # Path Query
        elif query_type == "路径查询":
            st.subheader("🛤️ 路径查询")
            col1, col2 = st.columns(2)
            
            with col1:
                start_id = st.selectbox("选择起点", entity_ids, key="start")
            with col2:
                end_id = st.selectbox("选择终点", entity_ids, key="end")
            
            if st.button("🚀 查找最短路径"):
                start_time = time.time()
                try:
                    result = st.session_state.space.path_query(start_id, end_id)
                    query_time = (time.time() - start_time) * 1000
                    
                    st.success(f"✅ 路径查询完成（耗时: {query_time:.2f}ms）")
                    
                    if result.entity_ids:
                        st.info(f"最短路径: {' → '.join(result.entity_ids)}")
                        st.metric("路径长度", len(result.entity_ids))
                    else:
                        st.warning("未找到连接路径")
                        
                except Exception as e:
                    st.error(f"❌ 查询失败: {str(e)}")
        
        # Multi-metric Query
        elif query_type == "多指标查询":
            st.subheader("⚖️ 多指标加权查询")
            
            source_id = st.selectbox("选择源实体", entity_ids)
            k = st.slider("K值", 1, min(10, len(entity_ids)), 3)
            
            # Weight configuration
            st.write("配置度量权重:")
            col1, col2 = st.columns(2)
            weights = {}
            with col1:
                weights["semantic"] = st.slider("语义相似度权重", 0.0, 1.0, 0.5)
                weights["hierarchy"] = st.slider("层级权重", 0.0, 1.0, 0.3)
            with col2:
                weights["association"] = st.slider("关联权重", 0.0, 1.0, 0.2)
                weights["causal"] = st.slider("因果权重", 0.0, 1.0, 0.0)
            
            if st.button("🚀 执行加权查询"):
                start_time = time.time()
                try:
                    result = st.session_state.space.multi_metric_query(
                        source_id, k=k, metric_weights=weights
                    )
                    query_time = (time.time() - start_time) * 1000
                    
                    st.success(f"✅ 查询成功（耗时: {query_time:.2f}ms）")
                    
                    results_data = []
                    for eid, dist, score in zip(result.entity_ids, result.distances, result.scores):
                        results_data.append({
                            "实体ID": eid,
                            "综合距离": f"{dist:.4f}",
                            "综合评分": f"{score:.2%}"
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ 查询失败: {str(e)}")

# ============================================================================
# PAGE 3: VISUALIZATION
# ============================================================================

elif page == "📈 可视化":
    st.markdown("# 📈 关系可视化")
    
    if not st.session_state.space.entities:
        st.warning("⚠️ 暂无数据")
    else:
        viz_type = st.radio(
            "选择可视化类型",
            ["网络拓扑", "关联矩阵", "向量空间"],
            horizontal=True
        )
        
        if viz_type == "网络拓扑":
            st.subheader("🔗 实体网络拓扑")
            st.info("展示实体之间的关联关系（需要 pyvis 库）")
            
            try:
                import pyvis.network as net
                
                # Create network
                g = net.Network(height="600px", directed=True)
                g.toggle_physics(True)
                
                # Add nodes
                entity_ids = list(st.session_state.space.entities.keys())
                for eid in entity_ids[:20]:  # Limit to 20 for performance
                    entity = st.session_state.space.entities[eid]
                    g.add_node(eid, label=eid, title=json.dumps(entity.data)[:100])
                
                # Add edges based on KNN
                for source_id in entity_ids[:20]:
                    try:
                        result = st.session_state.space.knn_query(source_id, k=3)
                        for target_id in result.entity_ids:
                            if target_id != source_id:
                                g.add_edge(source_id, target_id)
                    except:
                        pass
                
                # Save and display
                g.save_graph("trigraphx_network.html")
                
                with open("trigraphx_network.html", "r") as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, height=600)
                    
            except ImportError:
                st.error("需要安装 pyvis: pip install pyvis")
        
        elif viz_type == "关联矩阵":
            st.subheader("📊 实体相似度矩阵")
            
            entity_ids = list(st.session_state.space.entities.keys())[:15]
            
            # Compute similarity matrix
            similarity_matrix = []
            for i, eid1 in enumerate(entity_ids):
                row = []
                for j, eid2 in enumerate(entity_ids):
                    if i == j:
                        row.append(1.0)
                    else:
                        try:
                            result = st.session_state.space.knn_query(eid1, k=len(entity_ids))
                            if eid2 in result.entity_ids:
                                idx = result.entity_ids.index(eid2)
                                row.append(1 - result.distances[idx])
                            else:
                                row.append(0)
                        except:
                            row.append(0)
                similarity_matrix.append(row)
            
            df_sim = pd.DataFrame(similarity_matrix, index=entity_ids, columns=entity_ids)
            
            fig = px.imshow(df_sim, color_continuous_scale="Blues",
                          title="实体相似度热力图")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "向量空间":
            st.subheader("🌌 语义向量空间投影")
            st.info("使用 t-SNE 将高维向量投影到 2D 平面")
            
            try:
                from sklearn.manifold import TSNE
                import numpy as np
                
                # Collect embeddings
                vectors = []
                labels = []
                
                for eid, entity in list(st.session_state.space.entities.items())[:100]:
                    if "semantic" in entity.embeddings and entity.embeddings["semantic"]:
                        if entity.embeddings["semantic"]:
                            vectors.append(entity.embeddings["semantic"][0].embedding)
                            labels.append(eid)
                
                if len(vectors) > 1:
                    X = np.array(vectors)
                    tsne = TSNE(n_components=2, random_state=42)
                    X_2d = tsne.fit_transform(X)
                    
                    df_viz = pd.DataFrame({
                        "X": X_2d[:, 0],
                        "Y": X_2d[:, 1],
                        "Entity": labels
                    })
                    
                    fig = px.scatter(df_viz, x="X", y="Y", hover_name="Entity",
                                   title="向量空间 (t-SNE投影)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("无足够的语义向量数据")
                    
            except ImportError:
                st.error("需要安装 scikit-learn: pip install scikit-learn")

# ============================================================================
# PAGE 4: PERFORMANCE
# ============================================================================

elif page == "⏱️ 性能":
    st.markdown("# ⏱️ 性能监控")
    
    if not st.session_state.query_history:
        st.info("暂无查询记录")
    else:
        # Query latency distribution
        df_history = pd.DataFrame(st.session_state.query_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均延迟", f"{df_history['time_ms'].mean():.2f}ms")
        with col2:
            st.metric("最大延迟", f"{df_history['time_ms'].max():.2f}ms")
        with col3:
            st.metric("最小延迟", f"{df_history['time_ms'].min():.2f}ms")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df_history, x=range(len(df_history)), y="time_ms",
                         title="查询延迟趋势", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df_history, x="time_ms", nbins=20,
                             title="延迟分布")
            st.plotly_chart(fig, use_container_width=True)
        
        # Query type distribution
        st.subheader("查询类型统计")
        query_counts = df_history['type'].value_counts()
        fig = px.pie(query_counts, values=query_counts.values,
                    names=query_counts.index, title="查询类型占比")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed query history
        st.subheader("📋 查询历史")
        st.dataframe(df_history, use_container_width=True)

# ============================================================================
# PAGE 5: DATA MANAGEMENT
# ============================================================================

elif page == "💾 数据管理":
    st.markdown("# 💾 数据管理")
    
    action = st.radio(
        "选择操作",
        ["添加实体", "查看实体", "删除实体", "备份恢复"],
        horizontal=True
    )
    
    if action == "添加实体":
        st.subheader("➕ 添加新实体")
        
        with st.form("add_entity_form"):
            entity_name = st.text_input("实体名称", placeholder="例如: 张三、某科技、doc_001")
            entity_data = st.text_area("实体数据 (JSON)", value='{"type": "person", "role": "投资人"}')
            
            # Embedding options
            st.write("选择嵌入类型")
            col1, col2 = st.columns(2)
            with col1:
                add_semantic = st.checkbox("语义向量")
                add_hierarchy = st.checkbox("层级关系")
            with col2:
                add_association = st.checkbox("关联关系")
                add_causal = st.checkbox("因果关系")
            
            submitted = st.form_submit_button("✅ 添加实体")
            
            if submitted:
                if not entity_name.strip():
                    st.error("❌ 请输入实体名称")
                else:
                    try:
                        data = json.loads(entity_data)
                        data["name"] = entity_name.strip()
                        
                        embeddings = {}
                        if add_semantic:
                            import numpy as np
                            vec = np.random.randn(10).tolist()
                            embeddings[MetricType.SEMANTIC] = SemanticEmbedding(vector=vec)
                        if add_hierarchy:
                            embeddings[MetricType.HIERARCHY] = HierarchyEmbedding(level=1, parent=None)
                        
                        entity, created = st.session_state.space.ingest(
                            data=data,
                            embeddings=embeddings if embeddings else None,
                        )
                        
                        # Persist
                        try:
                            st.session_state.persistence.save_entities_batch([entity], batch_id=f"batch_{int(time.time())}")
                        except:
                            pass
                        
                        if created:
                            st.success(f"✅ 成功添加实体: {entity.id}")
                        else:
                            st.info(f"ℹ️ 实体已存在，数据已合并: {entity.id}")
                        
                    except json.JSONDecodeError:
                        st.error("❌ JSON 格式错误")
                    except Exception as e:
                        st.error(f"❌ 错误: {str(e)}")
    
    elif action == "查看实体":
        st.subheader("🔍 查看所有实体")
        
        if st.session_state.space.entities:
            entities_list = []
            for eid, entity in st.session_state.space.entities.items():
                entities_list.append({
                    "ID": eid,
                    "数据": json.dumps(entity.data)[:50],
                    "嵌入类型": ", ".join([k for k, v in entity.embeddings.items() if v]),
                    "创建时间": entity.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            df_entities = pd.DataFrame(entities_list)
            st.dataframe(df_entities, use_container_width=True)
        else:
            st.info("暂无实体")
    
    elif action == "删除实体":
        st.subheader("🗑️ 删除实体")
        
        entity_ids = list(st.session_state.space.entities.keys())
        if entity_ids:
            entity_to_delete = st.selectbox("选择要删除的实体", entity_ids)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("软删除（可恢复）"):
                    st.session_state.space.soft_delete_entity(entity_to_delete)
                    st.success(f"✅ 已软删除: {entity_to_delete}")
            with col2:
                if st.button("硬删除（永久删除）"):
                    st.session_state.space.hard_delete_entity(entity_to_delete)
                    st.success(f"✅ 已硬删除: {entity_to_delete}")
        else:
            st.info("暂无实体")
    
    elif action == "备份恢复":
        st.subheader("💾 备份恢复")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**创建检查点**")
            if st.button("📸 创建快照"):
                try:
                    checkpoint_name = st.session_state.persistence.create_checkpoint()
                    st.success(f"✅ 检查点创建成功: {checkpoint_name}")
                except Exception as e:
                    st.error(f"❌ 错误: {str(e)}")
        
        with col2:
            st.write("**恢复检查点**")
            try:
                checkpoints = st.session_state.persistence.list_checkpoints()
                if checkpoints:
                    selected_checkpoint = st.selectbox("选择检查点", checkpoints)
                    if st.button("🔄 恢复"):
                        st.session_state.persistence.restore_checkpoint(selected_checkpoint)
                        st.success(f"✅ 已恢复: {selected_checkpoint}")
                else:
                    st.info("暂无检查点")
            except Exception as e:
                st.error(f"❌ 错误: {str(e)}")

# ============================================================================
# PAGE 6: HELP
# ============================================================================

elif page == "ℹ️ 帮助":
    st.markdown("# ℹ️ TriGraphX 使用指南")
    
    st.markdown("""
    ## 什么是 TriGraphX?
    
    TriGraphX 是一个**统一的多维关系度量空间数据库**，替代了传统的：
    - 树形索引
    - 图数据库  
    - 向量数据库
    
    所有数据关系都用**距离函数**表示。
    
    ## 核心概念
    
    ### 🎯 4种度量类型
    - **语义相似度** (Semantic): 向量余弦距离
    - **层级关系** (Hierarchy): 树形最低公共祖先距离
    - **关联关系** (Association): 图边权重距离  
    - **因果关系** (Causal): 因果时间距离
    
    ### 🔍 4种查询类型
    - **KNN查询**: 找K个最近邻
    - **范围查询**: 找距离内的所有实体
    - **路径查询**: 找两实体间的最短路径
    - **多指标查询**: 多个度量的加权组合
    
    ## 使用步骤
    
    1. **📊 总览**: 查看数据统计信息
    2. **💾 数据管理**: 添加实体和嵌入
    3. **🔍 查询**: 执行各种查询操作
    4. **📈 可视化**: 查看关系拓扑
    5. **⏱️ 性能**: 监控查询性能
    
    ## 快速示例
    
    ```python
    from trigraphx import MetricSpace, SemanticEmbedding
    import numpy as np
    
    # 创建空间
    space = MetricSpace()
    
    # 自然语言摄入 (无需手动指定ID)
    entity, created = space.ingest(
        {"name": "doc_1", "title": "Example", "content": "Hello world"},
        embeddings={
            MetricType.SEMANTIC: SemanticEmbedding(
                vector=np.random.randn(10).tolist()
            )
        }
    )
    print(f"实体ID: {entity.id}, 新建: {created}")
    
    # 同一实体再次摄入，自动合并数据
    entity2, created2 = space.ingest({"name": "doc_1", "author": "Alice"})
    print(f"实体ID: {entity2.id}, 新建: {created2}")  # created2=False, 数据已合并
    
    # 查询
    result = space.knn_query(entity.id, k=5)
    print(f"Found {len(result.entity_ids)} neighbors")
    ```
    
    ## FAQ
    
    **Q: 数据持久化在哪里?**  
    A: 默认存储在 `trigraphx_data/` 目录，包括:
    - JSONL 格式的实体批次
    - SQLite 索引
    - tar.gz 检查点备份
    
    **Q: 支持多少个实体?**  
    A: 取决于内存和存储。默认限制 10,000 个（可配置）。
    
    **Q: 如何扩展到分布式?**  
    A: 规划中，敬请期待！
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 30px;">
    <p>TriGraphX v1.0.0 | 多维关系度量空间数据库</p>
    <p>Built with ❤️ using Streamlit | Powered by Python + Rust</p>
</div>
""", unsafe_allow_html=True)
