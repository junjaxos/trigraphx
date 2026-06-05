"""
TriGraphX UI System - Complete Overview
完整的可视化系统架构
"""

# 📊 TriGraphX UI 系统架构与功能

## 🏗️ 项目结构

```
trigraphx/
├── ui_streamlit.py           [28 KB] Streamlit 应用主程序
├── run_ui.sh                 [1 KB]  启动脚本
├── README_UI.md              [6 KB]  UI 使用指南
├── requirements.txt          已更新  包含 UI 依赖
│
├── trigraphx/                [核心包]
│   ├── __init__.py
│   ├── entity.py
│   ├── space.py
│   ├── persistence.py
│   └── enterprise.py
│
└── trigraphx_rust/           [Rust 加速]
```

## 🎯 核心功能

### 1️⃣ 📊 总览 (Overview)
展示系统整体状态的仪表板
- 实体总数统计
- 查询次数计数
- 存储大小监控
- 度量类型分布（饼图）
- 存储分布分析（柱状图）
- 系统配置信息

### 2️⃣ 🔍 查询 (Query)
4种交互式查询界面
- **KNN查询**: 
  - 选择源实体
  - 设置 K 值
  - 选择度量类型
  - 结果表格 + 可视化

- **范围查询**:
  - 源实体选择
  - 半径滑块调整
  - 距离分布直方图

- **路径查询**:
  - 起点和终点选择
  - 路径长度显示
  - 最短路径结果

- **多指标查询**:
  - 权重滑块配置
  - 4种度量加权组合
  - 综合评分显示

### 3️⃣ 📈 可视化 (Visualization)
3种数据关系展示方式
- **网络拓扑** (PyVis)
  - 交互式节点图
  - 拖拽和缩放
  - KNN 边连接

- **关联矩阵** (Heatmap)
  - 相似度热力图
  - 颜色深浅表示相似度
  - 快速发现相似实体

- **向量空间** (t-SNE)
  - 高维向量投影
  - 2D 平面展示
  - 聚类模式识别

### 4️⃣ ⏱️ 性能 (Performance)
实时性能监控和分析
- **延迟统计**:
  - 平均、最大、最小延迟
  - 单位: 毫秒 (ms)

- **延迟趋势**:
  - 时间序列图表
  - 性能走势分析

- **延迟分布**:
  - 直方图展示
  - 20个 bin 细粒度分析

- **查询类型占比**:
  - 饼图展示
  - 各类型查询统计

- **查询历史**:
  - 完整的查询记录表
  - 时间戳和耗时记录

### 5️⃣ 💾 数据管理 (Data Management)
完整的数据生命周期管理
- **添加实体**:
  - 自定义实体 ID
  - JSON 格式数据输入
  - 多种嵌入类型选择:
    ✓ 语义向量 (Semantic)
    ✓ 层级关系 (Hierarchy)
    ✓ 关联关系 (Association)
    ✓ 因果关系 (Causal)

- **查看实体**:
  - 表格列表显示
  - 数据预览
  - 嵌入类型展示
  - 创建时间记录

- **删除实体**:
  - 软删除 (Soft Delete)
    → 可通过备份恢复
  - 硬删除 (Hard Delete)
    → 永久删除

- **备份恢复**:
  - 创建快照 (Checkpoint)
  - 列表查看检查点
  - 一键恢复数据

### 6️⃣ ℹ️ 帮助 (Help)
完整的文档和使用指南
- TriGraphX 概念介绍
- 4种度量类型说明
- 4种查询类型说明
- 快速代码示例
- FAQ 常见问题

## 🚀 快速开始

### 方式 1: 使用启动脚本
```bash
./run_ui.sh
```

### 方式 2: 直接运行
```bash
streamlit run ui_streamlit.py
```

### 方式 3: 指定端口
```bash
streamlit run ui_streamlit.py --server.port 9000
```

### 方式 4: 远程访问
```bash
streamlit run ui_streamlit.py --server.address 0.0.0.0 --server.port 8080
```

## 📋 依赖清单

| 包名 | 版本 | 用途 |
|------|------|------|
| streamlit | ≥1.28.0 | Web UI 框架 |
| plotly | ≥5.17.0 | 交互式图表 |
| pandas | ≥1.5.0 | 数据处理 |
| pyvis | ≥0.3.0 | 网络拓扑 |
| scikit-learn | ≥1.0.0 | t-SNE 投影 |
| numpy | ≥1.20.0 | 数值计算 |
| trigraphx | local | 核心库 |

## 🎨 UI 组件

### Sidebar（左侧栏）
```
🎯 TriGraphX Dashboard
─────────────────────
[📊 总览]
[🔍 查询]
[📈 可视化]
[⏱️ 性能]
[💾 数据管理]
[ℹ️ 帮助]

📈 实时统计
─────────────
实体数量: 0
查询次数: 0
```

### 主内容区
- 响应式栅栏布局
- 2-3 列自适应
- 实时更新
- 交互式控件

### 数据表格
- 可排序列
- 数据预览
- 高度优化

### 图表组件
- Plotly 交互式图表
- 放大、缩放、导出
- 实时更新

## 💡 高级特性

### 会话状态管理
```python
st.session_state.space          # MetricSpace 对象
st.session_state.persistence   # PersistenceLayer 对象
st.session_state.query_history # 查询历史列表
st.session_state.metrics_collector  # 性能指标
```

### 交互式表单
- 实体添加表单
- 嵌入类型多选
- JSON 数据验证

### 错误处理
- try/except 异常捕获
- 友好的错误提示
- 堆栈跟踪显示

## 📊 性能参数

| 指标 | 值 | 说明 |
|------|-----|------|
| 最大实体数 | 10,000 | 可配置 |
| 网络拓扑限制 | 20 | 性能考虑 |
| 相似度矩阵限制 | 15 | 热力图大小 |
| KNN 默认 K | 3 | 可调整 |
| 范围查询默认半径 | 0.5 | 可调整 |

## 🔧 配置

### 修改端口
编辑 `run_ui.sh`:
```bash
streamlit run ui_streamlit.py --server.port YOUR_PORT
```

### 修改数据目录
编辑 `ui_streamlit.py`:
```python
db_root = Path("your_data_dir")  # 第 30 行
```

### 修改主题
启动后在浏览器菜单:
Settings → Theme → 选择主题

## 🌐 访问地址

| 环境 | 地址 | 说明 |
|------|------|------|
| 本地 | http://localhost:8501 | 默认端口 |
| 本机 LAN | http://192.168.x.x:8501 | 替换为实际 IP |
| 远程服务器 | http://server:8080 | 需要 0.0.0.0 |

## 📱 浏览器兼容性

✅ 推荐:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

⚠️ 部分支持:
- IE 11 (不支持某些功能)

## 🐛 常见问题

**Q: 导入错误 "No module named 'trigraphx'"?**
A: 确保在项目根目录运行，或修改导入为相对路径

**Q: Streamlit 启动很慢?**
A: 首次启动会编译 Cython，后续会快很多

**Q: PyVis 图显示空白?**
A: 确保安装了 pyvis，或使用 `pip install pyvis --upgrade`

**Q: 数据不保存?**
A: 检查 trigraphx_data 目录权限

## 📈 扩展建议

### 短期 (1-2周)
- [ ] 导入/导出 CSV
- [ ] 数据过滤和搜索
- [ ] 自定义查询脚本

### 中期 (1-2月)
- [ ] REST API 层
- [ ] 实时数据流
- [ ] Kafka 集成

### 长期 (3-6月)
- [ ] 多用户支持
- [ ] 权限管理 UI
- [ ] 分布式查询

## 📚 相关文档

- [TriGraphX 数据库模型](TriGraphX_DATABASE_MODEL.md)
- [快速入门指南](TriGraphX_QUICK_START.md)
- [主 README](README.md)
- [实现日志](IMPLEMENTATION_LOG.md)

## 📞 支持

遇到问题？
1. 检查 README_UI.md
2. 查看代码注释
3. 查阅帮助页面

---

**TriGraphX UI System v1.0.0**
*Interactive Data Visualization for Metric Space Database*
