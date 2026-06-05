#!/bin/bash
# TriGraphX UI Quick Start Script
# 快速启动 Streamlit 仪表板

echo "=========================================="
echo "  TriGraphX - Interactive Dashboard"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install streamlit plotly pandas pyvis scikit-learn -q
    echo "✅ Dependencies installed"
fi

echo ""
echo "🚀 Starting TriGraphX Dashboard..."
echo ""
echo "📱 UI Address:"
echo "   Local:  http://localhost:8501"
echo "   Remote: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "📖 Documentation:"
echo "   - README_UI.md (this directory)"
echo "   - TriGraphX_DATABASE_MODEL.md (architecture)"
echo ""
echo "⚠️  Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Start Streamlit
streamlit run ui_streamlit.py
