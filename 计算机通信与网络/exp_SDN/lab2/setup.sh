# Lab2 环境设置脚本
# 用法：source setup.sh 或 . setup.sh

echo "🔧 正在同步项目依赖..."
uv sync

if [ $? -eq 0 ]; then
    echo "✅ 依赖同步成功"
    echo "🚀 正在激活虚拟环境..."
    source .venv/bin/activate
    
    if [ $? -eq 0 ]; then
        echo "✔️ 虚拟环境已激活"
        echo "📦 当前 Python: $(which python3)"
        echo "📁 当前目录: $(pwd)"
        echo "💡 提示：退出虚拟环境请输入：deactivate"
        echo "🎯 可用命令："
        echo "  - 运行控制器：osken-manager <script_name>.py"
        echo "  - 运行拓扑：  sudo ./topo_1.py 或 sudo ./topo_2.py"
    else
        echo "❌ 虚拟环境激活失败"
    fi
else
    echo "❌ 依赖同步失败，请检查网络连接或 uv 配置"
fi