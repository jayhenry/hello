
# Step1. 配置好 .vscode/launch.json，然后在vscode中打开 Run and Debug面板，选择并开启 attach mode的Debug

# Step2. Run the script
cd $(dirname $0)
python -m debugpy --connect 5678 $(which torchrun) --nproc-per-node 4 dist_demo.py

