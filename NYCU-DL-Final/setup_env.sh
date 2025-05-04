#!/bin/bash

ENV_NAME="TradeMaster"

echo " 建立 Conda 環境中..."
conda env create -f environment.yml

echo " Conda 環境已建立，請確認啟動成功"

echo "請手動執行：conda activate $ENV_NAME"

echo "📝 開始覆蓋自定義修改套件..."

# CUSTOM_DIR="./custom_packages"
# SITE_PACKAGES_DIR="$HOME/.conda/envs/$ENV_NAME/lib/python3.10/site-packages"

# cp -r $CUSTOM_DIR/torch/ $SITE_PACKAGES_DIR/torch/
# cp -r $CUSTOM_DIR/diffusers/ $SITE_PACKAGES_DIR/diffusers/

echo "✅ 環境與套件覆蓋完成！"
