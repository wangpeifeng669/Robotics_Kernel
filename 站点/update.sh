#!/bin/bash
# 拉取失败或生成失败时停住，线上 index.html 保持上一份可用页面。
set -euo pipefail

REPO=/opt/Robotics_Kernel
WEB_DIR=/var/www/kernel-docs
OUT="${WEB_DIR}/index.html"

mkdir -p "${WEB_DIR}"
cd "${REPO}"
/usr/bin/git fetch origin
/usr/bin/git merge --ff-only origin/main
/usr/bin/python3.6 "${REPO}/站点/build_site.py" "${OUT}"
