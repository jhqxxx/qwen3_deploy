#!/bin/bash

PROJECT_DIR="$(pwd)"
TARGET_DIR="${PROJECT_DIR}/target/release"
PACKAGE_DIR="package"
OUTPUT_FILE="${PACKAGE_DIR}/deploy.tar.gz"
REMOTE="wxg@192.168.1.242"
REMOTE_RELEASE_DIR="/home/pub/packages/releases/fs-kb-app/models/deploy/qwen3/windows"

cargo build -r

mkdir -p "$PACKAGE_DIR"
tar -czvf "${OUTPUT_FILE}" -C "${TARGET_DIR}" "deploy.exe"

ssh ${REMOTE} "mkdir -p ${REMOTE_RELEASE_DIR}"
scp  "${OUTPUT_FILE}" "${REMOTE}:${REMOTE_RELEASE_DIR}/deploy.tar.gz"
echo "Package success"