#!/bin/bash

# Zotero存储文件夹的路径
ZOTERO_STORAGE_PATH="/Users/qaz1214/Downloads/ref/ref-to-extract"

# 所有PDF文件将被复制到这个文件夹
OUTPUT_FOLDER_PATH="/Users/qaz1214/Downloads/ref//ref-to-extract/misc"

# 创建输出文件夹，如果它不存在的话
mkdir -p "$OUTPUT_FOLDER_PATH"

# 查找并复制所有的PDF文件
find "$ZOTERO_STORAGE_PATH" -name '*.pdf' -exec cp {} "$OUTPUT_FOLDER_PATH" \;

echo "所有的PDF文件已经被复制到 $OUTPUT_FOLDER_PATH"

# 设定视频目录路径
VIDEO_DIR="data/video"

# 每个分割文件的大小
SPLIT_SIZE="50m"

# 切分视频目录下的所有.mov文件
for file in "$VIDEO_DIR"/*.mov; do
  # 只有文件大小超过阈值时才切分文件
  if [ $(stat -c%s "$file") -gt $((50*1024*1024)) ]; then
    echo "正在切分文件: $file"
    # 使用split命令切分文件
    split -b $SPLIT_SIZE "$file" "${file%.mov}_part_"
  else
    echo "文件 $file 小于切分大小阈值，跳过切分"
  fi
done

echo "所有文件切分完成。"
