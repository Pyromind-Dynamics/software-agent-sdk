# LoadImage 节点

![alt text](/imgs/LoadImage/LoadImage.png) 

## 1.1 功能概述

从 `/workspace/image_cache/` 目录下读取所选的图片，并输出路径。

支持的图片格式：`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.svg`

## 1.2 输入类型

| 参数  | 数据类型      | 必填 | 描述 |
|-------|---------------|----|------|
| image | STRING | 是 | 从 `/workspace/image_cache/` 中动态读取的图片文件列表。用户通过下拉菜单选择所需图片。支持 `image_upload` 扩展（允许用户上传新图片）。默认值：列表中第一个图片路径。 |
## 1.3 输出类型

| 参数            | 数据类型 | 描述 |
|-----------------|----------|------|
| image_file_path | STRING | 图片在工作区中的完整路径，格式为 `/workspace/{所选图片路径}`。供下游节点（如 RL 训练、Rollout 等）读取图片内容使用。当前无路径存在性校验。 |
