# 操作细节

## 校验配置（可选）

技能目录带了与服务端同规则的自检脚本：

```bash
python <技能目录>/scripts/validate_label_config.py <label_config.xml> --adapter <adapter>
```

声明了 field_map 时一并传入：`--field-map <field_map.json>`。

`<技能目录>` 必须是**绝对路径** —— 技能目录在 workspace 之外，相对路径
`.agents/skills/...` 只在 read/write/edit 工具里有效，terminal 里会找不到文件。
技能目录与 workspace 根同址，直接用 `ls "$PWD/.agents/skills/label-studio/scripts/"`
确认即可，不要全盘 `find`。
路径不确定就别折腾：直接跳到 create，它会在转换**之前**做同样的校验并返回可读
错误，结论一致；这一步只省一次全量转换的时间。退出码：0 通过，1 不通过
（打印原因），2 用法错误。

## 修改已有项目

`label_studio_project(operation="get", project_ref=...)` 获取当前配置 → 在
workspace 中修改 XML → 校验 → `operation="update_config"`（带
`expected_config_version`）。如果新 XML 删除了已有标注使用的控件名，工具会拒绝
更新。

## 导出标注结果

`label_studio_project(operation="export", project_ref=...)`。工具自动把标注结果
转回 PyroMind 格式并保存到用户 Storage，并按该项目建立时的绑定把控件读回字段。

导出之后要转训练格式时，走 data-processing 的 format-conversion 范式；训练
schema 不在本 skill 内定义。

## 图片链接

任务数据里的图片地址是**导入时签发的**，Label Studio 不会重签；但 portal 对已
签发的地址**不设到期时间**：只要该地址指向的账号仍然可用，旧项目里的图就一直是
可以渲染的。

因此**不要**向用户提"图片有效期""多久后失效""需要刷新"这类内容 —— 不存在这件
事。`refresh_media` 仍然保留（原地重签，任务 ID、标注与预测都不受影响，重复
执行也安全），但它是可选的维护动作，不必按期执行。

## 没有删除操作

工具**不提供删除项目**。建错了、或者用户要推倒重来，只能在 Label Studio 界面里
删（把 `open_url` 给用户，让他自己删）。

如果只是参数写错，先用同参数重跑 create（幂等，会复用已有项目），不要试图"删了
重建"。确实需要另一个项目时，传一个不同的 `idempotency_key`。
