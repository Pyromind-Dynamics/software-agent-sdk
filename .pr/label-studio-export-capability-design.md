# Label Studio 导出落盘：能力票方案（C）

> 标注员点一次 Export，原始 JSON 就落到**该用户自己的存储**。
> **Label Studio 侧不持有任何密钥、任何共享值、任何存储凭证。**

状态：portal ✅ / SDK ✅ / LS 插件与 yaml ✅ —— 三处各自的测试与真机干跑见文末。

> **2026-09-14 首次真机上线后追加**：插件装上了但每次导出都静默丢弃，两个叠加的部署侧原因
> ——① 钩点层选错（`generate_export_file` 只见转换后产物，`download_resources` 默认 `True`
> 导致判断恒真）；② Cloudflare 字面封了 urllib 的默认 UA。均已修复并在 pre 上验证落盘成功。
> 细节与实测数据见 `label-studio-raw/export-hook-rollout.md` 第 0 节。

## 为什么不是"共享密钥 + 邮箱头"（A）也不是"用户的 LS token"（B）

| | A 共享密钥 | B 用用户 LS token | **C 能力票** |
|---|---|---|---|
| 属主来源 | 请求头声称 + 密钥背书 | 回调 LS 推导 | **签名固化** |
| LS 侧是否留秘密 | 留一个共享密钥 | 留（用户的 LS token 在库里） | **一个都不留** |
| 请求方可选的字段 | 属主、路径、cluster | 属主（推导） | **无（全在签名里）** |
| 凭证泄露半径 | 任意用户的固定 key | 该用户整个 LS 账号 | **一个 object key** |
| 新增依赖 | 无 | 每次推送回调 LS | 无 |
| 需要两边保持一致的配置 | 有（密钥） | 无 | **无** |

## 票的定义

一张 portal 签发、portal 校验的**能力票**：谁拿着它，就能让 portal 把一段 JSON 写到票面上
固定的那**一个** object key。它不是身份凭证（不能登录）、不是密钥（不能签别的票）。

载体沿用本模块里 media token 已有的做法 —— **不是 JWT**，而是同一族的紧凑签名载荷：

```
v1.<base64url(payload)>.<base64url(HMAC-SHA256(encoded_payload, secret))>
```

```json
{"aud": "label-studio-export-v1",
 "uid": 123456,                                    // 属主：写到谁的存储
 "path": "/.pyromind-agent/label-studio/<ref>/export/label_studio_export.json",
 "cluster": "us-west-1#pre",                       // 用哪个环境的存储后端
 "project_ref": "<ref>",                           // 仅用于日志/审计，不参与授权
 "iat": 1757856000, "exp": 1773408000}             // 180 天
```

**签名密钥**：`derive_label_studio_purpose_secret(sso_secret, "label-studio-export-signing-v1")`
—— 从 portal 的 SSO 秘密按用途派生出的子密钥。这样做的两个好处：

1. **域隔离是结构性的**：票与登录 JWT、media token 的签名密钥**互不相同**，所以
   "一张票被当成登录态"或"一个 media token 被当成票"在验签那一步就失败，
   不需要在 `verify_jwt_token` 里加类型黑名单（那种做法只是碰巧依赖字段缺失，很脆）。
2. **零新增配置**：不从任何地方读新的 secret，生产环境本来就没有 SSO 秘密时派生会直接
   抛错，于是"没配置 = 签不出票 = 全部 401"，与既有失败关闭的语义一致。

**校验流程**（`verify_label_studio_export_ticket`）：版本 → 常量时间比对签名 → 解载荷 →
校验 `aud` → 校验 `uid` 类型 → **再次用 object key 正则校验 `path`** → 校验 `iat/exp`
（含 60s leeway）→ 校验 `cluster` 形状。路由随后只从票面取 `uid/key/cluster`。

## 两段链路

### 第一段：建项目时签发（一次性）

```
agent ──create(cluster)──▶ SDK
SDK ──POST {portal}/label_studio/export_token（带用户自己的 portal 凭证）──▶ portal
        body: {"project_ref": "<ref>", "cluster": "us-west-1#pre"}
                                    │
                                    ├─ _require_active_user() → uid 取自认证结果（不是参数）
                                    ├─ _required_storage_cluster() → 只接受 allowlist 里的 cluster
                                    ├─ export_object_key() → 自己拼出唯一的 object key
                                    └─ create_label_studio_export_ticket() → 签票
SDK ◀── {"token", "path", "cluster", "expires_at"} ─┘
SDK ──PATCH /api/projects/<id>──▶ Label Studio
        description = 头部行 + 路径行 + "export-token: <票>" + 最后导出行(可选)
```

调用方在这条链路上**只能指定 project_ref 和 cluster**，属主和路径都由 portal 决定。

### 第二段：每次导出

```
标注员点 Export ──▶ LS: DataExport.save_export_files（插件已包住）
                        ├─ 从 description 里取 "export-token: ..."（取不到 → warning + 跳过）
                        └─ 后台线程 POST，只带三样：
                             Content-Type: application/json
                             User-Agent: PyroMind-LabelStudio-Export/1.0   ← 不带会被 Cloudflare 403
                             X-Pyromind-Export-Token: <票>
                             body: 原始 JSON
                                │
                        portal /label_studio/export_upload
                          ├─ 验签 → uid / key / cluster 全部取自票面
                          ├─ 以该 uid 的身份取 delegated credential
                          └─ presigned_upload_url → PUT → 200 {"path", "bytes"}
```

包在 `save_export_files` 而不是 `generate_export_file` 上，是因为前者拿到的 `data`
是 LS 写进自己 `EXPORT_DIR` 的**原始 JSON**（在 converter 之前）；包在后者上只能看到
转换后的产物，而 `CONVERTER_DOWNLOAD_RESOURCES` 默认 `True` 时那个判断会误伤
（详见 `label-studio-raw/export-hook-rollout.md` 第 0 节）。`Export.save_file`
（Export API 的延迟导出）另有一条独立的包装，两条路不重叠。

请求里**没有任何可声称的字段**：没有用户头、没有路径头、没有 cluster 头。
即使 LS 被完全攻陷，它能做的也只是把一个 JSON 写到它自己项目简介里那张票对应的
那一个 object key，换不了用户，也换不了路径。

顺带简化掉的：插件不再需要 owner 邮箱（原来要把 `project.created_by.email` 发给 portal
→ 多一个"邮箱必须在 portal 侧查得到"的失败点），LS 也不再需要自己配一份 cluster
（原来两份 cluster 有可能配漂 → 导出和媒体落到不同存储）。

## 谁等谁

| 时刻 | 事件 | 谁等谁 | 与改造前相比 |
|---|---|---|---|
| T0 | agent 建项目 | agent → SDK → portal → LS | 多一次 portal 往返（毫秒级） |
| T0′ | 写简介（路径 + 票） | SDK → LS | 多一行（仍然一次 PATCH） |
| T1 | 标注员点 Export | 标注员 → LS | 无变化 |
| T1+ε | 推送 | **后台线程**，标注员**不等** | 无变化 |
| T2 | 落盘完成 | — | 无变化 |

## 过期、失败与吊销

| 情形 | 表现 | 处理 |
|---|---|---|
| 老项目简介里没有票 | 插件 warning + 跳过 | 重跑一次 agent 操作（create/export）即补上 |
| 票过期（180 天） | portal 401，插件 warning | 同上（每次写简介都会续签） |
| 用户把简介改坏 | 同上 | 自伤；uid/key 在签名内，改不宽范围 |
| portal 不可达 | 插件 warning | 标注无感；下次导出再试 |
| 票里的 cluster 不在 allowlist | portal 400，插件 warning | 检查签发时的 cluster |
| SDK 取票失败 | 简介只写路径，不写票；建项目**照常成功** | 标注绝不被 portal 故障阻塞 |

**吊销**：无状态校验，没有单张票的黑名单 —— 删掉项目简介里那一行即立即失效
（或等过期）。轮换 SSO 秘密会让所有历史票失效，但也会影响其他链路，不作为手段。

## 改动清单

| 仓库 | 改动 |
|---|---|
| portal | 新增 `POST /label_studio/export_token`；`/export_upload` 只收票 + body；票的签发/校验 helper（复用 media token 的编解码与用途派生密钥）；`LabelStudioConfig.export_signing_secret` 取代 `export_secret` |
| SDK | 新增 `export_ticket.py`（`PortalExportTicketProvider`）；`_project_description` 增加票行；create / export 写简介前取票（失败则降级为不写票行） |
| LS `app-deploy.yaml` | 新增 `PYROMIND_EXPORT_PORTAL`；插件改为读简介里的票、只发一个自定义头（另有 `Content-Type` 与自报家门的 `User-Agent`）；**删除** `PYROMIND_EXPORT_SECRET` 与 `PYROMIND_EXPORT_CLUSTER` |

## 验证

| 层 | 证据 |
|---|---|
| portal 单元 | `tests/webapp/test_label_studio.py` 65 passed；含"票能过上传路由"、"跨用途密钥互不通用（两个方向）"、"路由只认 `X-Pyromind-Export-Token`" |
| SDK 单元 | `tests/tools/label_studio` 111 passed（`test_export_ticket.py` 9 项覆盖四种失败模式；`test_executor.py` 断言票行紧跟路径行、portal 故障不致命、export 会续签）；4 个文件 pre-commit 全绿（含 pyright） |
| 插件脚本级 | 从 yaml 里取出插件正文（逐字节比对）→ 两条导出路径各推一次；断言请求头**只有** `content-type`、`user-agent` 与 `x-pyromind-export-token`；原实现仍读到完整内容；4 种静默跳过 |
| 真机干跑 | 在 pre 的 `label-studio-0` 里以独立进程 `install()`：两个方法都被包住且绑定不变；构造出的请求 URL/头/正文与上面一致；无票的项目不推送（该次运行不开 socket，现场已清理） |
| 真机端到端 | 2026-09-14 修复后：8 号项目 684 任务走 UI 同参数导出 → `INFO Stored the export of project 8 at /.pyromind-agent/label-studio/339f08f6-.../export/label_studio_export.json (1470545 bytes, status 200)`；离线对照台在新旧两版上 14+ 用例，旧版 0 次上传 / 新版 1 次且不重复推 |
| 三处契约联验 | portal 真实签发 → SDK 真实 `_project_description` → 插件真实正则（取自 yaml）→ portal 真实验签：票与路径逐字节一致，uid/180 天有效期对得上 |
