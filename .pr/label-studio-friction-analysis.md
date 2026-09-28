# Label Studio 预打标流程「费劲」的根因分析

分析对象：
- 线上对话 `36313db921b749bbaa2a873f34cac6ca`（webapp-pre / software-agent-sdk-0）
- 本地对话 `47e829f8c3ee4bd6bb37a71752bb8f23`（workspace/conversations/）

两者是同一条流水线的上下游：线上对话做数据准备（产出 `processed.jsonl` / `source_manifest.jsonl` / `vlm_images`），本地对话消费这份产物做格式转换 + 导入 LS。

---

## 一、先用事实说话

| | 线上对话（数据准备→导 LS） | 本地对话（导 LS + 参考图） |
|---|---|---|
| 时间跨度 | 06:07 → 10:57（**4h50m**） | 11:17 → 12:04（47m） |
| 事件数 | 43,290 | 14,883 |
| 工具调用 | 294 | 178 |
| 其中单文件上传 | **119** | **50** |
| 其中 preview_dataset | 41 | 44 |
| 其中 terminal | 63 | 47 |
| 失败事件 | **17** | 8 |

单文件上传合计 **169 次** —— 这是"费劲"最直观的量化表征。

### 用户在这一天里实际说的话（原文）

线上：

1. 06:15 「你就跑 100 个 case 吧」
2. 07:10 「我看好像跑完了？**但是任务还没停止？**」
3. 07:29 「**说中文**」← 模型在用英文自言自语
4. 08:14 / 08:17（**同一句话连发两遍**）「**你这都在处理啥，bmp 是可以导入 label studio 的啊**」
5. 08:47 「**你给我说一下，你导出 label studio 的卡点是什么**」
6. 09:22 「There was an issue loading URL from `$gt_image` value ... **有的图片加载失败**」
7. 09:49 「**为什么预打标的缺陷区域标注框选，没有在 label studio 中直接渲染出来**」
8. 10:28 「**你怎么样才会渲染出来**」
9. 10:30 「**为什么一定要增强适配器，你可以不按照这个来吧，核心是要渲染的 LS 有缺陷框渲染**」

本地：

1. 11:17 「预打标的 box 缺陷渲染不要漏了，先本地做数据格式转换，快速打样…验证后再全量处理」
2. 11:40 「**为什么 LS 中没有参考图**」
3. 11:47 「重试一下」
4. 11:55 「**你参考样本怎么来的**」
5. 11:56 「你导一些数据内有提供 .cam 的 case 到 LS 中吧，搞个 10 条」
6. 11:58 「记得导这些有 cam 的打完标的数据啊」

其中 4 个问句是**同一种困惑**：「我要的东西为什么没出现」，而模型的回答都不是「工具没这个能力」，而是「我再试试」。

---

## 二、为什么费劲 —— 按杠杆大小排序

### 根因 1（最大）：能力缺失被误判成「用法不对」，于是只能穷举试探

线上对话的核心诉求是「预打标框要在 LS 里渲染」。agent 的处理路径是：

- 怀疑 meta 字段名不对 → 造 6 个探测样本（`_ls_probe`，p1–p6）覆盖 `label` / `rectanglelabels` / `xywh` / `bbox` 等写法 → **全败**
- 再怀疑是 `original_width/height` 或 `value` 嵌套 → 造第二轮探测（`_ls_probe2`，p7–p9）→ 还是败
- 中间原话：「**我已经试了 6 种，都不行**」「技能目录里没有 `meta_vlm.json` 的输入样例，只有 task-manifest（导出格式）」

而事实是：**当时服务端的 adapter 根本不具备生成矩形预标注的能力。**

证据（代码考古）：

```
git log -S'_region_results' -- openhands-tools/openhands/tools/label_studio/converter.py
git log -S'finding.get("category")' -- openhands-tools/openhands/tools/label_studio/converter.py
→ 两处都只命中 c2576200（09-17 20:01「打标问题优化」）
```

矩形预标注能力的引入提交是 `c2576200` —— **当天 20:01**，也就是线上对话结束 9 小时之后。

所以 agent 的探测从第一发就注定失败：它在试图让一个不支持矩形的 adapter 产出矩形。它无法知道这一点，因为**没有任何地方声明「工具能做什么」**，它只能通过穷举反推能力边界。每轮探测要重传 18 个文件、重建项目 —— 119 次上传里有相当一部分是这么烧掉的。

### 根因 2：输入契约未文档化，字段名靠猜

线上对话时期（`b6f00d39`，126 行版本）的 SKILL.md 里：

- 第 49 行提到了控件名 `finding_category`
- 第 83 行写着「`aoi_export`：样本目录含 `meta.json`（整图判定字段，**无 bbox**）」
- **关于 meta 里 finding 该写哪个字段（`category`）、坐标怎么写，一个字都没有**

于是出现了这段「内心独白」（本地对话 11:40–11:41，用户可见）：

> "let me reconsider... which one is the actual reference (参考图)?"
> "`gt.jpg`: 4.6KB (much smaller → this is the **cam/reference design image**, not a copy of the defect!)"
> "Wait, but for A-type... Hmm. That contradicts..."

agent 在用**文件大小**反推图片语义。这不是模型笨，是文档里没有它要的信息。

### 根因 3：图片槽位硬编码三件套，业务语义无处安放

`converter.py:49-53` 定义 `_MEDIA_FIELDS = (defect_image→defect.jpg, diff_image→diff.jpg, gt_image→gt.jpg)`；
`converter.py:394-398` 对这三个槽位**强制要求文件存在**，缺一个直接抛：

```
ConversionError(f"No media URL resolved for {object_path}")
```

后果链条：

- 源数据真实存在两种样本：A 类（只有缺陷图）、B 类（缺陷图 + `*_cam.bmp` 参考图，617 条里 251 条有 `reference_image`）
- 代码只认 defect/diff/gt 三个槽位，**没有「参考图」这个角色**
- B 类样本的参考图要人肉映射到 `gt.jpg`；A 类没有参考图，但槽位不能空 → **只能拿缺陷图复制一份顶上**
- 用户看到的就是「LS 里没有参考图」/「你参考样本怎么来的」

这是本地对话 47 分钟的主要消耗点，也是 `$gt_image` 加载失败（用户 09:22 贴的报错）的同一根因。

### 根因 4：没有批量能力，所有操作都是单文件

- `upload_file_to_pyromind`：线上 119 次 + 本地 50 次 = **169 次**
- 每次 schema 试探都要「重传 18 个文件 → 重建项目 → 查 predictions」
- 与之配套的禁止条款（SKILL.md「由 Agent 遍历完整数据集或生成 Manifest」）本意是防错，但结果是 agent 连"批量"这条替代路径也没有

### 根因 5：preview_dataset 60 秒硬超时

线上 4 次、本地 1 次失败，错误文本固定：

```
preview_dataset timed out after 60 seconds. Narrow dataset_path to a specific file or a smaller directory.
```

另有 1 次参数校验失败：`Input should be less than or equal to 100 [input_value=106]`。

大目录不可预览 ⇒ agent 无法「先看清数据」，只能边做边试。

### 根因 6：执行环境的摩擦叠加

| 现象 | 证据 |
|---|---|
| `/tmp` 只读 | `cp: cannot create regular file '/tmp/cam_test.jsonl': Read-only file system` |
| heredoc 被禁 | `/bin/bash: cannot create temp file for here document: Operation not permitted` |
| 沙箱不存在 | `Sandbox not found`（`sandbox_terminal` 连续 2 次） |
| terminal 里看不到 `.agents` | `ls .agents` → exit 1 |
| 技能脚本相对路径失灵 | skill 自己在第 34-36 行承认「相对路径只在 read/write/edit 有效，terminal 里会找不到」 |

这些都逼着 agent 把逻辑写成一行行拼接的命令，出错率高（`SyntaxError: unexpected character after line continuation character` 出现 2 次，都是 `\!` 转义问题）。

---

## 三、工程侧优化建议（按投入产出比）

| 优先级 | 动作 | 解决的根因 | 说明 |
|---|---|---|---|
| **P0** | **批量上传/批量 materialize 接口** | 根因 4 | 169 次单文件上传是最大的一笔浪费。一次调用传整目录 |
| **P0** | **给每个 adapter 配可运行的最小样例**（`meta_vlm.json` + 3 张图 + 期望 predictions） | 根因 2 | 纯文档/样例工作，不用改代码，直接消灭"盲试 schema" |
| **P0** | **能力矩阵文档**：字段 → 是否生成预标注 → 落到哪个控件 | 根因 1 | 让 agent 一次判断"这工具能不能做"，而不是穷举 |
| P1 | preview_dataset 分批/流式，或超时可配 | 根因 5 | 大目录至少能"抽样看头 N 条" |
| P1 | 图片角色数据驱动（meta 里声明 `images: [{role, path}]`），取消 defect/diff/gt 三槽位强制 | 根因 3 | A/B 类样本天然支持，参考图有正规位置 |
| P2 | 放开 `/tmp` 写权限与 heredoc | 根因 6 | 让 agent 能正常写临时脚本，而不是拼一行命令 |

---

## 四、Skill 设计有没有缺陷 —— 有，而且不是"该不该设契约"的问题

结论：**契约该有，但现在的问题是把「业务语义」压扁成了「固定枚举」，同时又不声明能力边界。** 具体五条：

### 1. 契约只覆盖"名字"，不覆盖"语义与映射"

- 现有契约：控件必须叫 `finding_category` / `quality_label` / `defect_image`（硬约束，配套服务端校验）
- 缺失契约：**业务概念 → 控件的映射规则**。「参考图」该进哪个槽位？「缺陷框」的类别字段叫什么？
- 于是「不要改名」这条硬约束保护了渲染，却对"用户想看到参考图"这类**新需求**毫无指引 —— 用户一加需求，agent 就掉进猜谜。

### 2. 不声明能力边界 ⇒ agent 无法区分"我错了"和"工具不行"

这是线上 5 小时的直接成因。正确的交互应该是：agent 查一次能力矩阵 → 发现「导入不生成矩形预标注」→ 直接告诉用户「当前版本不支持，需要加功能」，而不是造 9 个探测样本去穷举。**能力缺失并不可怕，可怕的是模型看不见边界。**

### 3. 枚举式 adapter 二选一，没有组合空间

`avi_train` / `aoi_export` 是两个硬编码分支（`converter.py:308` 还会对非法值直接报错）。用户的数据是这样的：

- 一部分样本有参考图，一部分没有
- 想让参考图并排显示
- 想同时保留「整图判定 + 区域框 + 备注」

**这些都在两个枚举值之外**。用户 10:30 那句「为什么一定要增强适配器，你可以不按照这个来吧」就是被这个枚举逼出来的 —— 他感觉自己的需求在迁就工具的模板。

### 4. 用「禁止清单」代替「可配置点」

SKILL.md 有 7 条「禁止」，而自定义空间只有一句轻描淡写的劝退：

> 用户强烈要求自定义控件名时：那些名字拿不到预标注，要说明这会是纯人工标注项目

这是"要么按我的来，要么你自己人工标"。用户想要的第三种选择（我给出映射，你按我的映射渲染）不存在。

### 5. 输入侧无 schema，输出侧无定制点

流程被固定成 4 步（preview → 生成 XML → 校验 → create）。用户需求一旦落到「加个参考图」「换个控件名」「混两类样本」上，整条流程都没有对应的入口。

### 改造方向（不是推翻契约，是让契约可组装）

1. **把 adapter 从"枚举"升级为"映射声明"**：模型产出一份映射（源字段 → 目标控件名 / 区域类型 / 图片角色），平台做校验后执行。
   - 契约从「必须叫这个名」变成「必须给出合法映射」
   - 现有白名单降级为**默认值 + 校验规则**，而不是唯一路径
   - 这样既保住了「预标注一定能渲染」的底线，又给了用户需求落地的入口
2. **每个 adapter 附一个可运行样例**（见 P0），把"文件大小猜语义"变成"对着样例抄"
3. **能力矩阵写进 SKILL.md**，并明确「不支持时应当直接告知用户，不要穷举试探」——这一条能直接把 5 小时的探测消除掉
4. **重审禁止清单**：像「由 Agent 遍历完整数据集或生成 Manifest」这种因噎废食的条款，应该改成"用批量接口做"，而不是"不许做"
5. **图片角色显式化**：meta 里声明 `images: [{role: "defect"|"reference", path}]`，控件按 role 对应，A/B 类样本不再需要"复制缺陷图顶替 gt.jpg"这种语义污染

---

## 五、一句话总结

两个对话暴露的是同一类问题：**skill 用「固定枚举 + 硬契约 + 禁止清单」换取了确定性的下限，但代价是抹掉了业务语义的表达能力，同时又不告诉模型能力边界在哪。**

于是模型遇到边界外的需求时，唯一的策略就是穷举试探 —— 而每一次试探都要付出「重传文件 + 重建项目」的真实成本（169 次上传、17 次失败、4h50m）。用户感受到的「僵硬、没有自由度」，本质是**需求没有落在 skill 预留的任何槽位里**。

最小的第一步不需要改代码：补能力矩阵 + 补可运行样例。

---

## 六、补充验证：用真实数据跑当前代码（09-18 10:20）

拿本地对话产出的真实 `meta_vlm.json`（`public_data/lsbuild/out10/…B0`、`out/…A0`），
直接调 `AVITrainToLabelStudioConverter._build_predictions(meta)`，结果：

```
>>> result 条目: [('quality_label', 'choices'), ('finding_category', 'rectanglelabels')]
>>> 含矩形预标注: True        ← A 类、B 类均如此
```

**结论：`c2576200` 之后，矩形预标注能力确实到位了**（两个类别的真实数据都能产出）。

### 6.1 与官方格式的差异（需要知道，但不阻塞）

当前产出缺 `original_width` / `original_height`（也没有 `image_rotation` / `rotation`）：

```json
{"id":"finding_1","from_name":"finding_category","to_name":"defect_image",
 "type":"rectanglelabels",
 "value":{"x":41.5,"y":45.8,"width":12.0,"height":8.4,"rectanglelabels":["异物"]}}
```

而用户 10:25 手工画框后导出的那份是带的：

```json
{"original_width":256,"original_height":256,"image_rotation":0,
 "value":{"x":30.66,"y":33.41,"width":40.96,"height":36.84,"rotation":0,
          "rectanglelabels":["异物"]}, "origin":"manual"}
```

`converter` 不读图片像素，所以给不出这两个值。**判断：不阻塞渲染** ——
LS 的矩形按百分比定位，尺寸取自实际加载的图片；本地对话里用户没有再抱怨过框，
间接支持这个判断。若日后出现「框位置异常/不显示」，优先怀疑这里。

### 6.2 参考图要显示，不需要改代码

链路是通的：
- `converter.py:394-402` 会把 `defect_image` / `diff_image` / `gt_image` **三个字段都写进 task data**
- `label_config.xml` 里已经有 `<Image name="gt_image" value="$gt_image"/>`
- 所以只要把 `*_cam.bmp` 存成 `gt.jpg`，参考图就会渲染

线上/本地当时的「没有参考图」，是**素材侧没把 cam 图放进 `gt.jpg`**，不是控件或工具的问题。

### 6.3 但留了一个语义污染：A 类样本的 `gt.jpg` 只能是假的

`converter.py:394-398` 对三个槽位**强制要求文件存在**，缺一个就抛
`ConversionError: No media URL resolved for …`。

A 类样本（无 cam 参考图）为了让转换通过，只能把缺陷图复制成 `gt.jpg` —— 结果界面上
「参考图/GT」栏显示的是缺陷图副本，**对标注员是误导**。这是 6.2 那个「能显示」的代价，
也是 P1「图片角色数据驱动」要解决的问题。

### 6.4 本地对话最后还在踩数据模型的坑

12:01 时 agent 自己发现并承认了一个 bug：图片映射用 basename 做 key，导致
`1-0805Y/105/B1` 与 `1-0805Y/24/B1`（都叫 `B1`）**撞成同一张图**；
`1-3-0804F/118/B0` 与 `1-3-0804F/248/B0` 同理。

这又一次印证主结论：**「样本 id → 图片」的映射规则没有显式契约**，
agent 只能自己设计 key，然后自己发现撞车、自己修。用户 11:58 那句
「记得导这些有 cam 的打完标的数据啊」就是在防这类静默错漏。

---

## 七、线上还没吃到这个修复（靠时间线定案）

查线上镜像内容这一步**没做成**：`kubectl` 报 AWS SSO 凭据过期
（关闭沙箱后错误明确为 `The provided authorization grant is invalid, expired, revoked`），
需要人工 `aws sso login` 才能继续。以下结论**由时间线推出**，不依赖集群读取：

| 事件 | 时间 |
|---|---|
| 线上 pod `software-agent-sdk-0` 启动 | **≤ 09-17 19:14**（09-18 10:14 查得 AGE=15h，kubectl 按小时取整，故实际启动不晚于 19:14） |
| `c2576200`（矩形预标注能力 + SKILL.md 补充）提交 | 09-17 20:01 |

**pod 启动时间早于该提交至少 47 分钟 ⇒ 线上镜像里不含矩形预标注能力。**

且 Python 包是打包进镜像的（`/agent-server/.venv/…`），不存在热更新；
按既有结论「**改技能必须重打镜像**」，本次改动（`converter.py` + `SKILL.md`）同样需要
**重新构建镜像并部署**，线上才会生效。

### 这解释了同一天两个环境表现为何不同

- **线上对话（06:07–10:57）**：跑的是 pod 镜像 ⇒ 无矩形能力 ⇒ 框必然不渲染
- **本地对话（11:17–12:04）**：`workspace/conversations/` 落在本地磁盘 ⇒ 跑的是**本地工作树代码**
  ⇒ 已经带上（尚未提交的）矩形能力 ⇒ 框正常，问题只暴露在「参考图」

**所以：线上再跑一遍，如果不发新镜像，框依然不会渲染。** 这是当前最需要先落实的一步。


