# 实例:Java 1.8 特性代码筛选

首个 environment-data-processing 落地场景:**从代码集合中筛出仅使用
Java 1.8 特性的文件/仓库**。需要 JDK 8 环境才能可靠判定(编译期 API
与语言特性检查),因此必须在 java8 镜像沙箱中执行。

## 判定规则(给 codex 的任务定义)

一份源码文件被判为"兼容 Java 1.8",需同时满足:

1. **语言特性 ≤ Java 8**:无 var(10+)、无 switch 表达式(14+)、无
   record/sealed(16/17+)、无文本块(15+)、无模式匹配 instanceof(16+)
2. **API 使用 ∈ JDK 8**:不使用 `java.time` 之外的新 API(如
   `List.of`/`Map.of`(9+)、`Optional.isEmpty`(11+)、`String.isBlank`(11+)、
   `Files.readString`(11+)、`Stream.toList`(16+) 等)
3. **编译验证**:`javac --release 8` 编译通过(环境为 JDK 8 时
   `javac -source 1.8 -target 1.8`)

输出:保留**兼容文件清单**,并附每个文件被排除的原因(命中哪条规则)。

## codex 任务 prompt(模板)

```text
你是 Java 版本兼容性分析专家。对 /workspace/repo 下的所有 .java 文件执行
Java 1.8 兼容性筛选,规则如下:

1. 语言特性:不允许使用 Java 9+ 语法(var、switch 表达式、文本块、
   record、sealed、模式匹配 instanceof 等)
2. 标准库 API:不允许使用 JDK 9+ 新增 API(如 List.of、Map.of、
   String.isBlank、Optional.isEmpty、Stream.toList、Files.readString 等)
3. 对每个文件用 javac 以 release 8 编译验证,记录编译错误

输出:
- /workspace/result/compatible.json:仅兼容 Java 1.8 的文件相对路径列表,
  格式 [{"path": "...", "reason": null}]
- /workspace/result/excluded.json:被排除文件及原因,
  格式 [{"path": "...", "reason": "使用了 var(Java 10 特性)"}]
- 统计信息打印到终端:总文件数 / 兼容数 / 排除数
```

## 沙箱执行流程

### 1. 创建沙箱(java8 镜像)

```json
{"operation": "create", "sandbox_type": "osworld", "image_path": "templates/java8.qcow2", "cpu": 4, "memory": 8, "wait_timeout": 300}
```

无需桌面环境时,优先用 custom 无头容器(启动更快):

```json
{"operation": "create", "sandbox_type": "custom", "image": "eclipse-temurin:8-jdk", "volume_mounts": [{"host_path": "/workspace", "mount_path": "/data"}], "cpu": 4, "memory": 8, "wait_timeout": 300}
```

### 2. 环境探测 + 安装 codex

```json
{"operation": "exec", "sandbox_id": "<id>", "mode": "sync", "command": "java -version 2>&1 && javac -version && npm install -g @openai/codex && codex --version", "timeout": 300}
```

> 若镜像未内置 JDK8,先 `apt-get install -y openjdk-8-jdk`(或对应包)

### 3. 拉取/放置代码

代码若已在 storage,先用现有工具解压到 storage,再判断沙箱是否挂载
`/target-workspace`(有挂载则直接在沙箱内访问;没有则用
`mode="sync"` + `cat >` / 下载方式放入沙箱)。

### 4. 执行 codex 筛选(terminal 模式,实时反馈)

```json
{
  "operation": "exec",
  "sandbox_id": "<id>",
  "mode": "terminal",
  "environment_variables": {"OPENAI_API_KEY": "<key>", "OPENAI_BASE_URL": "<base-url>"},
  "command": "codex exec --sandbox danger-full-access --full-auto -C /workspace/repo \"<上方任务 prompt>\"",
  "timeout": 600
}
```

### 5. 回传产物

```json
{"operation": "upload", "sandbox_id": "<id>", "sandbox_path": "/workspace/result", "storage_path": "datasets/java8-code-filtered/<run-id>"}
```

### 6. 清理沙箱(硬约束)

```json
{"operation": "delete", "sandbox_id": "<id>"}
```

## 判定口径说明

- 以 **javac --release 8 编译结果**为最终仲裁:即使人工规则漏判,
  编译错误也会暴露不兼容点
- 排除原因必须具体(命中语法/API 名称),便于用户复核
- 兼容/排除清单统一为 JSON,方便下游 data-preparation 继续处理
