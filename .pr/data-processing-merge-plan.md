# data-processing 技能整合移植方案与验证记录

基线：`origin/recovery/version-one-openhands-sse-merge` @ `689b53b7`
来源：`feature/sandbox` @ `d8d10c49`（统一技能由其 v3/v4 提交落地）

## 移植范围

### Commit 1 `3fa562b8` — 技能本体
- 新增 `.agents/skills/data-processing/`（3 范式：format-conversion / llm-pipeline / environment-processing + 共享 SOP 路由）
- 删除 `.agents/skills/data-cleaning/`、`.agents/skills/data-preparation/`
- 内容级合并：recovery 对旧 data-preparation 的 6 处改进中，4 处已同步存在于 sandbox 统一技能（实测 IDENTICAL）；`multimodal-labeling.md` 差异仅为链接相对路径（sandbox 版正确）；唯一手工合并项是把 sandbox 重写 playbook 时丢失的两组操作纪律（sample 路径纪律、failure_stage/error_code 修复纪律）合回 `llm-pipeline/playbook.md`
- `pyproject.toml`：新增 edp 脚本的 ruff `UP017` per-file-ignore（3.10 节点兼容）

### Commit 2 `8e3fd20f` — 代码引用适配
- `pyromind_router.py`：技能加载列表 `data-cleaning`+`data-preparation` → `data-processing`；系统提示词路由文案改为统一技能单入口；cleaning/preparation 工具 `runtime_dir` → `data-processing/scripts/{cleaning,preparation}`
- `pi_adapter/adapter.py`：Pi 系统提示词路由 + 默认技能根列表
- `pi_adapter/business_tool_host.py`：cleaning/preparation 运行时解析到新布局
- 测试路径适配：router 测试、pi tools 测试、cleaning/preparation 运行时测试、df_logging；`test_data_cleaning_runtime.py` 直接取 sandbox 版（recovery 无独立改动）
- 明确不移植：`/data-preparation/progress` REST 路由（公开 API，指数据目录概念）、`public_data/data-preparation` 工作区数据目录、docs 历史周报、start_inference.sh 端点配置

### Commit 3 `3942c752` — edp 工具链（第三范式激活）
- `openhands.tools.environment_processing`（EdpRenderTool/EdpSubmitTool/EdpAggregateTool）+ router 接线（edp_params → `data-processing/scripts/edp`）
- `workflow/task_submission.py` 瞬时网络重试（recovery 侧未动过该文件，干净应用）
- `pyromind_dataset/definition.py` 增加 `download_tail_from_pyromind` + `_parse_content_range_total`（render_submit 依赖）
- 测试：environment_processing 6 文件 + test_task_submission
- 修复统一提交时 ruff UP017 自动改写破坏的 2 个 edp 脚本（恢复 3.10 安全的 `datetime.timezone.utc` 写法）

## 明确排除（不在本次范围）
- sandbox 技能/工具包、SDK profiles、response_dispatch、pyromind_subagent、start_inference.sh 端点改动（属 coding 场景 v1-v4 其他工作）
- Pi 侧（business_tool_host）暂不挂 edp 工具：Pi 后端 dev-only，后续按现有 pattern 补
- SKILL.md 负向边界中提到的 sandbox 技能在 recovery 未加载（`_PYROMIND_SKILL_NAMES` 不含），引用为惰性

## 验证结果

| 验证项 | 结果 |
|---|---|
| `pytest tests/pyromind_runtime/`（排除 1 项预存挂起） | 76 passed ✅ |
| `pytest tests/tools/environment_processing/ tests/tools/workflow/test_task_submission.py` | 68 passed ✅ |
| `pytest tests/tools/data_preparation/ tests/tools/pyromind_cleaning/ tests/tools/workflow/ tests/agent_server/test_pyromind_router.py test_pyromind_workflow_sync.py` | 278 passed ✅ |
| pi-runtime TS 测试（`npm test`，需先 `npm ci && npm run build`） | 23 passed / 0 failed ✅ |
| 组合 app 冒烟（openapi 路由检查） | Product API v2 五路由 + v1 路由正常挂载 ✅ |
| 旧技能名残留扫描（`*.py`/`*.ts`，排除 public_data/REST 路由/工具文案） | 零残留 ✅ |
| pre-commit（ruff/pyright/依赖规则/工具注册） | 全绿 ✅ |

## 已知问题
1. **预存挂起（非本次引入）**：`test_product_api_creates_pi_metadata_and_reports_missing_checkpoint` 在本环境挂起。已用干净 worktree（689b53b7 + 已构建 dist）复现同样挂起，判定为基线预存问题（pi fork 全链路依赖本环境不可达的外部资源），本次验证将其排除。待在可达环境下单独核查。
2. pi-runtime `dist/` 为构建产物（gitignore），部署需 `npm ci && npm run build`。
3. ruff UP017 与 edp 脚本的坑已用 per-file-ignore 固化；若新增 edp 脚本请继续使用 `datetime.timezone.utc` 写法。
