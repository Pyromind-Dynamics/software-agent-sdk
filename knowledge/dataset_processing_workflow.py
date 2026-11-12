# ruff: noqa: F821
# pyright: reportUndefinedVariable=false
# workflow: Dataset Processing
# 当前 us-west-1 校验会对节点 02→03、03→05 返回类型不匹配。
# 此例保留数据处理拓扑，不作为已通过校验的模板；生成时按实时节点契约重新绑定路径。

dataset = CloneAndCacheDataset(
    id="01",
    dataset="openai/gsm8k",
    target_path="/workspace/datasets/",
)

train_file = PathJoinNode(
    id="02",
    base_path=dataset.dataset_path,
    subpath="main/train-00000-of-00001.parquet",
)

train_jsonl = DatasetToJsonlNode(
    id="03",
    dataset_path=train_file.joined_path,
)

dataset_kind = DatasetConfigBuilderTextNode(
    id="04",
    user_prompt_field="question",
    assistant_response_field="answer",
)

dataset_config = DatasetConfigBuilderNode(
    id="05",
    train_data_path=train_jsonl.jsonl_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)

dataset_validation = DatasetValidatorNode(
    id="06",
    dataset_config=dataset_config.dataset_config,
    check_reasoning=False,
    limit=0,
    verbose=False,
    preview_html_path="/workspace/datasets/preview.html",
)
