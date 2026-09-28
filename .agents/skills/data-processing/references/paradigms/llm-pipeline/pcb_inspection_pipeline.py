"""Editable PCB pre-annotation pipeline, producing annotation JSONL directly.

The case document shows a readable annotation example, not a fixed protocol.
Choose the fields and region representation for the task; runtime adds source id.
Copy to public_data/data-preparation/; empty vocabulary allows open categories.
Keep obsolete user_prompt/prompt fields under source_metadata in the manifest.
"""

from image_utils import ImagePipelineConfig, run_image_pipeline_from_cli


CATEGORY_VALUES: list[str] = []

CONFIG = ImagePipelineConfig(
    output_format="structured",
    labeling_system_prompt=(
        "对 PCB 裸板 AVI/AOI 样本做预标注，结论只覆盖当前待检图。"
        "按图片角色区分待检原图、正常参考和辅助图，不能由文件名或明暗猜测。"
        "跨模态或未对齐图片不作逐像素比较；反色、差分或增强响应不是缺陷真值。"
        "区分视觉预判与客户验收。有依据时给出真点(label=true)/假点(label=false)；"
        "仅当证据不足或缺失规则实质影响结论时 label=null，并在 note 中说明原因。"
        "缺少客户规则不抹去已知类别、区域和描述。氧化、垃圾、错位等不自动等于假点，"
        "不能排除同时存在的真缺陷；保守拦截不等于已确认的实体缺陷。"
        f"类别词表：{CATEGORY_VALUES}。非空时使用词表；为空时按任务和领域参考给出"
        "有依据的类别，仅无法确定类别时为 null。"
        "交付可读的标注对象，包含有依据的判定或类别、可定位区域和简短现象描述。"
        "例如可用 label/category/boxes/note；字段、层级和区域表示按当前任务确定，"
        "不必固定为这四个字段，也可按区域分别记录不同类别。"
        "明确区域对应的待检原图和坐标系，变换视图上的区域须映射回原图。"
        "可定位的垃圾、氧化等假点也可以给框；无异常或无法可靠定位时说明原因，"
        "不伪造框。只返回 JSON 对象，不包装成训练 messages；id 由运行时注入。"
    ),
    user_prompt_key=None,
    user_prompt_template=(
        "样本 ID：{id}\n"
        "图片角色（与图片顺序一致）：{image_labels}\n"
        "当前验收规则：{inspection_rules}\n"
        "给出真假点判定、类别、待检原图区域框和简短描述。"
    ),
    # Define a response Schema only when the task/downstream contract specifies it.
    # The PCB case is output guidance, not a fixed five-field protocol.
    response_json_schema={"type": "object"},
)


if __name__ == "__main__":
    run_image_pipeline_from_cli(CONFIG)
