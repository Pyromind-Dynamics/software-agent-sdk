(function (modules) {
  "use strict";

  const QUEUE_RADIUS = 30;
  const QUEUE_ROW_HEIGHT = 42;

  function parsePrediction(value) {
    const raw = typeof value === "string" ? value : prettyValue(value);
    const reasoning = [];
    let text = raw;
    let thought;
    while ((thought = text.match(/^\s*<think>([\s\S]*?)<\/think>\s*/i))) {
      reasoning.push(thought[1].trim());
      text = text.slice(thought[0].length);
    }
    text = text.trim();
    const json = text.replace(/^```(?:json)?\s*\n?([\s\S]*?)\n?```$/i, "$1");
    try {
      const structured = JSON.parse(json);
      return { raw, text, reasoning: reasoning.join("\n\n"), structured };
    } catch (_) {
      return { raw, text, reasoning: reasoning.join("\n\n"), structured: null };
    }
  }

  function prettyValue(value) {
    return value === null || value === undefined ? "" : JSON.stringify(value, null, 2);
  }

  function resultLabel(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return resultLabel(value.result ?? value.label);
    }
    return ["string", "number", "boolean"].includes(typeof value)
      ? String(value) : "";
  }

  function createRenderer(options) {
    const global = options.global;
    const document = options.document;
    const main = options.container;
    const data = options.data;
    const store = options.store;
    const storage = options.storage;
    const caseSource = options.caseSource;
    const annotation = modules.annotationStore;
    const panels = data.panels || [];
    const annotationEnabled = panels.includes("annotation");
    const RESOLUTIONS = annotation.RESOLUTIONS;
    const FAILURE_MODES = annotation.FAILURE_MODES;
    const DATA_QUALITY_REASONS = annotation.DATA_QUALITY_REASONS;
    const SEVERITIES = annotation.SEVERITIES;
    const MAX_COMMENT = annotation.MAX_COMMENT;
    const MAX_CORRECTED = annotation.MAX_CORRECTED;
    const MAX_LABELS = annotation.MAX_LABELS;
    const annotationSchema = data.annotation_schema || {};
    const TAXONOMY = annotation.labelList(annotationSchema.taxonomy_labels) || [];
    const OPTION_LABELS = {
      needs_cleaning: "需要清洗",
      exclude: "建议剔除",
      missing: "内容缺失",
      format_error: "格式异常",
      corrupt: "数据损坏",
      other: "其他",
      model_miss: "漏检", model_false_positive: "误检",
      classification_error: "分类错误", severity_misclassification: "等级误判",
      gt_issue: "GT 问题", rubric_issue: "评测规则问题",
      inference_input_issue: "输入问题", report_issue: "报告问题",
    };
    const STATUS_LABELS = {
      reviewed: "✓ 已完成", in_progress: "◐ 标注中", unreviewed: "○ 未标注",
    };
    const editable = store.online && store.unlocked;
    let reviewFilter = "all";
    let currentIndex = 0;
    let currentId = null;
    let renderedId = null;
    let searchTimer = null;
    let activeGallery = null;
    let closeGallery = null;
    let activeTextView = null;
    let closeTextView = null;
    let mediaIdleHandle = null;
    const prefetchedImages = new Map();
    const queueRows = new Map();
    const annotationFields = new Map();
    let validationCaseId = null;
    let validationSummary = null;

    function mk(tag, cls, text) {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text !== null && text !== undefined) node.textContent = text;
      return node;
    }

    function pretty(value) {
      return JSON.stringify(value, null, 2);
    }

    function fold(summary, text) {
      const details = mk("details");
      details.appendChild(mk("summary", null, summary));
      let populated = false;
      details.addEventListener("toggle", () => {
        if (!details.open || populated) return;
        populated = true;
        details.appendChild(mk("pre", null, typeof text === "function" ? text() : text));
      });
      return details;
    }

    function stat(label, value) {
      const box = mk("div");
      box.appendChild(document.createTextNode(label));
      box.appendChild(mk("strong", null, value));
      return box;
    }

    function mediaUrl(media) {
      if (media.url) return media.url;
      if (media.media_id && storage && storage.mediaUrl) {
        return storage.mediaUrl(media.media_id);
      }
      return null;
    }

    function renderMedia(item, target) {
      if (!item.media || !item.media.length) return;
      const wrap = mk("section", "image-review");
      activeGallery = wrap;
      const toolbar = mk("div", "image-tools");
      toolbar.appendChild(mk("span", "group-label", "图像对照"));
      const frames = [];
      let linked = true;
      let dragging = null;
      function apply(frame) {
        frame.image.style.transform = "translate(" + frame.x + "px, " +
          frame.y + "px) scale(" + frame.scale + ")";
      }
      function change(frame, scale, x, y) {
        (linked ? frames : [frame]).forEach((entry) => {
          entry.scale = Math.max(1, Math.min(8, scale));
          entry.x = entry.scale === 1 ? 0 : x;
          entry.y = entry.scale === 1 ? 0 : y;
          apply(entry);
        });
      }
      function tool(label, action) {
        const button = mk("button", null, label);
        button.type = "button";
        button.addEventListener("click", action);
        toolbar.appendChild(button);
        return button;
      }
      tool("−", () => frames[0] && change(frames[0], frames[0].scale / 1.25, 0, 0));
      tool("+", () => frames[0] && change(frames[0], frames[0].scale * 1.25, 0, 0));
      tool("复位", () => frames.forEach((frame) => change(frame, 1, 0, 0)));
      const link = tool("联动：开", () => {
        linked = !linked;
        link.textContent = "联动：" + (linked ? "开" : "关");
        if (linked && frames[0]) change(frames[0], frames[0].scale, frames[0].x, frames[0].y);
      });
      const expand = tool("放大查看", () => {
        const expanded = wrap.classList.toggle("expanded");
        expand.textContent = expanded ? "收起（Esc）" : "放大查看";
      });
      closeGallery = () => {
        wrap.classList.remove("expanded");
        expand.textContent = "放大查看";
      };
      toolbar.appendChild(mk("span", "image-hint", "滚轮缩放 · 拖动查看"));
      wrap.appendChild(toolbar);
      const grid = mk("div", "review-images");
      item.media.forEach((media, index) => {
        const figure = mk("figure", "review-image");
        const caption = mk("figcaption", null, "图 " + (index + 1));
        caption.title = media.caption || "";
        figure.appendChild(caption);
        const viewport = mk("div", "image-viewport");
        const url = mediaUrl(media);
        if (url) {
          const img = prefetchedImages.get(url) || mk("img");
          if (!prefetchedImages.has(url)) img.src = url;
          prefetchedImages.delete(url);
          img.onerror = null;
          img.decoding = "async";
          img.fetchPriority = "high";
          img.alt = media.caption || "图 " + (index + 1);
          img.draggable = false;
          const frame = { image: img, scale: 1, x: 0, y: 0 };
          frames.push(frame);
          viewport.appendChild(img);
          viewport.addEventListener("wheel", (event) => {
            event.preventDefault();
            change(frame, frame.scale * (event.deltaY < 0 ? 1.15 : 1 / 1.15), frame.x, frame.y);
          }, { passive: false });
          viewport.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || frame.scale === 1) return;
            dragging = { frame, x: event.clientX, y: event.clientY,
              originX: frame.x, originY: frame.y };
            viewport.setPointerCapture(event.pointerId);
          });
          viewport.addEventListener("pointermove", (event) => {
            if (!dragging || dragging.frame !== frame) return;
            change(frame, frame.scale, dragging.originX + event.clientX - dragging.x,
              dragging.originY + event.clientY - dragging.y);
          });
          viewport.addEventListener("pointerup", () => { dragging = null; });
          viewport.addEventListener("pointercancel", () => { dragging = null; });
          img.addEventListener("error", () => {
            viewport.textContent = "图片加载失败";
            const retry = mk("button", null, "重试");
            retry.addEventListener("click", () => {
              viewport.textContent = "";
              viewport.appendChild(img);
              img.src = url;
            });
            viewport.appendChild(retry);
          });
        } else viewport.appendChild(mk("p", "meta", media.caption || "图片不可用"));
        figure.appendChild(viewport);
        grid.appendChild(figure);
      });
      wrap.appendChild(grid);
      target.appendChild(wrap);
    }

    function prefetchMedia(items, index) {
      cancelMediaPrefetch();
      if (typeof global.Image !== "function") return;
      if (global.navigator?.connection?.saveData) return;
      const urls = [index + 1, index + 2].flatMap((nextIndex) =>
        (items[nextIndex]?.media || []).map(mediaUrl).filter(Boolean));
      for (const [url, img] of prefetchedImages) {
        if (urls.includes(url)) continue;
        img.removeAttribute("src");
        prefetchedImages.delete(url);
      }
      const load = () => {
        mediaIdleHandle = null;
        urls.forEach((url) => {
          if (prefetchedImages.has(url)) return;
          const img = new global.Image();
          img.decoding = "async";
          img.fetchPriority = "low";
          img.onerror = () => {
            if (prefetchedImages.get(url) === img) prefetchedImages.delete(url);
          };
          prefetchedImages.set(url, img);
          img.src = url;
        });
      };
      if (typeof global.requestIdleCallback === "function") {
        mediaIdleHandle = global.requestIdleCallback(load, { timeout: 500 });
      } else mediaIdleHandle = global.setTimeout(load, 0);
    }

    function cancelMediaPrefetch() {
      if (mediaIdleHandle === null) return;
      if (typeof global.cancelIdleCallback === "function") {
        global.cancelIdleCallback(mediaIdleHandle);
      } else global.clearTimeout(mediaIdleHandle);
      mediaIdleHandle = null;
    }

    function renderRubrics(item, target) {
      if (!panels.includes("rubric") || !item.rubric_items.length) return;
      const wrap = mk("div", "rubrics");
      const score = typeof item.overall_score === "number"
        ? item.overall_score.toFixed(3) : "n/a";
      wrap.appendChild(mk("h3", null, "Rubric evaluation · overall " + score));
      item.rubric_items.forEach((rubric) => {
        const section = mk(
          "section", "rubric " + (rubric.passed ? "passed" : "failed"));
        const header = mk("header");
        header.appendChild(mk("strong", null, rubric.name));
        header.appendChild(mk(
          "span", null, rubric.score.toFixed(2) + " × " + rubric.weight));
        section.appendChild(header);
        section.appendChild(mk("p", null, rubric.criterion));
        const list = mk("ul");
        rubric.evidence.forEach((entry) =>
          list.appendChild(mk("li", null, entry)));
        section.appendChild(list);
        section.appendChild(mk("p", "failure", rubric.failure_reason));
        wrap.appendChild(section);
      });
      target.appendChild(wrap);
    }

    function chipField(id, field, label, values, selected, required) {
      const row = mk("div", "row wide");
      row.appendChild(mk("label", required ? "required" : null, label));
      const chips = mk("div", "chips");
      const current = new Set(selected || []);
      let firstControl = null;
      values.forEach((value) => {
        const chip = mk("label");
        const box = mk("input");
        if (!firstControl) firstControl = box;
        box.type = "checkbox";
        box.value = value;
        box.checked = current.has(value);
        box.addEventListener("change", () => {
          if (box.checked) {
            if (current.size >= MAX_LABELS) {
              box.checked = false;
              return;
            }
            current.add(value);
          } else {
            current.delete(value);
          }
          const patch = {};
          patch[field] = [...current];
          store.commit(id, patch);
        });
        chip.appendChild(box);
        chip.appendChild(mk("span", null, OPTION_LABELS[value] || value));
        chips.appendChild(chip);
      });
      row.appendChild(chips);
      registerAnnotationField(label, row, firstControl);
      return row;
    }

    function textField(id, field, label, value, required, options) {
      const settings = options || {};
      const row = mk("div", "row wide");
      row.appendChild(mk("label", required ? "required" : null, label));
      const input = mk(settings.multiline ? "textarea" : "input");
      if (settings.multiline) {
        input.rows = settings.rows || 3;
        input.maxLength = settings.maxLength || MAX_COMMENT;
      } else if (settings.type) {
        input.type = settings.type;
      }
      if (settings.min !== undefined) input.min = settings.min;
      if (settings.max !== undefined) input.max = settings.max;
      input.value = value === null || value === undefined ? "" : String(value);
      input.addEventListener("input", () => {
        const patch = {};
        patch[field] = settings.type === "number"
          ? (input.value ? Number(input.value) : null)
          : input.value;
        store.commit(id, patch);
      });
      row.appendChild(input);
      registerAnnotationField(label, row, input);
      return row;
    }

    function selectField(id, field, label, values, value, required) {
      const row = mk("div", "row");
      row.appendChild(mk("label", required ? "required" : null, label));
      const select = mk("select");
      const placeholder = mk("option", null, "— 请选择 —");
      placeholder.value = "";
      select.appendChild(placeholder);
      values.forEach((option) => {
        const node = mk("option", null, option);
        node.value = option;
        select.appendChild(node);
      });
      select.value = value || "";
      select.addEventListener("change", () => {
        const patch = {};
        patch[field] = select.value;
        store.commit(id, patch);
      });
      row.appendChild(select);
      registerAnnotationField(label, row, select);
      return row;
    }

    function registerAnnotationField(label, row, control) {
      const feedback = mk("p", "field-error");
      feedback.id = "annotation-error-" + annotationFields.size;
      feedback.hidden = true;
      row.appendChild(feedback);
      if (control) control.setAttribute("aria-describedby", feedback.id);
      annotationFields.set(label, { row, control, feedback });
    }

    function refreshValidation() {
      const missing = validationCaseId === currentId
        ? missingFields(store.getAnnotation(currentId)) : [];
      annotationFields.forEach(({ row, control, feedback }, label) => {
        const invalid = missing.includes(label);
        row.classList.toggle("field-invalid", invalid);
        feedback.hidden = !invalid;
        feedback.textContent = invalid ? "请填写" + label : "";
        if (control) control.setAttribute("aria-invalid", String(invalid));
      });
      if (validationSummary) {
        validationSummary.hidden = !missing.length;
        validationSummary.textContent = missing.length
          ? "还需填写：" + missing.join("、") : "";
      }
      return missing;
    }

    function saveAndNext(id) {
      if (!editable || annotationPanel.inert) return;
      validationCaseId = id;
      const missing = refreshValidation();
      if (missing.length) {
        const first = annotationFields.get(missing[0]);
        if (first) {
          first.row.scrollIntoView({ block: "nearest", behavior: "smooth" });
          if (first.control) first.control.focus({ preventScroll: true });
        }
        return;
      }
      gotoRelative(1);
    }

    function currentItems() {
      return caseSource.items;
    }

    function missingFields(annotationValue) {
      const missing = [];
      if (!annotationValue.resolution_code) missing.push("处理结论");
      if (["model_gap", "coverage_gap"].includes(
        annotationValue.resolution_code)) {
        if (!annotationValue.failure_modes.length) missing.push("失败类型");
        if (!annotationValue.severity) missing.push("严重程度");
      }
      if (annotationValue.resolution_details_version === 1 &&
          annotationValue.resolution_code === "gt_error" &&
          !annotationValue.corrected_answer) missing.push("修正答案");
      if (annotationValue.resolution_details_version === 1 &&
          annotationValue.resolution_code === "poor_data" &&
          !annotationValue.data_quality_reasons.length) missing.push("数据问题");
      if (annotationValue.resolution_details_version === 1 &&
          annotationValue.resolution_code === "coverage_gap" &&
          !annotationValue.missing_scenario) missing.push("缺失场景");
      if (annotationValue.resolution_details_version === 1 &&
          annotationValue.resolution_code === "acceptable" &&
          !annotationValue.acceptance_reason) missing.push("可接受原因");
      if (annotationValue.resolution_details_version === 1 &&
          annotationValue.resolution_code === "uncertain" &&
          !annotationValue.verification_items) missing.push("待核查事项");
      return missing;
    }

    function renderAnnotation(item) {
      annotationFields.clear();
      validationSummary = null;
      const wrap = mk("section", "annotate" + (editable ? "" : " locked"));
      wrap.appendChild(mk("h2", null, "人工复核"));
      wrap.appendChild(mk("p", "annotation-hint", "1–6 选结论 · Ctrl/⌘ Enter 下一条"));
      if (!editable) {
        wrap.appendChild(mk(
          "p", "meta", !store.unlocked
            ? "Locked until the evaluation is validated."
            : "Open the stable report entry to annotate."));
        return wrap;
      }
      const current = store.getAnnotation(item.id);
      const controls = mk("div", "controls");
      const decisionField = mk("div", "row wide");
      decisionField.appendChild(mk("label", "required", "处理结论"));
      const decisionWrap = mk("div", "decision");
      let firstRadio = null;
      RESOLUTIONS.forEach((entry, index) => {
        const label = mk("label");
        const radio = mk("input");
        if (!firstRadio) firstRadio = radio;
        radio.type = "radio";
        radio.name = "resolution-" + item.id;
        radio.value = entry[0];
        radio.checked = current.resolution_code === entry[0];
        radio.addEventListener("change", () => {
          store.commit(item.id, {
            resolution_code: entry[0],
            resolution_inferred: false,
            resolution_details_version: 1,
            decision: entry[2],
            disposition: entry[3],
          });
          renderWorkspace(true);
        });
        label.appendChild(radio);
        label.appendChild(mk("span", null, (index + 1) + ". " + entry[1]));
        decisionWrap.appendChild(label);
      });
      decisionField.appendChild(decisionWrap);
      registerAnnotationField("处理结论", decisionField, firstRadio);
      controls.appendChild(decisionField);
      const needsModelDetail = ["model_gap", "coverage_gap"].includes(
        current.resolution_code);
      if (needsModelDetail) {
        controls.appendChild(chipField(
          item.id, "failure_modes", "失败类型", FAILURE_MODES,
          current.failure_modes, true));
        if (TAXONOMY.length) {
          controls.appendChild(chipField(
            item.id, "taxonomy_labels", "场景标签", TAXONOMY,
            current.taxonomy_labels, false));
        }
        controls.appendChild(selectField(
          item.id, "severity", "严重程度", SEVERITIES, current.severity, true));
      }
      if (current.resolution_code === "gt_error") {
        controls.appendChild(textField(
          item.id, "corrected_answer", "修正答案", current.corrected_answer,
          current.resolution_details_version === 1,
          { multiline: true, maxLength: MAX_CORRECTED }));
      } else if (current.resolution_code === "poor_data") {
        controls.appendChild(chipField(
          item.id, "data_quality_reasons", "数据问题", DATA_QUALITY_REASONS,
          current.data_quality_reasons,
          current.resolution_details_version === 1));
      } else if (current.resolution_code === "model_gap") {
        controls.appendChild(textField(
          item.id, "training_suggestion", "训练优化建议",
          current.training_suggestion, false, { multiline: true }));
      } else if (current.resolution_code === "coverage_gap") {
        controls.appendChild(textField(
          item.id, "missing_scenario", "缺失场景", current.missing_scenario,
          current.resolution_details_version === 1, { multiline: true }));
        controls.appendChild(textField(
          item.id, "target_sample_count", "建议采集/合成数量",
          current.target_sample_count, false,
          { type: "number", min: 1, max: 1000000 }));
      } else if (current.resolution_code === "acceptable") {
        controls.appendChild(textField(
          item.id, "acceptance_reason", "可接受原因",
          current.acceptance_reason, current.resolution_details_version === 1,
          { multiline: true }));
      } else if (current.resolution_code === "uncertain") {
        controls.appendChild(textField(
          item.id, "verification_items", "待核查事项",
          current.verification_items,
          current.resolution_details_version === 1,
          { multiline: true }));
      }
      const commentField = mk("div", "row wide");
      commentField.appendChild(mk("label", null, "备注（可选）"));
      const comment = mk("textarea");
      comment.rows = 3;
      comment.maxLength = MAX_COMMENT;
      comment.value = current.comment || "";
      comment.addEventListener("input", () =>
        store.commit(item.id, { comment: comment.value }));
      commentField.appendChild(comment);
      controls.appendChild(commentField);
      wrap.appendChild(controls);
      const actions = mk("div", "actions");
      validationSummary = mk("p", "validation-summary");
      validationSummary.setAttribute("role", "status");
      validationSummary.setAttribute("aria-live", "polite");
      validationSummary.hidden = true;
      actions.appendChild(validationSummary);
      const saveNext = mk("button", "primary", "保存并下一条 →");
      saveNext.addEventListener("click", () => saveAndNext(item.id));
      const clear = mk("button", null, "清除标注");
      clear.addEventListener("click", () => {
        store.clear(item.id);
        renderWorkspace(true);
      });
      actions.appendChild(saveNext);
      actions.appendChild(clear);
      wrap.appendChild(actions);
      refreshValidation();
      return wrap;
    }

    function renderCaseContent(item) {
      const textMode = data.template_id === "evaluation-basic-v1" || !(item.media && item.media.length);
      const article = mk("article", "case-detail " + item.status + (textMode ? " text-case" : ""));
      const header = mk("header");
      header.appendChild(mk("code", null, item.id));
      const output = parsePrediction(item.prediction);
      const prediction = resultLabel(output.structured) ||
        (output.text.length <= 80 ? output.text : "未解析");
      const reference = resultLabel(item.reference) || "查看详情";
      const comparison = mk("div", "result-comparison");
      const gt = mk("span", "result-tag", textMode &&
        (reference.length > 40 || reference.includes("\n") || reference === "查看详情")
        ? "参考答案见下方" : "GT：" +
          (reference.length > 80 ? reference.slice(0, 80) + "…" : reference));
      gt.title = prettyValue(item.reference);
      comparison.appendChild(gt);
      comparison.appendChild(mk("span", "result-tag model", textMode &&
        (prediction.length > 40 || prediction.includes("\n") || prediction === "未解析")
        ? "模型输出见下方" : "模型：" + prediction));
      header.appendChild(comparison);
      article.appendChild(header);
      if (textMode) article.appendChild(renderTextWorkbench(item, output));
      else renderMedia(item, article);
      const summary = mk("section", "prediction-summary");
      const structured = output.structured;
      if (!textMode && structured && typeof structured === "object" && !Array.isArray(structured)) {
        if (typeof structured.reason === "string") {
          summary.appendChild(mk("p", "model-reason", "模型原因：" + structured.reason));
        } else summary.appendChild(mk("pre", "model-text", prettyValue(structured)));
        if (typeof structured.confidence === "number" && Number.isFinite(structured.confidence)) {
          const confidence = structured.confidence >= 0 && structured.confidence <= 1
            ? (structured.confidence * 100).toFixed(0) + "%" : String(structured.confidence);
          summary.appendChild(mk("span", "meta", "置信度：" + confidence));
        }
        if (Array.isArray(structured.boxes)) {
          summary.appendChild(mk("span", "meta", "框数量：" + structured.boxes.length));
          if (structured.boxes.length) summary.appendChild(fold("查看预测框坐标", pretty(structured.boxes)));
        }
      } else if (!textMode) {
        summary.appendChild(mk("pre", "model-text", output.text || "无模型输出"));
      }
      const failures = (item.rubric_items || []).filter((rubric) => !rubric.passed);
      if (failures.length) {
        const labelMismatch = failures.some((rubric) => rubric.name === "label_accuracy") &&
          prediction !== "未解析" && reference !== "查看详情" && prediction !== reference;
        summary.appendChild(mk("p", "evaluation-note", labelMismatch
          ? "标签不一致：GT 与模型判断不同，请结合" + (textMode ? "原始输入" : "图片") + "复核"
          : "评测未通过：" + failures.map((rubric) => rubric.name).join("、")));
      }
      const distribution = item.detail?.details?.distribution_context;
      if (distribution) {
        const context = mk("section", "distribution-context");
        context.appendChild(mk("strong", null, "类别数据与效果：" + distribution.label));
        for (const [name, value] of [["训练集", distribution.train], ["验证集", distribution.validation]]) {
          context.appendChild(mk("p", null, name + (value ? `：${value.count} 条，占比 ${(value.ratio * 100).toFixed(1)}%` : "：暂无统计")));
        }
        if (distribution.performance) context.appendChild(mk("p", null,
          `本类通过率 ${(distribution.performance.pass_rate * 100).toFixed(1)}%，共 ${distribution.performance.support} 条`));
        context.appendChild(mk("p", "meta", "样本量用于判断原因假设，不能单独证明数据不足。"));
        article.appendChild(context);
      }
      article.appendChild(summary);
      const details = mk("section", "review-details");
      if (!textMode && item.input && Object.keys(item.input).length) {
        const input = fold("原始输入", () => pretty(item.input));
        input.className = "sample-input";
        input.open = !(item.media && item.media.length);
        details.appendChild(input);
      }
      if (output.reasoning) details.appendChild(fold("推理过程", output.reasoning));
      details.appendChild(fold("完整模型输出", output.raw));
      if (!textMode) details.appendChild(fold("完整 GT", () => prettyValue(item.reference)));
      else details.appendChild(fold("原始输入数据", () => prettyValue(item.input)));
      const evaluation = mk("details");
      evaluation.appendChild(mk("summary", null, "评测详情"));
      renderRubrics(item, evaluation);
      if (panels.includes("raw_output")) {
        evaluation.appendChild(fold("指标与诊断数据", () => pretty(item.detail)));
      }
      evaluation.appendChild(mk("p", "meta", "耗时：" + item.latency_seconds +
        "s · 错误：" + (item.error || "无")));
      details.appendChild(evaluation);
      article.appendChild(details);
      if (item.validation_errors && item.validation_errors.length) {
        const box = mk("div", "banner error");
        box.appendChild(mk("strong", null, "Validation errors"));
        const list = mk("ul", "errors");
        item.validation_errors.forEach((message) =>
          list.appendChild(mk("li", null, message)));
        box.appendChild(list);
        article.appendChild(box);
      }
      return article;
    }

    function renderTextValue(value, target) {
      if (Array.isArray(value) && value.length && value.every((entry) =>
        entry && typeof entry === "object" && typeof entry.role === "string")) {
        const roles = { system: "系统", user: "用户", assistant: "助手", tool: "工具" };
        value.forEach((message) => {
          const turn = mk("section", "conversation-turn");
          turn.appendChild(mk("span", "turn-role", roles[message.role] || message.role));
          if (Array.isArray(message.content)) {
            message.content.forEach((part) => {
              if (part && part.type === "text" && typeof part.text === "string") {
                turn.appendChild(mk("div", "reading-text", part.text));
              } else turn.appendChild(fold("非文本内容", () => prettyValue(part)));
            });
          } else renderTextValue(message.content, turn);
          target.appendChild(turn);
        });
      } else if (typeof value === "string") {
        target.appendChild(mk("div", "reading-text", value || "（空文本）"));
      } else target.appendChild(mk("pre", "reading-json", prettyValue(value) || "（无数据）"));
    }

    function renderTextWorkbench(item, output) {
      const workbench = mk("section", "text-workbench");
      activeTextView = workbench;
      const tools = mk("div", "text-tools");
      tools.appendChild(mk("span", "group-label", "文本对照"));
      const sync = mk("button", null, "联动滚动：关");
      sync.type = "button";
      let linked = false;
      sync.addEventListener("click", () => {
        linked = !linked;
        sync.textContent = "联动滚动：" + (linked ? "开" : "关");
      });
      tools.appendChild(sync);
      const expand = mk("button", null, "放大阅读");
      expand.type = "button";
      expand.addEventListener("click", () => {
        const expanded = workbench.classList.toggle("expanded");
        expand.textContent = expanded ? "收起（Esc）" : "放大阅读";
      });
      closeTextView = () => {
        workbench.classList.remove("expanded");
        expand.textContent = "放大阅读";
      };
      tools.appendChild(expand);
      workbench.appendChild(tools);
      function pane(title, cls) {
        const section = mk("section", "text-pane " + cls);
        section.appendChild(mk("h3", null, title));
        const body = mk("div", "text-pane-body");
        body.tabIndex = 0;
        body.setAttribute("role", "region");
        body.setAttribute("aria-label", title);
        section.appendChild(body);
        return { section, body };
      }
      const input = pane("原始输入", "text-input");
      const fields = item.input && Object.keys(item.input).length ? item.input : null;
      if (fields) {
        const labels = { prompt: "用户输入", query: "问题", messages: "对话", system: "系统指令" };
        Object.entries(fields).forEach(([key, value]) => {
          if (Object.keys(fields).length > 1) input.body.appendChild(mk("h4", null, labels[key] || key));
          renderTextValue(value, input.body);
        });
      } else input.body.appendChild(mk("p", "meta", "此报告未包含原始输入，可重新生成报告补齐。"));
      workbench.appendChild(input.section);
      const answers = mk("div", "text-answers");
      const reference = pane("参考答案（GT）", "text-reference");
      renderTextValue(item.reference, reference.body);
      const prediction = pane("模型输出", "text-prediction");
      renderTextValue(output.structured === null ? output.text : output.structured, prediction.body);
      answers.appendChild(reference.section);
      answers.appendChild(prediction.section);
      let syncing = false;
      [reference.body, prediction.body].forEach((body, index, bodies) => {
        body.addEventListener("scroll", () => {
          if (!linked || syncing) return;
          const other = bodies[1 - index];
          const distance = body.scrollHeight - body.clientHeight;
          if (distance <= 0) return;
          syncing = true;
          other.scrollTop = (body.scrollTop / distance) * (other.scrollHeight - other.clientHeight);
          global.requestAnimationFrame(() => { syncing = false; });
        });
      });
      workbench.appendChild(answers);
      return workbench;
    }

    const hero = mk("section", "hero compact");
    const heroTitle = mk("div");
    heroTitle.appendChild(mk("h1", null, data.title));
    const datasetLabel = mk("p", null, data.dataset_path);
    datasetLabel.title = data.dataset_path;
    heroTitle.appendChild(datasetLabel);
    hero.appendChild(heroTitle);
    if (data.classification) {
      main.appendChild(fold("业务分类指标与混淆矩阵", pretty(data.classification)));
    }
    if (data.original_label_metrics) {
      main.appendChild(fold("原始标签表现", pretty(data.original_label_metrics)));
    }
    if (data.integrity_issues?.length) {
      main.appendChild(fold("数据完整性问题（不可用于选模）", pretty(data.integrity_issues)));
    }
    if (data.dataset_distribution && Object.keys(data.dataset_distribution).length) {
      main.appendChild(fold("训练集与验证集标签分布", pretty(data.dataset_distribution)));
    }
    const summary = data.summary;
    const grid = mk("div", "summary");
    grid.appendChild(stat("总数", summary.total));
    if (data.metrics_available !== false) grid.appendChild(stat("通过", summary.passed));
    grid.appendChild(stat("待复核", summary.failed));
    if (data.metrics_available !== false && summary.pass_rate != null)
      grid.appendChild(stat("通过率", (summary.pass_rate * 100).toFixed(1) + "%"));
    hero.appendChild(grid);
    main.appendChild(hero);
    main.appendChild(fold("评测指标（全部样本）", () => pretty({
      rubric_metrics: data.rubric_metrics,
      mean_latency_seconds: data.mean_latency_seconds,
    })));

    const notices = mk("div", "notices");
    main.appendChild(notices);
    const toolbar = mk("div", "toolbar");
    main.appendChild(toolbar);
    const queueBackdrop = mk("button", "queue-backdrop");
    queueBackdrop.type = "button";
    queueBackdrop.title = "Close case queue";
    queueBackdrop.addEventListener("click", () =>
      main.classList.remove("queue-open"));
    main.appendChild(queueBackdrop);
    const workspace = mk("div", "review-workspace");
    const queuePanel = mk("aside", "case-queue");
    const queueList = mk("div", "queue-list");
    queuePanel.appendChild(queueList);
    const contentPanel = mk("section", "case-content");
    const annotationPanel = mk("aside", "annotation-panel");
    workspace.appendChild(queuePanel);
    workspace.appendChild(contentPanel);
    workspace.appendChild(annotationPanel);
    main.appendChild(workspace);

    const progressFill = mk("span");
    const progressText = mk("span", "progress", "");
    const saveStateEl = mk("span", "save-state", "");
    const navLabel = mk("span", "progress", "0 / 0");

    function buttonGroup(labelText, values, initial, onPick) {
      const group = mk("div", "group");
      group.appendChild(mk("span", "group-label", labelText));
      const buttons = [];
      values.forEach((value) => {
        const button = mk("button", null, value[1]);
        if (value[0] === initial) button.classList.add("active");
        button.addEventListener("click", () => {
          buttons.forEach((other) =>
            other.classList.toggle("active", other === button));
          onPick(value[0]);
        });
        buttons.push(button);
        group.appendChild(button);
      });
      return group;
    }

    const queueToggle = mk("button", "queue-toggle", "样本队列");
    queueToggle.type = "button";
    queueToggle.addEventListener("click", () =>
      main.classList.toggle("queue-open"));
    toolbar.appendChild(queueToggle);

    if (annotationEnabled) {
      toolbar.appendChild(buttonGroup(
        "筛选",
        [["all", "全部"], ["reviewed", "已完成"],
          ["in_progress", "标注中"], ["unreviewed", "未标注"]],
        reviewFilter,
        (value) => {
          if (value === reviewFilter) return;
          reviewFilter = value;
          currentIndex = 0;
          caseSource.setReviewStatus(value);
        }));
    }
    const search = mk("input");
    search.type = "search";
    search.placeholder = "搜索样本编号";
    search.addEventListener("input", () => {
      if (searchTimer) global.clearTimeout(searchTimer);
      searchTimer = global.setTimeout(() => {
        currentIndex = 0;
        currentId = null;
        caseSource.setQuery(search.value);
      }, 250);
    });
    toolbar.appendChild(search);
    toolbar.appendChild(mk("div", "spacer"));
    if (annotationEnabled) {
      const progressWrap = mk("div", "progress-wrap");
      const progressBar = mk("div", "progress-bar");
      progressBar.appendChild(progressFill);
      progressWrap.appendChild(progressBar);
      progressWrap.appendChild(progressText);
      progressWrap.appendChild(saveStateEl);
      toolbar.appendChild(progressWrap);
    }
    const navGroup = mk("div", "group");
    const previous = mk("button", null, "←");
    previous.title = "Previous case (Left arrow)";
    previous.addEventListener("click", () => gotoRelative(-1));
    const next = mk("button", null, "→");
    next.title = "Next case (Right arrow)";
    next.addEventListener("click", () => gotoRelative(1));
    navGroup.appendChild(previous);
    navGroup.appendChild(navLabel);
    navGroup.appendChild(next);
    toolbar.appendChild(navGroup);
    if (annotationEnabled) {
      const downloadGroup = mk("details", "export-menu");
      downloadGroup.appendChild(mk("summary", null, "导出 ▾"));
      const exportOptions = mk("div", "export-options");
      [["evaluation_annotations.json", "JSON"],
        ["evaluation_annotations.jsonl", "JSONL"],
        ["labeled_evaluation_results.jsonl", "Labeled"]].forEach((entry) => {
        const link = mk("a", "download-link", entry[1]);
        const enabled = store.online && typeof store.exportUrl === "function";
        link.href = enabled ? store.exportUrl(entry[0]) : "#";
        if (!enabled) {
          link.classList.add("disabled");
          link.title = "Open the stable report entry to download";
        }
        link.addEventListener("click", () => { downloadGroup.open = false; });
        exportOptions.appendChild(link);
      });
      downloadGroup.appendChild(exportOptions);
      toolbar.appendChild(downloadGroup);
    }

    function renderNotices(message) {
      notices.textContent = "";
      (data.warnings || []).forEach((warning) =>
        notices.appendChild(banner("", warning)));
      if (!store.online && annotationEnabled) {
        notices.appendChild(banner(
          "error", "Autosave unavailable: open the stable report entry to " +
          "load cases and annotate."));
      } else if (!store.unlocked && annotationEnabled) {
        notices.appendChild(banner(
          "lock", "Human annotation is locked until validation succeeds."));
      }
      if (message) notices.appendChild(banner("error", message));
    }

    function renderLoadError() {
      renderNotices(null);
      const box = banner("error", "Loading annotations failed. ");
      const retry = mk("button", null, "Retry");
      retry.addEventListener("click", () => store.load());
      box.appendChild(retry);
      notices.appendChild(box);
    }

    function banner(kind, text) {
      return mk("div", "banner" + (kind ? " " + kind : ""), text);
    }

    function renderQueue(items, followCurrent = true) {
      queueRows.clear();
      queueList.textContent = "";
      if (!items.length) {
        queueList.appendChild(mk(
          "p", "queue-empty", caseSource.loading ? "加载样本…" :
            "没有符合条件的样本"));
        return;
      }
      const viewportHeight = Number(queuePanel.clientHeight) || 600;
      if (followCurrent) {
        const rowTop = 6 + currentIndex * QUEUE_ROW_HEIGHT;
        const rowBottom = rowTop + QUEUE_ROW_HEIGHT;
        if (rowTop < queuePanel.scrollTop) queuePanel.scrollTop = rowTop;
        else if (rowBottom > queuePanel.scrollTop + viewportHeight) {
          queuePanel.scrollTop = rowBottom - viewportHeight;
        }
      }
      const visibleStart = Math.floor((Number(queuePanel.scrollTop) || 0) / QUEUE_ROW_HEIGHT);
      const start = Math.max(0, visibleStart - QUEUE_RADIUS);
      const end = Math.min(items.length,
        visibleStart + Math.ceil(viewportHeight / QUEUE_ROW_HEIGHT) + QUEUE_RADIUS);
      if (start) {
        const spacer = mk("div", "queue-spacer");
        spacer.style.height = (start * QUEUE_ROW_HEIGHT) + "px";
        queueList.appendChild(spacer);
      }
      items.slice(start, end).forEach((item, offset) => {
        const index = start + offset;
        const state = store.badgeState(item.id);
        const button = mk("button", "queue-item " + state);
        button.classList.toggle("current", index === currentIndex);
        button.appendChild(mk(
          "span", "queue-index", String(caseSource.startOffset + index + 1)));
        button.appendChild(mk("span", "queue-id", item.id));
        const stateLabel = mk("span", "queue-state", STATUS_LABELS[state]);
        button.appendChild(stateLabel);
        queueRows.set(item.id, { button, stateLabel });
        button.addEventListener("click", () => {
          main.classList.remove("queue-open");
          gotoIndex(index);
        });
        queueList.appendChild(button);
      });
      if (end < items.length) {
        const spacer = mk("div", "queue-spacer");
        spacer.style.height =
          ((items.length - end) * QUEUE_ROW_HEIGHT) + "px";
        queueList.appendChild(spacer);
      }
      if (caseSource.loading) {
        queueList.appendChild(mk("p", "queue-loading", "加载下一批…"));
      } else if (caseSource.error) {
        const retry = mk("button", "queue-retry", "重试加载");
        retry.addEventListener("click", () => caseSource.retry());
        queueList.appendChild(retry);
      }
    }

    function renderCurrent(items, force) {
      if (!items.length) {
        if (!caseSource.loading) {
          contentPanel.textContent = "";
          annotationPanel.textContent = "";
          renderedId = null;
        }
        return;
      }
      if (currentId) {
        const retained = items.findIndex((item) => item.id === currentId);
        if (retained >= 0) currentIndex = retained;
      }
      currentIndex = Math.max(0, Math.min(currentIndex, items.length - 1));
      const item = items[currentIndex];
      currentId = item.id;
      const changed = renderedId !== currentId;
      caseSource.ensureAhead(currentIndex);
      if (changed) {
        renderedId = currentId;
        contentPanel.textContent = "";
        activeGallery = null;
        closeGallery = null;
        activeTextView = null;
        closeTextView = null;
        contentPanel.appendChild(renderCaseContent(item));
        prefetchMedia(items, currentIndex);
      }
      if (changed || force) {
        annotationPanel.textContent = "";
        if (annotationEnabled) annotationPanel.appendChild(renderAnnotation(item));
      }
    }

    function refreshStats() {
      const total = caseSource.total || data.summary.failed;
      const reviewed = store.reviewedCount();
      progressFill.style.width = (total ? (reviewed / total) * 100 : 0) + "%";
      progressText.textContent = reviewed + " / " + total + " 已完成";
      const labels = {
        loading: "加载标注…",
        saving: "保存中…",
        saved: "已保存",
        error: "保存失败，正在重试…",
        rejected: "未保存：服务端拒绝了数据",
        conflict: "同步最新标注…",
        readonly: "只读",
      };
      saveStateEl.textContent = labels[store.saveState] || "";
      saveStateEl.className = "save-state " + store.saveState;
    }

    function renderWorkspace(force = false) {
      const items = currentItems();
      annotationPanel.inert = caseSource.loading && !items.length;
      renderCurrent(items, force);
      renderQueue(items);
      const displayedTotal = caseSource.filteredTotal;
      navLabel.textContent = items.length
        ? (caseSource.startOffset + currentIndex + 1) + " / " + displayedTotal
        : caseSource.loading ? "筛选中…" : "0 / " + displayedTotal;
      refreshStats();
    }

    function gotoIndex(index) {
      const items = currentItems();
      if (!items.length) return;
      currentIndex = Math.max(0, Math.min(index, items.length - 1));
      currentId = items[currentIndex].id;
      renderWorkspace();
    }

    async function gotoRelative(delta) {
      const items = currentItems();
      if (!items.length) return;
      const nextId = await caseSource.move(items[currentIndex].id, delta);
      if (!nextId) return;
      currentId = nextId;
      const nextItems = currentItems();
      const nextIndex = nextItems.findIndex((item) => item.id === nextId);
      if (nextIndex >= 0) currentIndex = nextIndex;
      renderWorkspace();
    }

    function isTypingTarget(target) {
      if (!target || !target.tagName) return false;
      return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
        target.isContentEditable;
    }

    function onKeydown(event) {
      if (event.key === "Escape") {
        if (closeGallery) closeGallery();
        if (closeTextView) closeTextView();
        main.classList.remove("queue-open");
        if (isTypingTarget(event.target) && typeof event.target.blur === "function") {
          event.target.blur();
        }
        return;
      }
      if (activeGallery && activeGallery.classList.contains("expanded")) return;
      if (activeTextView && activeTextView.classList.contains("expanded")) return;
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        const item = currentItems()[currentIndex];
        if (item) {
          event.preventDefault();
          saveAndNext(item.id);
        }
        return;
      }
      if (isTypingTarget(event.target)) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        gotoRelative(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        gotoRelative(1);
      } else if (event.key === "/") {
        event.preventDefault();
        search.focus();
      } else if (/^[1-6]$/.test(event.key)) {
        const item = currentItems()[currentIndex];
        const resolution = RESOLUTIONS[Number(event.key) - 1];
        if (item && resolution && editable) {
          event.preventDefault();
          store.commit(item.id, {
            resolution_code: resolution[0],
            resolution_inferred: false,
            resolution_details_version: 1,
            decision: resolution[2],
            disposition: resolution[3],
          });
          renderWorkspace(true);
        }
      }
    }

    const unsubscribeStore = store.subscribe((event) => {
      if (event.type === "notice") renderNotices(event.detail);
      else if (event.type === "load-error") renderLoadError();
      if (event.type === "replace") renderWorkspace(true);
      else if (event.type === "change") {
        if (!caseSource.paged && reviewFilter !== "all") {
          caseSource.refresh();
          return;
        }
        const row = queueRows.get(event.detail);
        if (row) {
          const state = store.badgeState(event.detail);
          row.button.className = "queue-item " + state;
          row.button.classList.toggle("current", event.detail === currentId);
          row.stateLabel.textContent = STATUS_LABELS[state];
        }
        refreshStats();
        refreshValidation();
      } else if (event.type === "state") {
        if (store.saveState === "saved" && reviewFilter !== "all") {
          const absoluteIndex = caseSource.startOffset + currentIndex;
          const start = caseSource.refresh(currentId);
          currentIndex = Math.max(0, absoluteIndex - start);
          return;
        }
        refreshStats();
      }
    });
    function onQueueScroll() { renderQueue(currentItems(), false); }
    queuePanel.addEventListener("scroll", onQueueScroll);
    const unsubscribeCases = caseSource.subscribe(() => {
      if (caseSource.loading && !caseSource.items.length) cancelMediaPrefetch();
      renderWorkspace();
    });
    global.addEventListener("keydown", onKeydown);
    renderNotices(null);
    renderWorkspace();

    return Object.freeze({
      destroy() {
        if (searchTimer) global.clearTimeout(searchTimer);
        global.removeEventListener("keydown", onKeydown);
        queuePanel.removeEventListener("scroll", onQueueScroll);
        unsubscribeCases();
        unsubscribeStore();
        cancelMediaPrefetch();
        prefetchedImages.forEach((img) => img.removeAttribute("src"));
        prefetchedImages.clear();
        queueRows.clear();
        annotationFields.clear();
        validationSummary = null;
        activeGallery = null;
        closeGallery = null;
        activeTextView = null;
        closeTextView = null;
        main.textContent = "";
      },
      refresh: renderWorkspace,
    });
  }

  modules.renderer = Object.freeze({ createRenderer, parsePrediction });
})(
  globalThis.__EVALUATION_REPORT_MODULES__ =
    globalThis.__EVALUATION_REPORT_MODULES__ || {}
);
