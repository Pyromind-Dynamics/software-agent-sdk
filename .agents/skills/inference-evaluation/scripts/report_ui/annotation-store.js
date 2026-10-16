(function (modules) {
  "use strict";

  const SCHEMA_VERSION = 1;
  const DECISIONS = [
    ["confirmed_badcase", "Confirmed badcase"],
    ["evaluation_false_alarm", "Evaluation false alarm"],
    ["needs_discussion", "Needs discussion"],
  ];
  const RESOLUTIONS = [
    ["gt_error", "GT 错误，需要修标", "evaluation_false_alarm", "fix_ground_truth"],
    ["poor_data", "数据质量差，需要清洗或剔除", "evaluation_false_alarm", "needs_analysis"],
    ["model_gap", "模型能力不足，需要训练优化", "confirmed_badcase", "training_workflow_adjustment"],
    ["coverage_gap", "某类覆盖不足，需要采集或合成", "confirmed_badcase", "data_augmentation"],
    ["acceptable", "业务可接受，无需处理", "evaluation_false_alarm", "no_action"],
    ["uncertain", "尚不能确定，需要进一步核查", "needs_discussion", "needs_analysis"],
  ];
  const FAILURE_MODES = [
    "model_miss", "model_false_positive", "classification_error",
    "severity_misclassification", "gt_issue", "rubric_issue",
    "inference_input_issue", "report_issue", "other",
  ];
  const SEVERITIES = ["P0", "P1", "P2", "P3"];
  const DATA_QUALITY_REASONS = [
    "needs_cleaning", "exclude", "missing", "format_error", "corrupt", "other",
  ];
  const BASE_DISPOSITIONS = [
    "fix_ground_truth", "data_augmentation",
    "training_workflow_adjustment", "evaluation_rule_fix",
    "needs_analysis", "no_action",
  ];
  const MAX_COMMENT = 2000;
  const MAX_CORRECTED = 2000;
  const MAX_LABELS = 50;

  function labelList(value) {
    return Array.isArray(value) && value.length &&
      value.every((item) => typeof item === "string" && item.trim())
      ? value.slice() : null;
  }

  function createHttpStorage(global, apiConfig) {
    if (!apiConfig || !apiConfig.annotations) return null;
    const storage = {
      loadAnnotations() {
        return global.fetch(apiConfig.annotations, {
          credentials: "same-origin",
        });
      },
      saveAnnotations(payload) {
        return global.fetch(apiConfig.annotations, {
          method: "PUT",
          credentials: "same-origin",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
      },
      saveAnnotationChanges(payload) {
        return global.fetch(apiConfig.annotations, {
          method: "PATCH",
          credentials: "same-origin",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
      },
    };
    if (apiConfig.annotation_summary) {
      storage.loadAnnotationSummary = () =>
        global.fetch(apiConfig.annotation_summary, {
          credentials: "same-origin",
        });
    }
    if (apiConfig.exports) {
      storage.exportUrl = (filename) => apiConfig.exports + filename;
    }
    if (apiConfig.cases) {
      storage.loadCases = (options) => {
        const params = ["limit=" + encodeURIComponent(options.limit)];
        if (options.cursor) {
          params.push("cursor=" + encodeURIComponent(options.cursor));
        }
        if (options.query) {
          params.push("query=" + encodeURIComponent(options.query));
        }
        if (options.reviewStatus && options.reviewStatus !== "all") {
          params.push(
            "review_status=" + encodeURIComponent(options.reviewStatus),
          );
        }
        return global.fetch(apiConfig.cases + "?" + params.join("&"), {
          credentials: "same-origin",
          signal: options.signal,
        });
      };
    }
    if (apiConfig.media) {
      storage.mediaUrl = (mediaId) =>
        apiConfig.media + encodeURIComponent(mediaId);
    }
    return Object.freeze(storage);
  }

  function createAnnotationStore(options) {
    const global = options.global;
    const data = options.data;
    const caseIds = [];
    const caseById = {};
    const serverRecordById = {};
    const loadedCaseIds = new Set();
    const storage = options.storage;
    const annotationEnabled = (data.panels || []).includes("annotation");
    const unlocked = data.evaluation_status === "validated";
    const online = !!(
      storage &&
      (typeof storage.loadAnnotationSummary === "function" ||
        typeof storage.loadAnnotations === "function") &&
      (typeof storage.saveAnnotationChanges === "function" ||
        typeof storage.saveAnnotations === "function") &&
      annotationEnabled
    );
    const usesPatchStorage = !!(
      storage && typeof storage.saveAnnotationChanges === "function");
    const usesIncrementalStorage = !!(
      storage && typeof storage.loadAnnotationSummary === "function" &&
      usesPatchStorage);
    const storageKey = "evaluation-annotations:" + data.run_id + ":" +
      data.evaluation_config_sha256;
    const listeners = [];

    let serverState = null;
    let unconfirmed = readUnconfirmedBuffer();
    let annotations = {};
    let serverRevision = null;
    let saveState = online && unlocked ? "loading" : "readonly";
    let overallStatus = "";
    let serverCounts = { reviewed: 0, in_progress: 0, unreviewed: 0 };
    let inflight = null;
    let pendingEdit = false;
    let saveTimer = null;
    let retryTimer = null;
    let retryDelay = 1000;

    registerCases(data.cases || []);

    function emit(type, detail) {
      listeners.slice().forEach((listener) => listener({ type, detail }));
    }

    function registerCases(items, revision) {
      items.forEach((item) => {
        if (!item || typeof item.id !== "string") return;
        loadedCaseIds.add(item.id);
        if (typeof item.review_status !== "string" &&
            !Object.prototype.hasOwnProperty.call(caseById, item.id)) {
          caseIds.push(item.id);
        }
        caseById[item.id] = item;
        if (typeof item.review_status === "string" &&
            !(Number.isInteger(revision) && serverRevision !== null &&
              revision < serverRevision)) {
          serverRecordById[item.id] = {
            case_id: item.id,
            review_status: item.review_status,
            human_annotation: item.human_annotation || null,
          };
        }
      });
      rebuildView();
    }

    function pruneReleasedCase(id) {
      if (!usesPatchStorage || loadedCaseIds.has(id) || unconfirmed[id]) return;
      delete caseById[id];
      delete serverRecordById[id];
    }

    function releaseCases(items) {
      items.forEach((item) => {
        if (!item || typeof item.id !== "string") return;
        loadedCaseIds.delete(item.id);
        pruneReleasedCase(item.id);
      });
      rebuildView();
    }

    function readUnconfirmedBuffer() {
      try {
        const raw = global.localStorage.getItem(storageKey);
        const parsed = raw ? JSON.parse(raw) : {};
        if (!parsed || typeof parsed !== "object") return {};
        const valid = {};
        Object.keys(parsed).forEach((caseId) => {
          const entry = parsed[caseId];
          if (!entry || typeof entry.updated_at !== "string") return;
          if (entry.cleared === true) valid[caseId] = entry;
          else if (entry.annotation && typeof entry.annotation === "object") {
            valid[caseId] = entry;
          }
        });
        return valid;
      } catch (error) {
        return {};
      }
    }

    function writeUnconfirmedBuffer() {
      try {
        global.localStorage.setItem(storageKey, JSON.stringify(unconfirmed));
      } catch (error) {
        // The buffer is only best-effort crash recovery.
      }
    }

    function hasUnconfirmed() {
      return Object.keys(unconfirmed).length > 0;
    }

    function blankAnnotation() {
      return {
        resolution_code: "", decision: "", failure_modes: [], taxonomy_labels: [],
        resolution_inferred: false, resolution_details_version: 0,
        disposition: "", severity: "", comment: "",
        corrected_answer: "", data_quality_reasons: [],
        training_suggestion: "", missing_scenario: "",
        target_sample_count: null, acceptance_reason: "",
        verification_items: "", confirmation_owner: "", updated_at: null,
      };
    }

    function resolutionCode(source) {
      if (typeof source.resolution_code === "string") {
        return source.resolution_code;
      }
      const match = RESOLUTIONS.find((entry) =>
        entry[2] === source.decision && entry[3] === source.disposition);
      if (match) return match[0];
      if (source.decision === "needs_discussion") return "uncertain";
      return "";
    }

    function normalizeAnnotation(source) {
      const hasExplicitResolution = typeof source.resolution_code === "string";
      const normalizedResolution = resolutionCode(source);
      return {
        resolution_code: normalizedResolution,
        resolution_inferred: !hasExplicitResolution && !!normalizedResolution,
        resolution_details_version:
          source.resolution_details_version === 1 ? 1 : 0,
        decision: typeof source.decision === "string" ? source.decision : "",
        failure_modes: Array.isArray(source.failure_modes)
          ? source.failure_modes.slice() : [],
        taxonomy_labels: Array.isArray(source.taxonomy_labels)
          ? source.taxonomy_labels.slice() : [],
        disposition: typeof source.disposition === "string"
          ? source.disposition : "",
        severity: typeof source.severity === "string" ? source.severity : "",
        comment: typeof source.comment === "string" ? source.comment : "",
        corrected_answer: typeof source.corrected_answer === "string"
          ? source.corrected_answer : "",
        data_quality_reasons: Array.isArray(source.data_quality_reasons)
          ? source.data_quality_reasons.slice() : [],
        training_suggestion: typeof source.training_suggestion === "string"
          ? source.training_suggestion : "",
        missing_scenario: typeof source.missing_scenario === "string"
          ? source.missing_scenario : "",
        target_sample_count: Number.isInteger(source.target_sample_count)
          ? source.target_sample_count : null,
        acceptance_reason: typeof source.acceptance_reason === "string"
          ? source.acceptance_reason : "",
        verification_items: typeof source.verification_items === "string"
          ? source.verification_items : "",
        confirmation_owner: typeof source.confirmation_owner === "string"
          ? source.confirmation_owner : "",
        updated_at: source.updated_at || new Date().toISOString(),
      };
    }

    function rebuildView() {
      const next = {};
      Object.values(serverRecordById).forEach((record) => {
        if (record.human_annotation) {
          next[record.case_id] = normalizeAnnotation(record.human_annotation);
        }
      });
      Object.keys(unconfirmed).forEach((caseId) => {
        const entry = unconfirmed[caseId];
        if (entry.cleared) delete next[caseId];
        else next[caseId] = normalizeAnnotation(entry.annotation);
      });
      annotations = next;
    }

    function reviewedCount() {
      let reviewed = serverCounts.reviewed;
      Object.keys(unconfirmed).forEach((caseId) => {
        const previous = serverRecordById[caseId];
        const wasReviewed = previous && previous.review_status === "reviewed";
        const entry = unconfirmed[caseId];
        const isNowReviewed = !!(
          entry && !entry.cleared && annotationComplete(entry.annotation)
        );
        if (wasReviewed !== isNowReviewed) reviewed += isNowReviewed ? 1 : -1;
      });
      return Math.max(0, reviewed);
    }

    function annotationComplete(annotation) {
      if (!annotation || !annotation.decision || !annotation.disposition) {
        return false;
      }
      if (annotation.decision === "confirmed_badcase") {
        if (!annotation.failure_modes || !annotation.failure_modes.length) {
          return false;
        }
        if (!annotation.severity) return false;
      }
      if (annotation.resolution_details_version === 1 &&
          annotation.resolution_code === "gt_error" &&
          !annotation.corrected_answer) return false;
      if (annotation.resolution_details_version === 1 &&
          annotation.resolution_code === "poor_data" &&
          !annotation.data_quality_reasons.length) return false;
      if (annotation.resolution_details_version === 1 &&
          annotation.resolution_code === "coverage_gap" &&
          !annotation.missing_scenario) return false;
      if (annotation.resolution_details_version === 1 &&
          annotation.resolution_code === "acceptable" &&
          !annotation.acceptance_reason) return false;
      if (annotation.resolution_details_version === 1 &&
          annotation.resolution_code === "uncertain" &&
          !annotation.verification_items) return false;
      return true;
    }

    function isReviewed(id) {
      return Object.prototype.hasOwnProperty.call(annotations, id);
    }

    function isComplete(id) {
      return isReviewed(id) && annotationComplete(annotations[id]);
    }

    function badgeState(id) {
      if (!isReviewed(id)) return "unreviewed";
      return isComplete(id) ? "reviewed" : "in_progress";
    }

    function getAnnotation(id) {
      return annotations[id] || blankAnnotation();
    }

    function buildHumanAnnotation(annotation) {
      const value = {
        decision: annotation.decision,
        failure_modes: annotation.failure_modes || [],
        taxonomy_labels: annotation.taxonomy_labels || [],
        disposition: annotation.disposition,
        severity: annotation.severity || null,
        comment: annotation.comment || "",
        corrected_answer: annotation.corrected_answer || null,
        data_quality_reasons: annotation.data_quality_reasons || [],
        training_suggestion: annotation.training_suggestion || null,
        missing_scenario: annotation.missing_scenario || null,
        target_sample_count: annotation.target_sample_count || null,
        acceptance_reason: annotation.acceptance_reason || null,
        verification_items: annotation.verification_items || null,
        confirmation_owner: annotation.confirmation_owner || null,
        reviewer_id: null,
        updated_at: annotation.updated_at,
      };
      if (annotation.resolution_code && !annotation.resolution_inferred) {
        value.resolution_code = annotation.resolution_code;
      }
      if (annotation.resolution_details_version === 1) {
        value.resolution_details_version = 1;
      }
      return value;
    }

    function buildRecord(id) {
      const item = caseById[id] || null;
      const previous = serverRecordById[id] || null;
      const annotation = annotations[id] || null;
      const failureTags = item
        ? (item.rubric_items || [])
            .filter((rubric) => !rubric.passed)
            .map((rubric) => rubric.name)
        : [];
      return {
        schema_version: SCHEMA_VERSION,
        run_id: data.run_id,
        evaluation_config_sha256: data.evaluation_config_sha256,
        case_id: id,
        review_status: annotation
          ? (isComplete(id) ? "reviewed" : "in_progress")
          : "unreviewed",
        machine_result: item ? {
          passed: item.status === "passed",
          reference: item.reference,
          prediction: item.prediction,
          overall_score: item.overall_score,
          failure_tags: failureTags,
        } : previous && previous.machine_result || null,
        human_annotation: annotation ? buildHumanAnnotation(annotation) : null,
      };
    }

    function currentRecords() {
      const ids = Object.keys(serverRecordById);
      const knownIds = new Set(ids);
      caseIds.forEach((id) => {
        if (!knownIds.has(id)) ids.push(id);
      });
      return ids.map(buildRecord);
    }

    function uuid() {
      return (global.crypto && global.crypto.randomUUID)
        ? global.crypto.randomUUID()
        : "save-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    }

    function scheduleSave() {
      if (!online || !unlocked || saveState === "loading") return;
      if (serverRevision === null) return;
      if (inflight) {
        pendingEdit = true;
        return;
      }
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      if (saveTimer) clearTimeout(saveTimer);
      saveState = "saving";
      emit("state");
      saveTimer = setTimeout(fireSave, 800);
    }

    function fireSave() {
      if (serverRevision === null) return;
      if (inflight) {
        pendingEdit = true;
        return;
      }
      if (saveTimer) {
        clearTimeout(saveTimer);
        saveTimer = null;
      }
      const changedIds = Object.keys(unconfirmed);
      const stamps = {};
      changedIds.forEach((id) => {
        stamps[id] = unconfirmed[id] ? unconfirmed[id].updated_at : null;
      });
      inflight = {
        save_id: uuid(),
        base_revision: serverRevision,
        records: usesPatchStorage ? null : currentRecords(),
        changes: changedIds.map((id) => {
          const record = buildRecord(id);
          return {
            case_id: id,
            human_annotation: record.human_annotation,
          };
        }),
        stamps,
      };
      pendingEdit = false;
      putSave(inflight, 1000);
    }

    function putSave(job, backoff) {
      saveState = "saving";
      emit("state");
      const save = usesPatchStorage
        ? storage.saveAnnotationChanges({
            save_id: job.save_id,
            base_revision: job.base_revision,
            changes: job.changes,
          })
        : storage.saveAnnotations({
            save_id: job.save_id,
            base_revision: job.base_revision,
            records: job.records,
          });
      save.then(async (response) => {
        if (response.status === 409) {
          await handleConflict(job, response);
          return;
        }
        if (response.status === 422) {
          inflight = null;
          saveState = "rejected";
          let detail = "your changes were rejected";
          try {
            const payload = await response.json();
            if (payload && payload.detail) {
              detail = JSON.stringify(payload.detail);
            }
          } catch (error) {
            // Keep the generic rejection message.
          }
          emit("notice", "Your changes were not saved: " + detail);
          emit("state");
          return;
        }
        if (!response.ok) throw new Error("HTTP " + response.status);
        const result = await response.json();
        if (result.superseded_by) {
          await adoptAfterConflict(
            job,
            "your save landed as revision " + result.revision +
              " but has been superseded by " + result.superseded_by
          );
          return;
        }
        serverState = {
          revision: result.revision,
          overall_status: result.overall_status,
        };
        if (job.records) {
          Object.keys(serverRecordById).forEach(
            (id) => delete serverRecordById[id]);
          job.records.forEach((record) => {
            serverRecordById[record.case_id] = record;
          });
        } else {
          job.changes.forEach((change) => {
            const annotation = change.human_annotation;
            serverRecordById[change.case_id] = {
              case_id: change.case_id,
              review_status: annotation
                ? (annotationComplete(annotation) ? "reviewed" : "in_progress")
                : "unreviewed",
              human_annotation: annotation,
            };
          });
        }
        serverCounts = {
          reviewed: result.reviewed,
          in_progress: result.in_progress,
          unreviewed: result.unreviewed,
        };
        const stamps = job.stamps || {};
        Object.keys(stamps).forEach((caseId) => {
          const entry = unconfirmed[caseId];
          const stamp = stamps[caseId];
          if (!entry && stamp === null) return;
          if (entry && entry.updated_at === stamp) delete unconfirmed[caseId];
          pruneReleasedCase(caseId);
        });
        writeUnconfirmedBuffer();
        rebuildView();
        inflight = null;
        retryDelay = 1000;
        serverRevision = result.revision;
        overallStatus = result.overall_status;
        emit("notice", null);
        if (pendingEdit || hasUnconfirmed()) {
          pendingEdit = false;
          fireSave();
          return;
        }
        saveState = "saved";
        emit("state");
      }).catch(() => {
        inflight = job;
        saveState = "error";
        emit("notice", null);
        if (retryTimer) clearTimeout(retryTimer);
        retryTimer = setTimeout(
          () => putSave(job, Math.min(backoff * 2, 30000)), backoff);
        emit("state");
      });
    }

    async function handleConflict(job, response) {
      let detail = "";
      try {
        const payload = await response.json();
        detail = typeof payload.detail === "string" ? payload.detail : "";
      } catch (error) {
        // The generic conflict message is sufficient.
      }
      await adoptAfterConflict(job, detail);
    }

    async function adoptAfterConflict(job, detail) {
      inflight = job;
      saveState = "conflict";
      emit("state");
      try {
        const latest = await storage.loadAnnotations();
        if (!latest.ok) throw new Error("HTTP " + latest.status);
        adoptServerState(await latest.json());
        inflight = null;
        retryDelay = 1000;
        emit(
          "notice",
          "Your edits were based on an older revision" +
            (detail ? " (" + detail + ")" : "") +
            ". The latest saved annotations are now shown; your unsaved edits" +
            " are being saved on top of it."
        );
        if (pendingEdit || hasUnconfirmed()) {
          pendingEdit = false;
          scheduleSave();
          return;
        }
        saveState = "saved";
        emit("state");
      } catch (error) {
        emit("notice", null);
        if (retryTimer) clearTimeout(retryTimer);
        const backoff = Math.min(retryDelay * 2, 30000);
        retryDelay = backoff;
        retryTimer = setTimeout(() => adoptAfterConflict(job, detail), backoff);
        emit("state");
      }
    }

    function adoptServerState(payload) {
      serverState = payload;
      serverRevision = payload.revision;
      Object.keys(serverRecordById).forEach((id) => delete serverRecordById[id]);
      (payload.records || []).forEach((record) => {
        serverRecordById[record.case_id] = record;
      });
      Object.keys(serverRecordById).forEach(pruneReleasedCase);
      serverCounts = {
        reviewed: Number(payload.reviewed) || 0,
        in_progress: Number(payload.in_progress) || 0,
        unreviewed: Number(payload.unreviewed) || 0,
      };
      rebuildView();
      writeUnconfirmedBuffer();
      overallStatus = payload.overall_status || overallStatus;
      emit("replace");
    }

    function adoptServerSummary(payload) {
      serverState = payload;
      serverRevision = payload.revision;
      serverCounts = {
        reviewed: Number(payload.reviewed) || 0,
        in_progress: Number(payload.in_progress) || 0,
        unreviewed: Number(payload.unreviewed) || 0,
      };
      overallStatus = payload.overall_status || overallStatus;
      rebuildView();
      writeUnconfirmedBuffer();
      emit("replace");
    }

    function load() {
      saveState = "loading";
      emit("state");
      const loadState = usesIncrementalStorage
        ? storage.loadAnnotationSummary()
        : storage.loadAnnotations();
      loadState
        .then(async (response) => {
          if (!response.ok) throw new Error("HTTP " + response.status);
          const payload = await response.json();
          if (usesIncrementalStorage) adoptServerSummary(payload);
          else adoptServerState(payload);
          emit("notice", null);
          saveState = "saved";
          if (hasUnconfirmed()) scheduleSave();
          emit("state");
        })
        .catch(() => {
          saveState = "error";
          emit("notice", null);
          emit("load-error");
          emit("state");
        });
    }

    function commit(id, patch) {
      const current = annotations[id]
        ? Object.assign({}, annotations[id]) : blankAnnotation();
      Object.assign(current, patch);
      current.updated_at = new Date().toISOString();
      unconfirmed[id] = {
        updated_at: current.updated_at,
        annotation: current,
      };
      rebuildView();
      writeUnconfirmedBuffer();
      emit("change", id);
      scheduleSave();
    }

    function clear(id) {
      unconfirmed[id] = {
        updated_at: new Date().toISOString(),
        cleared: true,
      };
      rebuildView();
      writeUnconfirmedBuffer();
      emit("change", id);
      scheduleSave();
    }

    function subscribe(listener) {
      listeners.push(listener);
      return () => {
        const index = listeners.indexOf(listener);
        if (index >= 0) listeners.splice(index, 1);
      };
    }

    function destroy() {
      if (saveTimer) clearTimeout(saveTimer);
      if (retryTimer) clearTimeout(retryTimer);
      listeners.length = 0;
    }

    return Object.freeze({
      badgeState,
      clear,
      commit,
      destroy,
      exportUrl: storage && storage.exportUrl
        ? (filename) => storage.exportUrl(filename) : null,
      getAnnotation,
      get overallStatus() { return overallStatus; },
      get revision() { return serverRevision; },
      get saveState() { return saveState; },
      get online() { return online; },
      get unlocked() { return unlocked; },
      hasPending() {
        return !!(inflight || saveTimer || pendingEdit || hasUnconfirmed());
      },
      isComplete,
      isReviewed,
      load,
      registerCases,
      releaseCases,
      reviewedCount,
      subscribe,
    });
  }

  modules.annotationStore = Object.freeze({
    BASE_DISPOSITIONS,
    DATA_QUALITY_REASONS,
    DECISIONS,
    FAILURE_MODES,
    MAX_COMMENT,
    MAX_CORRECTED,
    MAX_LABELS,
    SEVERITIES,
    RESOLUTIONS,
    createAnnotationStore,
    createHttpStorage,
    labelList,
  });
})(
  globalThis.__EVALUATION_REPORT_MODULES__ =
    globalThis.__EVALUATION_REPORT_MODULES__ || {}
);
