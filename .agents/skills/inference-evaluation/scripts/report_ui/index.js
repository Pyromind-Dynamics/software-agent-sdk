(function (global, modules) {
  "use strict";

  const annotation = modules.annotationStore;
  const cases = modules.caseSource;
  const renderer = modules.renderer;
  if (!annotation || !cases || !renderer) {
    throw new Error("EvaluationReport bundle is incomplete");
  }

  function createHttpStorage(apiConfig) {
    return annotation.createHttpStorage(global, apiConfig);
  }

  function mount(container, options) {
    const settings = options || {};
    const document = settings.document || global.document;
    const dataElement = settings.dataElement ||
      document.getElementById("report-data");
    const data = settings.report || JSON.parse(dataElement.textContent);
    const main = container || document.getElementById("report");
    if (!main) throw new Error("EvaluationReport requires a container");
    main.classList.add("evaluation-report");

    const apiConfig = settings.apiConfig === undefined
      ? global.__EVAL_REPORT_API__ || null
      : settings.apiConfig;
    const storage = settings.storage || createHttpStorage(apiConfig);
    const store = annotation.createAnnotationStore({
      data,
      global,
      storage,
    });
    const caseSource = cases.createCaseSource({ data, global, storage, store });
    const view = renderer.createRenderer({
      caseSource,
      container: main,
      data,
      document,
      global,
      store,
      storage,
    });

    function beforeUnload(event) {
      if (store.hasPending()) {
        event.preventDefault();
        event.returnValue = "";
      }
    }

    global.addEventListener("beforeunload", beforeUnload);
    if (store.online && store.unlocked) store.load();

    return Object.freeze({
      destroy() {
        global.removeEventListener("beforeunload", beforeUnload);
        view.destroy();
        caseSource.destroy();
        store.destroy();
      },
      refresh: view.refresh,
    });
  }

  global.EvaluationReport = Object.freeze({
    createHttpStorage,
    mount,
    version: "1.0.0",
  });
  delete global.__EVALUATION_REPORT_MODULES__;
})(window, globalThis.__EVALUATION_REPORT_MODULES__ || {});
