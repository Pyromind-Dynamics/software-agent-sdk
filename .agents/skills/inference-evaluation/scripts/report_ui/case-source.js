(function (modules) {
  "use strict";

  const PAGE_SIZE = 20;
  const MAX_CACHED_PAGES = 3;
  const PREFETCH_REMAINING = 5;
  const MAX_RETRY_DELAY = 30000;

  function createCaseSource(options) {
    const data = options.data;
    const global = options.global || globalThis;
    const storage = options.storage;
    const store = options.store;
    const listeners = [];
    const items = [];
    const itemById = {};
    const pages = [];
    const pageCache = new Map();
    const cursorByStart = new Map([[0, null]]);
    let total = 0;
    let filteredTotal = 0;
    let query = "";
    let reviewStatus = "all";
    let loading = false;
    let error = null;
    let generation = 0;
    let inflightPromise = null;
    let lastFailedRequest = null;
    let retryDelay = 1000;
    let retryTimer = null;
    let idleHandle = null;
    let requestController = null;

    const paged = !!(storage && typeof storage.loadCases === "function");

    function emit() {
      listeners.slice().forEach((listener) => listener());
    }

    function replaceLegacyItems(nextItems) {
      items.length = 0;
      Object.keys(itemById).forEach((id) => delete itemById[id]);
      nextItems.forEach((item) => {
        if (!item || typeof item.id !== "string") return;
        items.push(item);
        itemById[item.id] = item;
      });
      store.registerCases(nextItems);
    }

    function rebuildItems() {
      items.length = 0;
      Object.keys(itemById).forEach((id) => delete itemById[id]);
      pages.sort((left, right) => left.start - right.start);
      pages.forEach((page) => {
        page.items.forEach((item) => {
          items.push(item);
          itemById[item.id] = item;
        });
      });
    }

    function removePage(page) {
      store.releaseCases(page.items);
      const index = pages.indexOf(page);
      if (index >= 0) pages.splice(index, 1);
    }

    function addPage(page, direction) {
      const existing = pages.find((candidate) => candidate.start === page.start);
      if (existing) removePage(existing);
      pages.push(page);
      pageCache.delete(page.cacheKey);
      pageCache.set(page.cacheKey, page);
      store.registerCases(page.items, page.revision);
      while (pages.length > MAX_CACHED_PAGES) {
        const ordered = pages.slice().sort((left, right) => left.start - right.start);
        const removed = direction === "previous" ? ordered[ordered.length - 1] : ordered[0];
        pageCache.delete(removed.cacheKey);
        removePage(removed);
      }
      while (pageCache.size > MAX_CACHED_PAGES) {
        const evicted = Array.from(pageCache.values()).find((entry) => !pages.includes(entry));
        if (!evicted) break;
        pageCache.delete(evicted.cacheKey);
      }
      rebuildItems();
    }

    function cancelRetry() {
      if (retryTimer !== null) global.clearTimeout(retryTimer);
      retryTimer = null;
    }

    function scheduleRetry(request) {
      cancelRetry();
      const delay = retryDelay;
      retryDelay = Math.min(retryDelay * 2, MAX_RETRY_DELAY);
      retryTimer = global.setTimeout(() => {
        retryTimer = null;
        loadPage(request);
      }, delay);
    }

    function loadPage(request) {
      if (!paged) return Promise.resolve(false);
      if (loading) return inflightPromise || Promise.resolve(false);
      const cacheKey = JSON.stringify([query, reviewStatus, request.start]);
      const cached = pageCache.get(cacheKey);
      if (!request.refresh && cached && Number.isInteger(cached.revision) &&
          cached.revision === store.revision) {
        addPage(cached, request.direction);
        cursorByStart.set(request.start, request.cursor);
        if (cached.nextCursor) {
          cursorByStart.set(request.start + cached.items.length, cached.nextCursor);
        }
        total = cached.total;
        filteredTotal = cached.filteredTotal;
        error = null;
        lastFailedRequest = null;
        retryDelay = 1000;
        cancelRetry();
        emit();
        return Promise.resolve(true);
      }
      const requestGeneration = generation;
      const controller = typeof global.AbortController === "function"
        ? new global.AbortController() : null;
      requestController = controller;
      loading = true;
      error = null;
      emit();
      inflightPromise = (async () => {
        let responseStatus = null;
        try {
          const response = await storage.loadCases({
            cursor: request.cursor,
            limit: PAGE_SIZE,
            query,
            reviewStatus,
            signal: controller ? controller.signal : undefined,
          });
          responseStatus = response.status;
          if (!response.ok) throw new Error("HTTP " + response.status);
          const payload = await response.json();
          if (requestGeneration !== generation) return false;
          if (Number.isInteger(payload.revision) &&
              Number.isInteger(store.revision) && payload.revision < store.revision) {
            throw new Error("Page annotations are out of date; reloading");
          }
          const pageItems = payload.items || [];
          if (!pageItems.length && request.refresh && request.start > 0) {
            lastFailedRequest = {
              start: Math.max(0, request.start - PAGE_SIZE),
              cursor: cursorByStart.get(Math.max(0, request.start - PAGE_SIZE)),
              direction: "previous",
              refresh: true,
            };
            scheduleRetry(lastFailedRequest);
            return false;
          }
          cursorByStart.set(request.start, request.cursor);
          if (payload.next_cursor) {
            cursorByStart.set(request.start + pageItems.length, payload.next_cursor);
          }
          addPage({
            start: request.start,
            items: pageItems,
            nextCursor: payload.next_cursor || null,
            revision: payload.revision,
            cacheKey,
            total: Number(payload.total) || 0,
            filteredTotal: Number(payload.filtered_total) || 0,
          }, request.direction);
          total = Number(payload.total) || 0;
          filteredTotal = Number(payload.filtered_total) || 0;
          retryDelay = 1000;
          lastFailedRequest = null;
          cancelRetry();
          return true;
        } catch (reason) {
          if (requestGeneration === generation) {
            error = reason;
            const start = request.refresh && request.start > 0 &&
              responseStatus === 422
              ? Math.max(0, request.start - PAGE_SIZE) : request.start;
            lastFailedRequest = start === request.start ? request : {
              ...request, start, cursor: cursorByStart.get(start),
            };
            scheduleRetry(lastFailedRequest);
          }
          return false;
        } finally {
          if (requestGeneration === generation) {
            loading = false;
            inflightPromise = null;
            requestController = null;
            emit();
          }
        }
      })();
      return inflightPromise;
    }

    function loadNext() {
      if (!pages.length) {
        return loadPage({ cursor: null, start: 0, direction: "next" });
      }
      const last = pages.reduce((left, right) =>
        left.start > right.start ? left : right);
      if (!last.nextCursor) return Promise.resolve(false);
      return loadPage({
        cursor: last.nextCursor,
        start: last.start + last.items.length,
        direction: "next",
      });
    }

    function loadPrevious() {
      if (!pages.length) return Promise.resolve(false);
      const first = pages.reduce((left, right) =>
        left.start < right.start ? left : right);
      if (first.start === 0) return Promise.resolve(false);
      const start = Math.max(0, first.start - PAGE_SIZE);
      if (!cursorByStart.has(start)) return Promise.resolve(false);
      return loadPage({
        cursor: cursorByStart.get(start),
        start,
        direction: "previous",
      });
    }

    function resetSelection(anchorId) {
      const anchorPage = pages.find((page) =>
        page.items.some((item) => item.id === anchorId));
      const start = anchorPage ? anchorPage.start : 0;
      const cursor = cursorByStart.get(start) || null;
      generation += 1;
      if (requestController) requestController.abort();
      requestController = null;
      loading = false;
      inflightPromise = null;
      error = null;
      lastFailedRequest = null;
      retryDelay = 1000;
      cancelRetry();
      cancelPrefetch();
      if (paged) {
        pages.slice().forEach(removePage);
        rebuildItems();
        if (!anchorPage) {
          cursorByStart.clear();
          cursorByStart.set(0, null);
        }
        filteredTotal = 0;
        loadPage({ cursor, start, direction: "next", refresh: !!anchorPage });
      } else {
        replaceLegacyItems((data.cases || []).filter(
          (item) =>
            (!query || item.id.toLowerCase().includes(query)) &&
            (reviewStatus === "all" ||
              store.badgeState(item.id) === reviewStatus)));
        total = (data.cases || []).length;
        filteredTotal = items.length;
        emit();
      }
      return start;
    }

    function setQuery(value) {
      const normalized = value.trim().toLowerCase();
      if (normalized === query) return;
      query = normalized;
      resetSelection();
    }

    function setReviewStatus(value) {
      if (value === reviewStatus) return;
      reviewStatus = value;
      resetSelection();
    }

    function runWhenIdle(callback) {
      if (idleHandle !== null) return;
      if (typeof global.requestIdleCallback === "function") {
        idleHandle = global.requestIdleCallback(() => {
          idleHandle = null;
          callback();
        }, { timeout: 500 });
      } else {
        idleHandle = global.setTimeout(() => {
          idleHandle = null;
          callback();
        }, 0);
      }
    }

    function cancelPrefetch() {
      if (idleHandle === null) return;
      if (typeof global.cancelIdleCallback === "function") {
        global.cancelIdleCallback(idleHandle);
      } else global.clearTimeout(idleHandle);
      idleHandle = null;
    }

    function ensureAhead(index) {
      if (
        paged && pages.length &&
        items.length - index - 1 <= PREFETCH_REMAINING
      ) {
        runWhenIdle(loadNext);
      }
    }

    async function move(currentId, delta) {
      let index = items.findIndex((item) => item.id === currentId);
      if (index < 0) index = 0;
      let target = index + delta;
      if (target >= 0 && target < items.length) return items[target].id;
      if (target < 0 && paged) await loadPrevious();
      else if (target >= items.length && paged) await loadNext();
      index = items.findIndex((item) => item.id === currentId);
      target = index + delta;
      return target >= 0 && target < items.length ? items[target].id : currentId;
    }

    if (paged) loadNext();
    else {
      replaceLegacyItems(data.cases || []);
      total = items.length;
      filteredTotal = items.length;
    }

    return Object.freeze({
      destroy() {
        generation += 1;
        if (requestController) requestController.abort();
        pageCache.clear();
        cancelRetry();
        cancelPrefetch();
      },
      ensureAhead,
      get error() { return error; },
      get filteredTotal() { return filteredTotal; },
      get items() { return items.slice(); },
      get loading() { return loading; },
      move,
      get paged() { return paged; },
      refresh: resetSelection,
      retry() {
        cancelRetry();
        return lastFailedRequest
          ? loadPage(lastFailedRequest)
          : Promise.resolve(false);
      },
      setQuery,
      setReviewStatus,
      get startOffset() { return pages.length ? pages[0].start : 0; },
      subscribe(listener) {
        listeners.push(listener);
        return () => {
          const index = listeners.indexOf(listener);
          if (index >= 0) listeners.splice(index, 1);
        };
      },
      get total() { return total; },
    });
  }

  modules.caseSource = Object.freeze({ createCaseSource });
})(
  globalThis.__EVALUATION_REPORT_MODULES__ =
    globalThis.__EVALUATION_REPORT_MODULES__ || {}
);
