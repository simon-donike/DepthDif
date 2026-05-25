(function () {
  const CESIUM_JS_URL =
    "https://cesium.com/downloads/cesiumjs/releases/1.140/Build/Cesium/Cesium.js";
  const CESIUM_CSS_URL =
    "https://cesium.com/downloads/cesiumjs/releases/1.140/Build/Cesium/Widgets/widgets.css";
  const LOADER_SCRIPT_NAME = "load-temporal-globe.js";
  const CESIUM_APP_SCRIPT_PATH = "temporal-globe.js";
  const CESIUM_JS_SCRIPT_ID = "depthdif-temporal-cesium-js";
  const CESIUM_CSS_LINK_ID = "depthdif-temporal-cesium-css";
  const CESIUM_APP_SCRIPT_ID = "depthdif-temporal-globe-app";
  const MOBILE_BLOCK_MEDIA_QUERY = "(max-width: 900px), (pointer: coarse) and (max-width: 1024px)";

  function resolveLoaderScriptBaseUrl() {
    if (document.currentScript && document.currentScript.src) {
      return new URL(".", document.currentScript.src).toString();
    }

    const scripts = document.querySelectorAll("script[src]");
    for (let index = scripts.length - 1; index >= 0; index -= 1) {
      const script = scripts[index];
      const src = script.getAttribute("src");
      if (!src) {
        continue;
      }
      const scriptUrl = new URL(src, document.baseURI);
      if (scriptUrl.pathname.endsWith("/" + LOADER_SCRIPT_NAME)) {
        return new URL(".", scriptUrl).toString();
      }
    }
    return new URL("/javascripts/", document.baseURI).toString();
  }

  const LOADER_SCRIPT_BASE_URL = resolveLoaderScriptBaseUrl();

  function isLikelyIPad() {
    const navigatorObject = window.navigator;
    if (!navigatorObject) {
      return false;
    }
    const userAgent = String(navigatorObject.userAgent || "");
    const platform = String(navigatorObject.platform || "");
    const maxTouchPoints = Number(navigatorObject.maxTouchPoints || 0);
    return /iPad/.test(userAgent) || (platform === "MacIntel" && maxTouchPoints > 1);
  }

  function shouldBlockOnMobile() {
    if (isLikelyIPad()) {
      return true;
    }
    if (typeof window.matchMedia !== "function") {
      return false;
    }
    return window.matchMedia(MOBILE_BLOCK_MEDIA_QUERY).matches;
  }

  function setMobileBlockVisible(visible) {
    const mobileBlock = document.getElementById("globe-mobile-block");
    if (mobileBlock) {
      mobileBlock.hidden = !visible;
    }
  }

  function setGlobeCanvasVisible(visible) {
    const container = document.getElementById("depthdif-cesium-globe");
    if (container) {
      container.hidden = !visible;
    }
  }

  function ensureCesiumStylesheet() {
    if (document.getElementById(CESIUM_CSS_LINK_ID)) {
      return;
    }
    const link = document.createElement("link");
    link.id = CESIUM_CSS_LINK_ID;
    link.rel = "stylesheet";
    link.href = CESIUM_CSS_URL;
    document.head.appendChild(link);
  }

  function ensureCesiumScript() {
    if (typeof window.Cesium !== "undefined") {
      return Promise.resolve();
    }
    if (window.__depthdifTemporalCesiumScriptPromise) {
      return window.__depthdifTemporalCesiumScriptPromise;
    }
    window.__depthdifTemporalCesiumScriptPromise = new Promise(function (resolve, reject) {
      const existingScript = document.getElementById(CESIUM_JS_SCRIPT_ID);
      if (existingScript) {
        existingScript.addEventListener("load", resolve, { once: true });
        existingScript.addEventListener("error", function () {
          reject(new Error("Failed to load Cesium script."));
        }, { once: true });
        return;
      }
      const script = document.createElement("script");
      script.id = CESIUM_JS_SCRIPT_ID;
      script.src = CESIUM_JS_URL;
      script.async = true;
      script.onload = resolve;
      script.onerror = function () {
        reject(new Error("Failed to load Cesium script."));
      };
      document.head.appendChild(script);
    }).catch(function (error) {
      window.__depthdifTemporalCesiumScriptPromise = null;
      throw error;
    });
    return window.__depthdifTemporalCesiumScriptPromise;
  }

  function ensureTemporalGlobeScript() {
    if (typeof window.initDepthDifTemporalGlobe === "function") {
      return Promise.resolve();
    }
    if (window.__depthdifTemporalGlobeScriptPromise) {
      return window.__depthdifTemporalGlobeScriptPromise;
    }
    window.__depthdifTemporalGlobeScriptPromise = new Promise(function (resolve, reject) {
      const existingScript = document.getElementById(CESIUM_APP_SCRIPT_ID);
      if (existingScript) {
        existingScript.addEventListener("load", resolve, { once: true });
        existingScript.addEventListener("error", function () {
          reject(new Error("Failed to load temporal globe script."));
        }, { once: true });
        return;
      }
      const script = document.createElement("script");
      script.id = CESIUM_APP_SCRIPT_ID;
      script.src = new URL(CESIUM_APP_SCRIPT_PATH, LOADER_SCRIPT_BASE_URL).toString();
      script.async = true;
      script.onload = resolve;
      script.onerror = function () {
        reject(new Error("Failed to load temporal globe script."));
      };
      document.head.appendChild(script);
    }).catch(function (error) {
      window.__depthdifTemporalGlobeScriptPromise = null;
      throw error;
    });
    return window.__depthdifTemporalGlobeScriptPromise;
  }

  function maybeInitTemporalGlobe() {
    const container = document.getElementById("depthdif-cesium-globe");
    if (!container) {
      if (typeof window.destroyDepthDifTemporalGlobe === "function") {
        window.destroyDepthDifTemporalGlobe();
      }
      return;
    }

    if (shouldBlockOnMobile()) {
      setMobileBlockVisible(true);
      setGlobeCanvasVisible(false);
      if (typeof window.destroyDepthDifTemporalGlobe === "function") {
        window.destroyDepthDifTemporalGlobe();
      }
      return;
    }

    setMobileBlockVisible(false);
    setGlobeCanvasVisible(true);
    ensureCesiumStylesheet();
    ensureCesiumScript()
      .then(ensureTemporalGlobeScript)
      .then(function () {
        if (typeof window.initDepthDifTemporalGlobe === "function") {
          window.initDepthDifTemporalGlobe();
        }
      })
      .catch(function (error) {
        console.error(error);
      });
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(maybeInitTemporalGlobe);
  } else {
    document.addEventListener("DOMContentLoaded", maybeInitTemporalGlobe);
  }
  window.addEventListener("resize", maybeInitTemporalGlobe);
})();
