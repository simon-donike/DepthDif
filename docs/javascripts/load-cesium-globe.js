(function () {
  const CESIUM_JS_URL =
    "https://cesium.com/downloads/cesiumjs/releases/1.140/Build/Cesium/Cesium.js";
  const CESIUM_CSS_URL =
    "https://cesium.com/downloads/cesiumjs/releases/1.140/Build/Cesium/Widgets/widgets.css";
  const CESIUM_APP_SCRIPT_PATH = "javascripts/cesium-globe.js";
  const CESIUM_JS_SCRIPT_ID = "depthdif-cesium-js";
  const CESIUM_CSS_LINK_ID = "depthdif-cesium-css";
  const CESIUM_APP_SCRIPT_ID = "depthdif-cesium-app";
  const MOBILE_BLOCK_MEDIA_QUERY = "(max-width: 900px), (pointer: coarse) and (max-width: 1024px)";

  function isLikelyIPad() {
    const navigatorObject = window.navigator;
    if (!navigatorObject) {
      return false;
    }

    const userAgent = String(navigatorObject.userAgent || "");
    const platform = String(navigatorObject.platform || "");
    const maxTouchPoints = Number(navigatorObject.maxTouchPoints || 0);

    // iPadOS can report a desktop-class Mac platform, so combine that with touch
    // support to keep the "desktop only" globe reliably blocked on iPads.
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
    if (!mobileBlock) {
      return;
    }
    mobileBlock.hidden = !visible;
  }

  function setGlobeCanvasVisible(visible) {
    const container = document.getElementById("depthdif-cesium-globe");
    if (!container) {
      return;
    }
    container.hidden = !visible;
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
    if (window.__depthdifCesiumScriptPromise) {
      return window.__depthdifCesiumScriptPromise;
    }

    window.__depthdifCesiumScriptPromise = new Promise(function (resolve, reject) {
      const existingScript = document.getElementById(CESIUM_JS_SCRIPT_ID);
      if (existingScript) {
        existingScript.addEventListener("load", resolve, { once: true });
        existingScript.addEventListener(
          "error",
          function () {
            reject(new Error("Failed to load Cesium script."));
          },
          { once: true }
        );
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
      window.__depthdifCesiumScriptPromise = null;
      throw error;
    });

    return window.__depthdifCesiumScriptPromise;
  }

  function ensureCesiumAppScript() {
    if (typeof window.initDepthDifCesiumGlobe === "function") {
      return Promise.resolve();
    }
    if (window.__depthdifCesiumAppScriptPromise) {
      return window.__depthdifCesiumAppScriptPromise;
    }

    window.__depthdifCesiumAppScriptPromise = new Promise(function (resolve, reject) {
      const existingScript = document.getElementById(CESIUM_APP_SCRIPT_ID);
      if (existingScript) {
        existingScript.addEventListener("load", resolve, { once: true });
        existingScript.addEventListener(
          "error",
          function () {
            reject(new Error("Failed to load Cesium app script."));
          },
          { once: true }
        );
        return;
      }

      const script = document.createElement("script");
      script.id = CESIUM_APP_SCRIPT_ID;
      script.src = new URL(CESIUM_APP_SCRIPT_PATH, document.baseURI).toString();
      script.async = true;
      script.onload = resolve;
      script.onerror = function () {
        reject(new Error("Failed to load Cesium app script."));
      };
      document.head.appendChild(script);
    }).catch(function (error) {
      window.__depthdifCesiumAppScriptPromise = null;
      throw error;
    });

    return window.__depthdifCesiumAppScriptPromise;
  }

  function maybeInitCesiumGlobe() {
    const container = document.getElementById("depthdif-cesium-globe");
    if (!container) {
      if (typeof window.destroyDepthDifCesiumGlobe === "function") {
        window.destroyDepthDifCesiumGlobe();
      }
      return;
    }

    if (shouldBlockOnMobile()) {
      setMobileBlockVisible(true);
      setGlobeCanvasVisible(false);
      if (typeof window.destroyDepthDifCesiumGlobe === "function") {
        window.destroyDepthDifCesiumGlobe();
      }
      return;
    }

    setMobileBlockVisible(false);
    setGlobeCanvasVisible(true);

    ensureCesiumStylesheet();
    ensureCesiumScript()
      .then(function () {
        return ensureCesiumAppScript();
      })
      .then(function () {
        if (typeof window.initDepthDifCesiumGlobe === "function") {
          window.initDepthDifCesiumGlobe();
        }
      })
      .catch(function (error) {
        console.error(error);
      });
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(maybeInitCesiumGlobe);
  } else {
    document.addEventListener("DOMContentLoaded", maybeInitCesiumGlobe);
  }

  window.addEventListener("resize", maybeInitCesiumGlobe);
})();
