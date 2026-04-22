(function () {
  const CESIUM_JS_URL =
    "https://cesium.com/downloads/cesiumjs/releases/1.140/Build/Cesium/Cesium.js";
  const CESIUM_CSS_URL =
    "https://cesium.com/downloads/cesiumjs/releases/1.140/Build/Cesium/Widgets/widgets.css";
  const CESIUM_JS_SCRIPT_ID = "depthdif-cesium-js";
  const CESIUM_CSS_LINK_ID = "depthdif-cesium-css";
  const MOBILE_BLOCK_MEDIA_QUERY = "(max-width: 900px), (pointer: coarse) and (max-width: 1024px)";

  function shouldBlockOnMobile() {
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
      if (typeof window.destroyDepthDifCesiumGlobe === "function") {
        window.destroyDepthDifCesiumGlobe();
      }
      return;
    }

    setMobileBlockVisible(false);

    ensureCesiumStylesheet();
    ensureCesiumScript()
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
