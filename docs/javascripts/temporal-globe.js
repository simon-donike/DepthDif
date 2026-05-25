(function () {
  const DEFAULT_TEMPORAL_GLOBE_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/temporal-globe/temporal-globe-config.json";
  const DEFAULT_CAMERA_DESTINATION = {
    lon: -38.56452881619089,
    lat: 34.53988238358822,
    height: 9500000.0,
  };
  const SPIN_RATE_RADIANS_PER_SECOND = 5.0 * (Math.PI / 180.0);
  const TEMPERATURE_COLOR_STOPS = [
    { value: 0.0, rgb: [18, 38, 140] },
    { value: 4.0, rgb: [30, 86, 196] },
    { value: 8.0, rgb: [44, 140, 255] },
    { value: 12.0, rgb: [58, 212, 255] },
    { value: 16.0, rgb: [255, 238, 98] },
    { value: 20.0, rgb: [255, 172, 54] },
    { value: 24.0, rgb: [240, 84, 32] },
    { value: 30.0, rgb: [180, 16, 26] },
  ];
  const SALINITY_EXPORT_COLOR_STOPS = [
    { offset: 0.0, rgb: [49, 54, 149] },
    { offset: 0.2, rgb: [69, 117, 180] },
    { offset: 0.4, rgb: [116, 173, 209] },
    { offset: 0.6, rgb: [171, 221, 164] },
    { offset: 0.8, rgb: [253, 224, 71] },
    { offset: 1.0, rgb: [168, 85, 36] },
  ];

  function getTemporalGlobeElements() {
    const container = document.getElementById("depthdif-cesium-globe");
    if (!container) {
      return null;
    }
    return {
      container: container,
      stage: container.closest(".globe-stage"),
      toolbar: document.querySelector(".globe-toolbar"),
      toolbarContent: document.getElementById("temporal-globe-toolbar-content"),
      toolbarToggle: document.getElementById("temporal-globe-toggle-toolbar"),
      variableControl: document.getElementById("temporal-globe-variable-control"),
      variableRadios: document.querySelectorAll('input[name="temporal-globe-variable"]'),
      layerRadios: document.querySelectorAll('input[name="temporal-globe-layer"]'),
      predictionLayerToggle: document.getElementById("temporal-globe-layer-prediction"),
      errorLayerToggle: document.getElementById("temporal-globe-layer-error"),
      weekSlider: document.getElementById("temporal-globe-week-slider"),
      weekLabel: document.getElementById("temporal-globe-week-label"),
      playToggle: document.getElementById("temporal-globe-play-toggle"),
      speedSelect: document.getElementById("temporal-globe-speed"),
      spinToggle: document.getElementById("temporal-globe-toggle-spin"),
      resetButton: document.getElementById("temporal-globe-reset-camera"),
      pageEyebrow: document.getElementById("temporal-globe-page-eyebrow"),
      pageTitle: document.getElementById("temporal-globe-page-title"),
      pageDescription: document.getElementById("temporal-globe-page-description"),
      valueLegend: document.getElementById("temporal-globe-value-legend"),
      valueLegendTitle: document.getElementById("temporal-globe-value-legend-title"),
      valueLegendMin: document.getElementById("temporal-globe-value-legend-min"),
      valueLegendMax: document.getElementById("temporal-globe-value-legend-max"),
      valueLegendBar: document.getElementById("temporal-globe-value-legend-bar"),
      errorLegend: document.getElementById("temporal-globe-error-legend"),
      errorLegendMin: document.getElementById("temporal-globe-error-legend-min"),
      errorLegendMax: document.getElementById("temporal-globe-error-legend-max"),
    };
  }

  function resolveConfigUrl() {
    const params = new URLSearchParams(window.location.search);
    const configParam = params.get("config");
    if (configParam && configParam.trim() !== "") {
      return new URL(configParam, window.location.href).toString();
    }
    return DEFAULT_TEMPORAL_GLOBE_CONFIG_URL;
  }

  function resolveAssetUrl(value, configUrl) {
    if (!value) {
      return null;
    }
    return new URL(String(value), configUrl).toString();
  }

  function requestRender(state) {
    if (state && state.viewer && !state.viewer.isDestroyed()) {
      state.viewer.scene.requestRender();
    }
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(maxValue, Math.max(minValue, value));
  }

  function firstFiniteNumber(values, fallback) {
    for (let index = 0; index < values.length; index += 1) {
      const value = Number(values[index]);
      if (Number.isFinite(value)) {
        return value;
      }
    }
    return fallback;
  }

  function getVariableConfigs(config) {
    if (!config || !config.variables || typeof config.variables !== "object") {
      return {};
    }
    return config.variables;
  }

  function resolveDefaultVariable(config) {
    const variables = getVariableConfigs(config);
    const configuredDefault = String(config && config.default_variable ? config.default_variable : "");
    if (configuredDefault && variables[configuredDefault]) {
      return configuredDefault;
    }
    if (variables.temperature) {
      return "temperature";
    }
    return Object.keys(variables)[0] || "temperature";
  }

  function activeVariableConfig(state) {
    const variables = getVariableConfigs(state.config);
    return variables[state.selectedVariable] || variables[resolveDefaultVariable(state.config)] || {};
  }

  function activeFrames(state) {
    const config = activeVariableConfig(state);
    return Array.isArray(config.frames) ? config.frames : [];
  }

  function selectedFrame(state) {
    const frames = activeFrames(state);
    const index = clamp(Number(state.selectedFrameIndex || 0), 0, Math.max(0, frames.length - 1));
    return frames[index] || null;
  }

  function selectedFrameUrl(state) {
    const frame = selectedFrame(state);
    if (!frame) {
      return null;
    }
    if (state.selectedLayer === "absolute_error") {
      return frame.absolute_error_tiles_url;
    }
    return frame.prediction_tiles_url;
  }

  function valueUnitLabel(config) {
    const explicit = config && config.value_unit_label ? String(config.value_unit_label) : "";
    if (explicit) {
      return explicit;
    }
    const units = String(config && config.value_units ? config.value_units : "");
    return units.toUpperCase() === "PSU" ? "PSU" : "°C";
  }

  function formatLegendValue(value, unitLabel) {
    const numericValue = Number(value);
    const unit = String(unitLabel || "");
    if (!Number.isFinite(numericValue)) {
      return unit === "PSU" ? "0 PSU" : "0°C";
    }
    const rounded = Math.abs(numericValue - Math.round(numericValue)) < 0.05
      ? String(Math.round(numericValue))
      : String(Number(numericValue.toFixed(1)));
    return unit === "PSU" ? rounded + " PSU" : rounded + "°C";
  }

  function updatePageText(state) {
    const year = state.config.validation_year || "validation year";
    const depthSuffix = state.config.depth_suffix || "10m";
    if (state.elements.pageEyebrow) {
      state.elements.pageEyebrow.textContent = "DepthDif Temporal Viewer";
    }
    if (state.elements.pageTitle) {
      state.elements.pageTitle.textContent = "Temporal Globe";
    }
    if (state.elements.pageDescription) {
      state.elements.pageDescription.textContent =
        "Weekly " + depthSuffix + " reconstruction animation for " + String(year) + ".";
    }
  }

  function updateVariableControl(state) {
    const variables = getVariableConfigs(state.config);
    state.elements.variableRadios.forEach(function (radio) {
      const available = Boolean(variables[radio.value]);
      radio.disabled = !available;
      radio.checked = available && radio.value === state.selectedVariable;
      const label = radio.closest("label");
      if (label) {
        label.hidden = !available;
      }
    });
  }

  function updateLayerControl(state) {
    const frame = selectedFrame(state);
    const hasError = Boolean(frame && frame.absolute_error_tiles_url);
    if (state.elements.predictionLayerToggle) {
      state.elements.predictionLayerToggle.checked = state.selectedLayer === "prediction";
    }
    if (state.elements.errorLayerToggle) {
      state.elements.errorLayerToggle.disabled = !hasError;
      state.elements.errorLayerToggle.checked = state.selectedLayer === "absolute_error" && hasError;
    }
    if (!hasError && state.selectedLayer === "absolute_error") {
      state.selectedLayer = "prediction";
    }
  }

  function updateFrameControl(state) {
    const frames = activeFrames(state);
    const frame = selectedFrame(state);
    const maxIndex = Math.max(0, frames.length - 1);
    state.selectedFrameIndex = clamp(Number(state.selectedFrameIndex || 0), 0, maxIndex);
    if (state.elements.weekSlider) {
      state.elements.weekSlider.min = "0";
      state.elements.weekSlider.max = String(maxIndex);
      state.elements.weekSlider.value = String(state.selectedFrameIndex);
      state.elements.weekSlider.disabled = frames.length <= 1;
    }
    if (state.elements.weekLabel) {
      state.elements.weekLabel.textContent = frame ? String(frame.label) : "No frames";
    }
  }

  function updateLegends(state) {
    const activeConfig = activeVariableConfig(state);
    const isError = state.selectedLayer === "absolute_error";
    if (state.elements.valueLegend) {
      state.elements.valueLegend.hidden = isError;
    }
    if (state.elements.errorLegend) {
      state.elements.errorLegend.hidden = !isError;
    }
    if (!isError) {
      const unitLabel = valueUnitLabel(activeConfig);
      if (state.elements.valueLegendTitle) {
        state.elements.valueLegendTitle.textContent = String(activeConfig.variable_label || activeConfig.variable || "Temperature");
      }
      if (state.elements.valueLegendMin) {
        state.elements.valueLegendMin.textContent = formatLegendValue(activeConfig.color_scale_min, unitLabel);
      }
      if (state.elements.valueLegendMax) {
        state.elements.valueLegendMax.textContent = formatLegendValue(activeConfig.color_scale_max, unitLabel);
      }
      if (state.elements.valueLegendBar) {
        state.elements.valueLegendBar.classList.toggle("globe-legend__bar--salinity", state.selectedVariable === "salinity");
      }
      return;
    }

    const errorUnitLabel = activeConfig.absolute_error_value_unit_label || valueUnitLabel(activeConfig);
    if (state.elements.errorLegendMin) {
      state.elements.errorLegendMin.textContent = formatLegendValue(
        firstFiniteNumber([activeConfig.absolute_error_legend_min], 0.0),
        errorUnitLabel
      );
    }
    if (state.elements.errorLegendMax) {
      state.elements.errorLegendMax.textContent = formatLegendValue(
        firstFiniteNumber([activeConfig.absolute_error_legend_max, activeConfig.absolute_error_color_scale_max], 0.0),
        errorUnitLabel
      );
    }
  }

  function syncControls(state) {
    updatePageText(state);
    updateVariableControl(state);
    updateFrameControl(state);
    updateLayerControl(state);
    updateLegends(state);
  }

  function addBaseMap(viewer, config, configUrl) {
    const naturalEarthUrl = Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII");

    function addProvider(url, credit, fallback) {
      Cesium.TileMapServiceImageryProvider.fromUrl(url, { credit: credit })
        .then(function (provider) {
          if (!viewer.isDestroyed()) {
            const baseLayer = viewer.imageryLayers.addImageryProvider(provider);
            viewer.imageryLayers.lowerToBottom(baseLayer);
            viewer.scene.requestRender();
          }
        })
        .catch(function (error) {
          console.error(error);
          fallback();
        });
    }

    function addOpenStreetMapFallback() {
      if (viewer.isDestroyed()) {
        return;
      }
      const fallbackLayer = viewer.imageryLayers.addImageryProvider(
        new Cesium.OpenStreetMapImageryProvider({
          url: "https://tile.openstreetmap.org/",
          credit: "OpenStreetMap contributors",
        })
      );
      viewer.imageryLayers.lowerToBottom(fallbackLayer);
      viewer.scene.requestRender();
    }

    function addBundledNaturalEarthFallback() {
      addProvider(naturalEarthUrl, "Natural Earth II", addOpenStreetMapFallback);
    }

    const hostedBaseMapUrl = resolveAssetUrl(config && config.base_map_tiles_url, configUrl);
    if (hostedBaseMapUrl) {
      addProvider(hostedBaseMapUrl, config.base_map_credit || "Natural Earth II", addBundledNaturalEarthFallback);
      return;
    }
    addBundledNaturalEarthFallback();
  }

  function buildViewer(container, config, configUrl) {
    const viewer = new Cesium.Viewer(container, {
      animation: false,
      baseLayer: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      contextOptions: {
        webgl: {
          preserveDrawingBuffer: true,
        },
      },
      navigationHelpButton: false,
      requestRenderMode: true,
      maximumRenderTimeChange: Number.POSITIVE_INFINITY,
      sceneModePicker: false,
      selectionIndicator: false,
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      timeline: false,
    });
    viewer.useBrowserRecommendedResolution = false;
    viewer.resolutionScale = window.devicePixelRatio || 1;
    addBaseMap(viewer, config, configUrl);
    viewer.scene.globe.enableLighting = false;
    viewer.clock.shouldAnimate = false;
    return viewer;
  }

  function resolveCameraDestination(config) {
    const destination = config.default_camera_destination || {};
    if (
      Number.isFinite(destination.lon) &&
      Number.isFinite(destination.lat) &&
      Number.isFinite(destination.height)
    ) {
      return destination;
    }
    return DEFAULT_CAMERA_DESTINATION;
  }

  function flyToConfig(state) {
    const destination = resolveCameraDestination(state.config);
    state.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(destination.lon, destination.lat, destination.height),
      orientation: {
        heading: 0.0,
        pitch: -Cesium.Math.PI_OVER_TWO,
        roll: 0.0,
      },
      duration: 1.8,
      complete: function () {
        requestRender(state);
      },
      cancel: function () {
        requestRender(state);
      },
    });
    requestRender(state);
  }

  function forceViewerResize(state) {
    if (!state.viewer || state.viewer.isDestroyed()) {
      return;
    }
    state.viewer.resolutionScale = window.devicePixelRatio || 1;
    state.viewer.resize();
    window.requestAnimationFrame(function () {
      if (!state.viewer || state.viewer.isDestroyed()) {
        return;
      }
      state.viewer.resize();
      requestRender(state);
    });
  }

  function watchContainerResize(state, element) {
    if (typeof window.ResizeObserver === "undefined") {
      return function () {};
    }
    const observer = new window.ResizeObserver(function () {
      window.requestAnimationFrame(function () {
        forceViewerResize(state);
      });
    });
    observer.observe(element);
    return function () {
      observer.disconnect();
    };
  }

  function setToolbarCollapsed(elements, collapsed) {
    if (!elements.toolbar || !elements.toolbarToggle || !elements.toolbarContent) {
      return;
    }
    elements.toolbar.classList.toggle("is-collapsed", collapsed);
    elements.toolbarToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    elements.toolbarToggle.textContent = collapsed ? "Show settings" : "Hide settings";
  }

  function layerCacheKey(state, frameIndex) {
    return [state.selectedVariable, state.selectedLayer, String(frameIndex)].join(":");
  }

  function frameUrlForIndex(state, frameIndex) {
    const frames = activeFrames(state);
    const frame = frames[frameIndex];
    if (!frame) {
      return null;
    }
    return state.selectedLayer === "absolute_error" ? frame.absolute_error_tiles_url : frame.prediction_tiles_url;
  }

  function creditForLayer(state) {
    const credits = state.config.credits || {};
    return state.selectedLayer === "absolute_error" ? credits.absolute_error : credits.prediction;
  }

  function pruneLayerCache(state) {
    const frames = activeFrames(state);
    const nextIndex = frames.length === 0 ? 0 : (state.selectedFrameIndex + 1) % frames.length;
    const allowed = new Set([state.currentLayerKey, layerCacheKey(state, nextIndex)]);
    Array.from(state.layerCache.keys()).forEach(function (key) {
      if (allowed.has(key)) {
        return;
      }
      const entry = state.layerCache.get(key);
      state.layerCache.delete(key);
      if (entry && entry.layer && !state.viewer.isDestroyed()) {
        state.viewer.imageryLayers.remove(entry.layer, true);
      }
    });
  }

  function ensureFrameLayer(state, frameIndex) {
    const key = layerCacheKey(state, frameIndex);
    const cached = state.layerCache.get(key);
    if (cached) {
      return cached.promise || Promise.resolve(cached.layer);
    }

    const frameUrl = resolveAssetUrl(frameUrlForIndex(state, frameIndex), state.configUrl);
    if (!frameUrl) {
      return Promise.resolve(null);
    }

    const entry = { layer: null, promise: null };
    state.layerCache.set(key, entry);
    entry.promise = Cesium.TileMapServiceImageryProvider.fromUrl(frameUrl, {
      credit: creditForLayer(state),
    })
      .then(function (provider) {
        if (!state.viewer || state.viewer.isDestroyed()) {
          return null;
        }
        const layer = state.viewer.imageryLayers.addImageryProvider(provider);
        layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
        layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
        layer.alpha = 1.0;
        layer.show = false;
        entry.layer = layer;
        requestRender(state);
        return layer;
      })
      .catch(function (error) {
        state.layerCache.delete(key);
        console.error(error);
        return null;
      })
      .finally(function () {
        entry.promise = null;
      });
    return entry.promise;
  }

  function preloadNextFrame(state) {
    const frames = activeFrames(state);
    if (frames.length <= 1) {
      return;
    }
    const nextIndex = (state.selectedFrameIndex + 1) % frames.length;
    ensureFrameLayer(state, nextIndex).then(function () {
      pruneLayerCache(state);
    });
  }

  function showSelectedFrame(state) {
    const frameIndex = state.selectedFrameIndex;
    const loadToken = state.frameLoadToken + 1;
    state.frameLoadToken = loadToken;
    syncControls(state);
    return ensureFrameLayer(state, frameIndex).then(function (layer) {
      if (loadToken !== state.frameLoadToken) {
        if (layer) {
          layer.show = false;
        }
        requestRender(state);
        return null;
      }
      if (state.currentLayer && state.currentLayer !== layer) {
        state.currentLayer.show = false;
      }
      state.currentLayer = layer;
      state.currentLayerKey = layerCacheKey(state, frameIndex);
      if (layer) {
        layer.show = true;
      }
      pruneLayerCache(state);
      preloadNextFrame(state);
      requestRender(state);
      return layer;
    });
  }

  function setFrameIndex(state, frameIndex) {
    const frames = activeFrames(state);
    state.selectedFrameIndex = clamp(Number(frameIndex), 0, Math.max(0, frames.length - 1));
    showSelectedFrame(state);
  }

  function playbackDelayMs(state) {
    const baseInterval = firstFiniteNumber([state.config.frame_interval_ms], 1000);
    const speed = firstFiniteNumber([state.elements.speedSelect && state.elements.speedSelect.value], 1.0);
    return Math.max(100, Math.round(baseInterval / Math.max(0.1, speed)));
  }

  function stopPlayback(state) {
    if (state.playbackTimer !== null) {
      window.clearInterval(state.playbackTimer);
      state.playbackTimer = null;
    }
    state.playing = false;
    if (state.elements.playToggle) {
      state.elements.playToggle.setAttribute("aria-pressed", "false");
      state.elements.playToggle.textContent = "Play";
    }
  }

  function startPlayback(state) {
    stopPlayback(state);
    state.playing = true;
    if (state.elements.playToggle) {
      state.elements.playToggle.setAttribute("aria-pressed", "true");
      state.elements.playToggle.textContent = "Pause";
    }
    state.playbackTimer = window.setInterval(function () {
      const frames = activeFrames(state);
      if (frames.length <= 1) {
        return;
      }
      setFrameIndex(state, (state.selectedFrameIndex + 1) % frames.length);
    }, playbackDelayMs(state));
  }

  function setSpinEnabled(state, enabled) {
    state.spinEnabled = Boolean(enabled);
    state.lastSpinTime = null;
    state.viewer.clock.shouldAnimate = state.spinEnabled;
    if (state.elements.spinToggle) {
      state.elements.spinToggle.setAttribute("aria-pressed", state.spinEnabled ? "true" : "false");
      state.elements.spinToggle.textContent = state.spinEnabled ? "Stop Spin" : "Spin Globe";
    }
    requestRender(state);
  }

  function attachSpinLoop(state) {
    state.spinTickListener = function (clock) {
      if (!state.spinEnabled) {
        state.lastSpinTime = null;
        return;
      }
      if (state.lastSpinTime === null) {
        state.lastSpinTime = Cesium.JulianDate.clone(clock.currentTime);
        requestRender(state);
        return;
      }
      const deltaSeconds = Cesium.JulianDate.secondsDifference(clock.currentTime, state.lastSpinTime);
      state.lastSpinTime = Cesium.JulianDate.clone(clock.currentTime);
      if (deltaSeconds <= 0.0) {
        requestRender(state);
        return;
      }
      state.viewer.scene.camera.rotate(
        Cesium.Cartesian3.UNIT_Z,
        -SPIN_RATE_RADIANS_PER_SECOND * deltaSeconds
      );
      requestRender(state);
    };
    state.viewer.clock.onTick.addEventListener(state.spinTickListener);
  }

  function clearLayerCache(state) {
    if (!state || !state.viewer || state.viewer.isDestroyed()) {
      return;
    }
    state.layerCache.forEach(function (entry) {
      if (entry.layer) {
        state.viewer.imageryLayers.remove(entry.layer, true);
      }
    });
    state.layerCache.clear();
    state.currentLayer = null;
    state.currentLayerKey = null;
  }

  function attachControls(state) {
    if (state.elements.toolbarToggle) {
      state.elements.toolbarToggle.addEventListener("click", function () {
        const collapsed = !state.elements.toolbar.classList.contains("is-collapsed");
        setToolbarCollapsed(state.elements, collapsed);
        requestRender(state);
      });
    }

    state.elements.variableRadios.forEach(function (radio) {
      radio.addEventListener("change", function () {
        if (!radio.checked || radio.disabled) {
          return;
        }
        stopPlayback(state);
        state.selectedVariable = radio.value;
        state.selectedFrameIndex = 0;
        clearLayerCache(state);
        setFrameIndex(state, 0);
      });
    });

    state.elements.layerRadios.forEach(function (radio) {
      radio.addEventListener("change", function () {
        if (!radio.checked || radio.disabled) {
          return;
        }
        state.selectedLayer = radio.value;
        clearLayerCache(state);
        showSelectedFrame(state);
      });
    });

    if (state.elements.weekSlider) {
      state.elements.weekSlider.addEventListener("input", function () {
        setFrameIndex(state, Number(state.elements.weekSlider.value));
      });
    }

    if (state.elements.playToggle) {
      state.elements.playToggle.addEventListener("click", function () {
        if (state.playing) {
          stopPlayback(state);
        } else {
          startPlayback(state);
        }
      });
    }

    if (state.elements.speedSelect) {
      state.elements.speedSelect.addEventListener("change", function () {
        if (state.playing) {
          startPlayback(state);
        }
      });
    }

    if (state.elements.spinToggle) {
      state.elements.spinToggle.addEventListener("click", function () {
        setSpinEnabled(state, !state.spinEnabled);
      });
    }

    if (state.elements.resetButton) {
      state.elements.resetButton.addEventListener("click", function () {
        flyToConfig(state);
      });
    }
  }

  function destroyState(state) {
    if (!state) {
      return;
    }
    stopPlayback(state);
    if (state.resizeCleanup) {
      state.resizeCleanup();
    }
    if (state.viewer && !state.viewer.isDestroyed()) {
      if (state.spinTickListener) {
        state.viewer.clock.onTick.removeEventListener(state.spinTickListener);
      }
      state.viewer.destroy();
    }
  }

  async function initDepthDifTemporalGlobe() {
    const elements = getTemporalGlobeElements();
    if (!elements) {
      return;
    }
    if (window.__depthdifTemporalGlobeState) {
      destroyState(window.__depthdifTemporalGlobeState);
      window.__depthdifTemporalGlobeState = null;
    }

    const configUrl = resolveConfigUrl();
    const response = await fetch(configUrl, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error("Failed to load temporal globe config: " + response.status);
    }
    const config = await response.json();
    const defaultVariable = resolveDefaultVariable(config);
    const state = {
      elements: elements,
      config: config,
      configUrl: configUrl,
      selectedVariable: defaultVariable,
      selectedLayer: config.default_layer === "absolute_error" ? "absolute_error" : "prediction",
      selectedFrameIndex: 0,
      viewer: buildViewer(elements.container, config, configUrl),
      layerCache: new Map(),
      currentLayer: null,
      currentLayerKey: null,
      frameLoadToken: 0,
      playbackTimer: null,
      playing: false,
      spinEnabled: false,
      lastSpinTime: null,
      spinTickListener: null,
      resizeCleanup: null,
    };
    window.__depthdifTemporalGlobeState = state;
    attachSpinLoop(state);
    attachControls(state);
    state.resizeCleanup = watchContainerResize(state, elements.container);
    syncControls(state);
    forceViewerResize(state);
    flyToConfig(state);
    await showSelectedFrame(state);
  }

  function destroyDepthDifTemporalGlobe() {
    destroyState(window.__depthdifTemporalGlobeState);
    window.__depthdifTemporalGlobeState = null;
  }

  window.initDepthDifTemporalGlobe = initDepthDifTemporalGlobe;
  window.destroyDepthDifTemporalGlobe = destroyDepthDifTemporalGlobe;
})();
