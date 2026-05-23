(function () {
  const DEFAULT_GLOBE_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/globe/globe-config.json";
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
  const ERROR_EXPORT_COLOR_STOPS = [
    { offset: 0.0, rgb: [34, 197, 94] },
    { offset: 0.5, rgb: [250, 204, 21] },
    { offset: 1.0, rgb: [220, 38, 38] },
  ];
  const DEFAULT_COLOR_SCALE = { min: 0.0, max: 30.0 };
  const PATCH_FILL_ALPHA = 0.18;
  const PATCH_FILL_COLOR = "#f97316";
  const PATCH_OUTLINE_COLOR = "#fb923c";
  const PATCH_OUTLINE_WIDTH = 2.75;
  const PROFILE_POPUP_CLOSE_DELAY_MS = 180;
  const BACKGROUND_PRELOAD_DELAY_MS = 180;
  const MONTH_ABBREVIATIONS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  const ARGO_POINT_MARKER_IMAGE = buildMarkerImage(
    [
      '<circle cx="32" cy="32" r="15" fill="rgba(6, 23, 38, 0.82)" stroke="#7cf5ff" stroke-width="3"/>',
      '<circle cx="32" cy="32" r="6" fill="#dffcff"/>',
      '<path d="M32 9 L39 16 L32 23 L25 16 Z" fill="#7cf5ff" fill-opacity="0.9"/>',
    ].join(""),
    64
  );
  const FULL_SAMPLE_MARKER_IMAGE = buildMarkerImage(
    [
      '<path d="M32 10 L48 22 L42 46 L32 54 L22 46 L16 22 Z" fill="rgba(9, 34, 56, 0.9)" stroke="#ffd166" stroke-width="3" />',
      '<circle cx="32" cy="30" r="8" fill="#fff4cf" stroke="#ffd166" stroke-width="2" />',
      '<path d="M32 54 L27 43 L37 43 Z" fill="#ffd166" fill-opacity="0.95" />',
    ].join(""),
    64
  );

  function getGlobeElements() {
    const container = document.getElementById("depthdif-cesium-globe");
    if (!container) {
      return null;
    }
    const stage = container.closest(".globe-stage");

    return {
      container: container,
      stage: stage,
      predictionToggle: document.getElementById("globe-toggle-prediction"),
      groundTruthToggle: document.getElementById("globe-toggle-ground-truth"),
      absoluteErrorToggle: document.getElementById("globe-toggle-absolute-error"),
      uncertaintyToggle: document.getElementById("globe-toggle-uncertainty"),
      pointsToggle: document.getElementById("globe-toggle-points"),
      pointsRadios: document.querySelectorAll('input[name="globe-points-layer"]'),
      variableControl: document.getElementById("globe-variable-control"),
      variableRadios: document.querySelectorAll('input[name="globe-variable"]'),
      patchSplitsToggle: document.getElementById("globe-toggle-patch-splits"),
      patchSplitsRadios: document.querySelectorAll('input[name="globe-patch-splits-layer"]'),
      toolbar: document.querySelector(".globe-toolbar"),
      toolbarContent: document.getElementById("globe-toolbar-content"),
      toolbarToggle: document.getElementById("globe-toggle-toolbar"),
      depthSlider: document.getElementById("globe-depth-level"),
      depthLabel: document.getElementById("globe-depth-level-label"),
      depthTicks: document.getElementById("globe-depth-level-ticks"),
      spinToggle: document.getElementById("globe-toggle-spin"),
      resetButton: document.getElementById("globe-reset-camera"),
      pictureExportButton: document.getElementById("globe-export-picture"),
      cinematicToggle: document.getElementById("globe-toggle-cinematic"),
      cinematicExit: document.getElementById("globe-exit-cinematic"),
      pageEyebrow: document.getElementById("globe-page-eyebrow"),
      pageTitle: document.getElementById("globe-page-title"),
      pageDescription: document.getElementById("globe-page-description"),
      mobileBlockTitle: document.getElementById("globe-mobile-block-title"),
      mobileBlockText: document.getElementById("globe-mobile-block-text"),
      argoLegend: document.getElementById("globe-argo-legend"),
      valueLegend: document.getElementById("globe-value-legend"),
      valueLegendTitle: document.getElementById("globe-value-legend-title"),
      valueLegendMin: document.getElementById("globe-value-legend-min"),
      valueLegendMax: document.getElementById("globe-value-legend-max"),
      valueLegendBar: document.getElementById("globe-value-legend-bar"),
      errorLegend: document.getElementById("globe-error-legend"),
      errorLegendTitle: document.getElementById("globe-error-legend-title"),
      errorLegendMin: document.getElementById("globe-error-legend-min"),
      errorLegendMax: document.getElementById("globe-error-legend-max"),
      profilePopup: document.getElementById("globe-profile-popup"),
      profilePopupTitle: document.getElementById("globe-profile-popup-title"),
      profilePopupSubtitle: document.getElementById("globe-profile-popup-subtitle"),
      profilePopupImage: document.getElementById("globe-profile-popup-image"),
      profilePopupClose: document.getElementById("globe-profile-popup-close"),
    };
  }

  function nextInitToken() {
    const nextToken = Number(window.__depthdifCesiumGlobeInitToken || 0) + 1;
    window.__depthdifCesiumGlobeInitToken = nextToken;
    return nextToken;
  }

  function getCurrentInitToken() {
    return Number(window.__depthdifCesiumGlobeInitToken || 0);
  }

  function requestRender(state) {
    if (!state || !state.viewer || state.viewer.isDestroyed()) {
      return;
    }
    state.viewer.scene.requestRender();
  }

  function scheduleBackgroundTask(callback) {
    if (typeof window.requestIdleCallback === "function") {
      return window.requestIdleCallback(callback, { timeout: 1200 });
    }
    return window.setTimeout(callback, BACKGROUND_PRELOAD_DELAY_MS);
  }

  function cancelBackgroundTask(taskId) {
    if (taskId === null || typeof taskId === "undefined") {
      return;
    }
    if (typeof window.cancelIdleCallback === "function") {
      window.cancelIdleCallback(taskId);
      return;
    }
    window.clearTimeout(taskId);
  }


  function getVariableConfigs(config) {
    if (!config || !config.variables || typeof config.variables !== "object") {
      return null;
    }
    return config.variables;
  }

  function resolveDefaultVariable(config) {
    const variables = getVariableConfigs(config);
    if (!variables) {
      return String(config && config.variable ? config.variable : "temperature");
    }
    const configuredDefault = String(config.default_variable || "").trim();
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
    if (!variables) {
      return state.config;
    }
    const selected = String(state.selectedVariable || resolveDefaultVariable(state.config));
    return variables[selected] || variables[resolveDefaultVariable(state.config)] || state.config;
  }

  function activeVariableKey(state) {
    const activeConfig = activeVariableConfig(state);
    return String(activeConfig.variable || state.selectedVariable || "temperature");
  }

  function valueUnitLabel(config) {
    const explicit = config && config.value_unit_label ? String(config.value_unit_label) : "";
    if (explicit) {
      return explicit;
    }
    const units = String(config && config.value_units ? config.value_units : "");
    return units.toUpperCase() === "PSU" ? "PSU" : "°C";
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

  function resolveConfigUrl() {
    const params = new URLSearchParams(window.location.search);
    const configParam = params.get("config");
    if (configParam && configParam.trim() !== "") {
      return new URL(configParam, window.location.href).toString();
    }
    return DEFAULT_GLOBE_CONFIG_URL;
  }

  function parseCompactUtcDate(selectedDate) {
    const raw = String(selectedDate || "").trim();
    if (!/^\d{8}$/.test(raw)) {
      return null;
    }

    const year = Number(raw.slice(0, 4));
    const monthIndex = Number(raw.slice(4, 6)) - 1;
    const day = Number(raw.slice(6, 8));
    const date = new Date(Date.UTC(year, monthIndex, day));
    if (
      Number.isNaN(date.getTime()) ||
      date.getUTCFullYear() !== year ||
      date.getUTCMonth() !== monthIndex ||
      date.getUTCDate() !== day
    ) {
      return null;
    }
    return date;
  }

  function dominantIsoWeekMonthLabel(date) {
    const weekStart = new Date(date.getTime());
    const isoDay = weekStart.getUTCDay() || 7;
    weekStart.setUTCDate(weekStart.getUTCDate() - isoDay + 1);

    const counts = new Map();
    for (let offset = 0; offset < 7; offset += 1) {
      const weekDate = new Date(weekStart.getTime());
      weekDate.setUTCDate(weekStart.getUTCDate() + offset);
      const monthIndex = weekDate.getUTCMonth();
      counts.set(monthIndex, (counts.get(monthIndex) || 0) + 1);
    }

    let dominantMonthIndex = date.getUTCMonth();
    let dominantCount = -1;
    counts.forEach(function (count, monthIndex) {
      if (count > dominantCount) {
        dominantMonthIndex = monthIndex;
        dominantCount = count;
      }
    });
    return MONTH_ABBREVIATIONS[dominantMonthIndex] || "";
  }

  function resolveSelectedDateParts(selectedDate) {
    const date = parseCompactUtcDate(selectedDate);
    if (!date) {
      return null;
    }
    const isoDate = new Date(date.getTime());
    const dayOfWeek = isoDate.getUTCDay() || 7;
    isoDate.setUTCDate(isoDate.getUTCDate() + 4 - dayOfWeek);
    const isoYear = isoDate.getUTCFullYear();
    const yearStart = new Date(Date.UTC(isoYear, 0, 1));
    const isoWeek = Math.ceil((((isoDate - yearStart) / 86400000) + 1) / 7);
    const dominantMonthLabel = dominantIsoWeekMonthLabel(date);
    return { isoYear, isoWeek, dominantMonthLabel };
  }

  function resolveConfigDateParts(config) {
    const targetDate = config.target_date || config.selected_date;
    const dateParts = resolveSelectedDateParts(targetDate);
    const configIsoYear = Number(config.iso_year);
    const configIsoWeek = Number(config.iso_week);
    if (!dateParts && (!Number.isFinite(configIsoYear) || !Number.isFinite(configIsoWeek))) {
      return null;
    }
    const isoYear = Number.isFinite(configIsoYear) ? configIsoYear : dateParts.isoYear;
    const isoWeek = Number.isFinite(configIsoWeek) ? configIsoWeek : dateParts.isoWeek;
    return {
      isoYear: Number.isFinite(isoYear) ? isoYear : dateParts.isoYear,
      isoWeek: Number.isFinite(isoWeek) ? isoWeek : dateParts.isoWeek,
      dominantMonthLabel: dateParts ? dateParts.dominantMonthLabel : "",
      dateLabel: formatCompactDate(targetDate),
    };
  }

  function formatCompactDate(dateValue) {
    const date = parseCompactUtcDate(dateValue);
    if (!date) {
      return "";
    }
    return (
      String(date.getUTCFullYear()) +
      "-" +
      String(date.getUTCMonth() + 1).padStart(2, "0") +
      "-" +
      String(date.getUTCDate()).padStart(2, "0")
    );
  }

  function formatCompactIsoWeek(dateValue) {
    const dateParts = resolveSelectedDateParts(dateValue);
    if (!dateParts) {
      return "Unknown week";
    }
    return (
      String(dateParts.isoYear) +
      "-W" +
      String(dateParts.isoWeek).padStart(2, "0") +
      (dateParts.dominantMonthLabel ? " (" + dateParts.dominantMonthLabel + ")" : "")
    );
  }

  function formatConfigIsoWeekLabel(dateParts) {
    return (
      String(dateParts.isoYear) +
      "-W" +
      String(dateParts.isoWeek).padStart(2, "0") +
      (dateParts.dominantMonthLabel ? " (" + dateParts.dominantMonthLabel + ")" : "")
    );
  }

  function updatePageHeader(elements, config) {
    if (
      !elements.pageEyebrow &&
      !elements.pageTitle &&
      !elements.pageDescription &&
      !elements.mobileBlockTitle &&
      !elements.mobileBlockText
    ) {
      return;
    }

    const selectedDateParts = resolveConfigDateParts(config);
    const weekLabel = selectedDateParts ? formatConfigIsoWeekLabel(selectedDateParts) : "";
    const compactWeekLabel = selectedDateParts
      ? String(selectedDateParts.isoYear) +
        "-W" +
        String(selectedDateParts.isoWeek).padStart(2, "0")
      : "";
    const titleText = "Ocean Variable Reconstruction";
    const descriptionText = selectedDateParts
      ? "Densifying ocean variables based on sparse ARGO submarine measurements.\nShowing ISO week " +
        weekLabel +
        "."
      : "Densifying ocean variables based on sparse ARGO submarine measurements.";

    document.title = compactWeekLabel ? titleText + " " + compactWeekLabel : titleText;
    const metaDescription = document.querySelector('meta[name="description"]');
    if (metaDescription) {
      metaDescription.setAttribute("content", descriptionText);
    }
    if (elements.pageEyebrow) {
      elements.pageEyebrow.textContent = compactWeekLabel
        ? "DepthDif Viewer " + compactWeekLabel
        : "DepthDif Viewer";
    }
    if (elements.pageTitle) {
      elements.pageTitle.textContent = titleText;
    }

    if (elements.pageDescription) {
      elements.pageDescription.textContent = descriptionText;
    }
    if (elements.mobileBlockTitle) {
      elements.mobileBlockTitle.textContent = compactWeekLabel
        ? "3D globe visualization for " + compactWeekLabel + " is disabled on mobile"
        : "3D globe visualization is disabled on mobile";
    }
    if (elements.mobileBlockText) {
      elements.mobileBlockText.textContent = selectedDateParts
        ? "Open the page on a desktop or laptop to load Cesium and inspect the " +
          weekLabel +
          " prediction globe."
        : "Open the page on a desktop or laptop to load Cesium and inspect the prediction globe.";
    }
  }

  async function loadConfig() {
    const configUrl = resolveConfigUrl();
    const response = await fetch(configUrl);
    if (!response.ok) {
      throw new Error("Failed to load globe config: " + response.status + " " + response.statusText);
    }
    const config = await response.json();
    return { config, configUrl };
  }

  function resolveAssetUrl(assetUrl, configUrl) {
    if (!assetUrl) {
      return null;
    }
    return new URL(assetUrl, configUrl).toString();
  }

  function selectOffRadioForToggle(toggle) {
    if (!toggle || toggle.type !== "radio" || !toggle.name) {
      return;
    }
    document.querySelectorAll('input[type="radio"]').forEach(function (radio) {
      if (radio.name === toggle.name && radio.value === "off") {
        radio.checked = true;
      }
    });
  }

  function markToggleUnavailable(toggle) {
    if (!toggle) {
      return;
    }
    toggle.checked = false;
    selectOffRadioForToggle(toggle);
    toggle.disabled = true;
    toggle.dataset.globeUnavailable = "true";
  }

  function clearToggleUnavailable(toggle) {
    if (!toggle) {
      return;
    }
    if (toggle.dataset.globeUnavailable === "true") {
      delete toggle.dataset.globeUnavailable;
      toggle.disabled = false;
    }
  }

  function setToggleAvailable(toggle, available) {
    if (available) {
      clearToggleUnavailable(toggle);
      return;
    }
    markToggleUnavailable(toggle);
  }

  function setToggleLabelHidden(toggle, hidden) {
    if (!toggle) {
      return;
    }
    const label = toggle.closest("label");
    if (label) {
      label.hidden = Boolean(hidden);
    }
  }

  function setToggleLoading(toggle, loading) {
    if (!toggle || toggle.dataset.globeUnavailable === "true") {
      return;
    }
    toggle.disabled = loading;
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(maxValue, Math.max(minValue, value));
  }

  function lerp(start, end, t) {
    return start + (end - start) * t;
  }

  function resolveColorScale(config) {
    const minValue = firstFiniteNumber([config.color_scale_min, config.color_scale_min_c], NaN);
    const maxValue = firstFiniteNumber([config.color_scale_max, config.color_scale_max_c], NaN);
    if (Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {
      return { min: minValue, max: maxValue };
    }
    return DEFAULT_COLOR_SCALE;
  }

  function getDepthLevels(config) {
    if (config && Array.isArray(config.depth_levels) && config.depth_levels.length > 0) {
      return config.depth_levels;
    }
    return [
      {
        label: "Surface",
        requested_depth_m: 0.0,
        actual_depth_m: 0.0,
        channel_index: 0,
        prediction_tiles_url: config ? config.prediction_tiles_url : null,
        ground_truth_tiles_url: config ? config.ground_truth_tiles_url : null,
        absolute_error_tiles_url: config ? config.absolute_error_tiles_url : null,
      },
    ];
  }

  function selectedDepthLevel(state) {
    const depthLevels = getDepthLevels(activeVariableConfig(state));
    const index = clamp(Number(state.selectedDepthIndex || 0), 0, depthLevels.length - 1);
    return depthLevels[index] || depthLevels[0];
  }

  function formatDepthMeters(depthLevel) {
    const requestedDepthM = Number(depthLevel.requested_depth_m);
    const actualDepthM = Number(depthLevel.actual_depth_m);
    const depthM = Number.isFinite(requestedDepthM) ? requestedDepthM : actualDepthM;
    if (!Number.isFinite(depthM)) {
      return String(depthLevel.label || "");
    }
    if (Math.abs(depthM) < 0.05) {
      return "0 m";
    }
    if (Math.abs(depthM) >= 1000.0) {
      const depthKm = depthM / 1000.0;
      const roundedKm = Math.round(depthKm);
      if (Math.abs(depthKm - roundedKm) < 0.05) {
        return String(roundedKm) + "k m";
      }
      return String(Number(depthKm.toFixed(1))) + "k m";
    }
    if (Math.abs(depthM - Math.round(depthM)) < 0.05) {
      return String(Math.round(depthM)) + " m";
    }
    return String(Number(depthM.toFixed(depthM < 10.0 ? 1 : 0))) + " m";
  }

  function updateDepthTicks(state, depthLevels) {
    const depthTicks = state.elements.depthTicks;
    if (!depthTicks) {
      return;
    }
    depthTicks.replaceChildren();
    depthLevels.forEach(function (depthLevel, index) {
      const tick = document.createElement("span");
      const offsetPercent = depthLevels.length <= 1 ? 0.0 : (index / (depthLevels.length - 1)) * 100.0;
      tick.className = "globe-depth-ticks__tick";
      tick.style.left = String(offsetPercent) + "%";
      tick.textContent = formatDepthMeters(depthLevel);
      depthTicks.appendChild(tick);
    });
  }

  function rasterLayerToggles(state) {
    return [
      state.elements.predictionToggle,
      state.elements.groundTruthToggle,
      state.elements.absoluteErrorToggle,
      state.elements.uncertaintyToggle,
    ].filter(Boolean);
  }

  function hasActiveRasterSelection(state) {
    return rasterLayerToggles(state).some(function (toggle) {
      return toggle.checked && !toggle.disabled;
    });
  }

  function ensureRasterSelection(state) {
    // The raster group intentionally supports an empty selection for a clear base globe.
    rasterLayerToggles(state).forEach(function (toggle) {
      if (toggle.disabled) {
        toggle.checked = false;
      }
    });
  }

  function syncRasterLayerVisibility(state) {
    const elements = state.elements;
    const showPrediction = Boolean(elements.predictionToggle && elements.predictionToggle.checked);
    const showGroundTruth = Boolean(elements.groundTruthToggle && elements.groundTruthToggle.checked);
    const showAbsoluteError = Boolean(elements.absoluteErrorToggle && elements.absoluteErrorToggle.checked);
    const showUncertainty = Boolean(elements.uncertaintyToggle && elements.uncertaintyToggle.checked);
    if (state.predictionLayer) {
      state.predictionLayer.show = showPrediction;
    }
    if (state.groundTruthLayer) {
      state.groundTruthLayer.show = showGroundTruth;
    }
    if (state.absoluteErrorLayer) {
      state.absoluteErrorLayer.show = showAbsoluteError;
    }
    if (state.uncertaintyLayer) {
      state.uncertaintyLayer.show = showUncertainty;
    }
    updateValueLegend(state);
    updateAbsoluteErrorLegend(state);
    requestRender(state);
  }

  function updateVariableControl(state) {
    const variables = getVariableConfigs(state.config);
    const elements = state.elements;
    if (!elements.variableControl || !elements.variableRadios) {
      return;
    }
    if (!variables) {
      elements.variableControl.hidden = true;
      return;
    }
    elements.variableControl.hidden = false;
    elements.variableRadios.forEach(function (radio) {
      const available = Boolean(variables[radio.value]);
      radio.disabled = !available;
      radio.checked = available && radio.value === state.selectedVariable;
      const label = radio.closest("label");
      if (label) {
        label.hidden = !available;
      }
    });
  }

  function updateValueLegend(state) {
    const elements = state.elements;
    if (!elements.valueLegend) {
      return;
    }
    elements.valueLegend.hidden = !hasActiveRasterSelection(state);
    if (elements.valueLegend.hidden) {
      return;
    }
    const activeConfig = activeVariableConfig(state);
    const colorScale = resolveColorScale(activeConfig);
    const unitLabel = valueUnitLabel(activeConfig);
    const variableLabel = String(activeConfig.variable_label || activeConfig.variable || "Temperature");
    if (elements.valueLegendTitle) {
      elements.valueLegendTitle.textContent = variableLabel;
    }
    if (elements.valueLegendMin) {
      elements.valueLegendMin.textContent = formatLegendValue(colorScale.min, unitLabel);
    }
    if (elements.valueLegendMax) {
      elements.valueLegendMax.textContent = formatLegendValue(colorScale.max, unitLabel);
    }
    if (elements.valueLegendBar) {
      elements.valueLegendBar.classList.toggle(
        "globe-legend__bar--salinity",
        activeVariableKey(state) === "salinity"
      );
    }
  }

  function updateDepthControl(state) {
    const activeConfig = activeVariableConfig(state);
    const depthLevels = getDepthLevels(activeConfig);
    const depthLevel = selectedDepthLevel(state);
    if (state.elements.depthSlider) {
      state.elements.depthSlider.min = "0";
      state.elements.depthSlider.max = String(Math.max(0, depthLevels.length - 1));
      state.elements.depthSlider.step = "1";
      state.elements.depthSlider.value = String(
        clamp(Number(state.selectedDepthIndex || 0), 0, depthLevels.length - 1)
      );
      state.elements.depthSlider.disabled = depthLevels.length <= 1;
    }
    if (state.elements.depthLabel) {
      state.elements.depthLabel.textContent = String(depthLevel.label || "Surface");
    }
    if (state.elements.predictionToggle) {
      setToggleAvailable(state.elements.predictionToggle, Boolean(depthLevel.prediction_tiles_url));
    }
    if (state.elements.groundTruthToggle) {
      setToggleAvailable(state.elements.groundTruthToggle, Boolean(depthLevel.ground_truth_tiles_url));
    }
    if (state.elements.absoluteErrorToggle) {
      setToggleAvailable(state.elements.absoluteErrorToggle, hasAbsoluteErrorLayer(depthLevel));
    }
    if (state.elements.uncertaintyToggle) {
      const uncertaintyAvailable = hasUncertaintyLayer(activeConfig);
      setToggleAvailable(state.elements.uncertaintyToggle, uncertaintyAvailable);
      setToggleLabelHidden(state.elements.uncertaintyToggle, !uncertaintyAvailable);
    }
    ensureRasterSelection(state);
    updateValueLegend(state);
    updateDepthTicks(state, depthLevels);
  }

  function colorForTemperature(tempC, colorScale) {
    if (!Number.isFinite(tempC)) {
      return Cesium.Color.WHITE;
    }

    const clampedTemp = clamp(tempC, colorScale.min, colorScale.max);
    let lowerStop = TEMPERATURE_COLOR_STOPS[0];
    let upperStop = TEMPERATURE_COLOR_STOPS[TEMPERATURE_COLOR_STOPS.length - 1];
    for (let index = 1; index < TEMPERATURE_COLOR_STOPS.length; index += 1) {
      const candidate = TEMPERATURE_COLOR_STOPS[index];
      if (clampedTemp <= candidate.value) {
        upperStop = candidate;
        lowerStop = TEMPERATURE_COLOR_STOPS[index - 1];
        break;
      }
    }

    if (upperStop.value <= lowerStop.value) {
      return Cesium.Color.fromBytes(lowerStop.rgb[0], lowerStop.rgb[1], lowerStop.rgb[2], 255);
    }

    const t = (clampedTemp - lowerStop.value) / (upperStop.value - lowerStop.value);
    return Cesium.Color.fromBytes(
      Math.round(lerp(lowerStop.rgb[0], upperStop.rgb[0], t)),
      Math.round(lerp(lowerStop.rgb[1], upperStop.rgb[1], t)),
      Math.round(lerp(lowerStop.rgb[2], upperStop.rgb[2], t)),
      255
    );
  }

  function formatCoordinate(value, positiveLabel, negativeLabel) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
      return "unknown";
    }
    const direction = numericValue >= 0.0 ? positiveLabel : negativeLabel;
    return Math.abs(numericValue).toFixed(4) + " deg " + direction;
  }

  function positionToLonLat(entity, now) {
    if (!entity || !entity.position) {
      return null;
    }
    const position = entity.position.getValue(now);
    if (!position) {
      return null;
    }
    const cartographic = Cesium.Cartographic.fromCartesian(position);
    return {
      lon: Cesium.Math.toDegrees(cartographic.longitude),
      lat: Cesium.Math.toDegrees(cartographic.latitude),
    };
  }

  function buildMarkerImage(shapeMarkup, sizePx) {
    const svg = [
      '<svg xmlns="http://www.w3.org/2000/svg" width="',
      String(sizePx),
      '" height="',
      String(sizePx),
      '" viewBox="0 0 64 64">',
      shapeMarkup,
      "</svg>",
    ].join("");
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }

  function stylePointEntity(entity, options) {
    if (!entity.billboard) {
      return;
    }

    // Replace Cesium's default GeoJSON pins with custom SVG billboards so each
    // observation type has a distinct, cleaner visual signature.
    entity.billboard.image = options.image;
    entity.billboard.width = options.width;
    entity.billboard.height = options.height;
    entity.billboard.scale = 1;
    entity.billboard.verticalOrigin = Cesium.VerticalOrigin.BOTTOM;
    entity.billboard.horizontalOrigin = Cesium.HorizontalOrigin.CENTER;
    entity.billboard.heightReference = Cesium.HeightReference.CLAMP_TO_GROUND;
    entity.billboard.pixelOffset = new Cesium.Cartesian2(0, options.pixelOffsetY);
    // Keep markers anchored to their geographic position instead of shifting
    // them toward the camera, which makes them appear to drift while orbiting.
    entity.billboard.eyeOffset = Cesium.Cartesian3.ZERO;
  }

  function stylePointEntities(dataSource, options) {
    dataSource.entities.values.forEach(function (entity) {
      stylePointEntity(entity, options);
    });
  }

  function markerKindForEntity(entity, now) {
    const properties = entity && entity.properties ? entity.properties : null;
    if (!properties) {
      return "argo";
    }
    if (properties.marker_kind) {
      const markerKind = String(properties.marker_kind.getValue(now) || "argo");
      if (markerKind !== "argo") {
        return markerKind;
      }
    }
    if (properties.has_full_depth_graph) {
      if (properties.has_full_depth_graph.getValue(now)) {
        return "full_depth_profile";
      }
    }
    if (properties.graph_png_path) {
      const graphPath = properties.graph_png_path.getValue(now);
      return graphPath ? "full_depth_profile" : "argo";
    }
    return "argo";
  }

  function entityHasFullDepthGraph(entity, now) {
    return markerKindForEntity(entity, now) === "full_depth_profile";
  }

  function styleArgoSampleEntities(dataSource) {
    const now = Cesium.JulianDate.now();
    dataSource.entities.values.forEach(function (entity) {
      if (!entity.billboard) {
        return;
      }

      const isFullDepthProfile = entityHasFullDepthGraph(entity, now);
      stylePointEntity(entity, {
        image: isFullDepthProfile ? FULL_SAMPLE_MARKER_IMAGE : ARGO_POINT_MARKER_IMAGE,
        width: isFullDepthProfile ? 34 : 24,
        height: isFullDepthProfile ? 34 : 24,
        pixelOffsetY: isFullDepthProfile ? -2 : 0,
      });
    });
  }

  function formatFullSampleTitle(locationId) {
    const raw = String(locationId || "").trim();
    const match = raw.match(/^full_sample_(\d+)$/i);
    if (match) {
      return "Full Sample #" + match[1];
    }
    return raw || "Full Sample";
  }

  function updateArgoLegendVisibility(state) {
    const argoLegend = state.elements.argoLegend;
    if (!argoLegend) {
      return;
    }
    const activeConfig = activeVariableConfig(state);
    const hasCombinedLayer = Boolean(activeConfig && activeConfig.argo_sample_locations_url);
    argoLegend.hidden = !hasCombinedLayer || !state.elements.pointsToggle.checked;
  }

  function hasAbsoluteErrorLayer(depthLevel) {
    return Boolean(depthLevel && depthLevel.absolute_error_tiles_url);
  }

  function hasUncertaintyLayer(config) {
    return Boolean(config && config.uncertainty_tiles_url);
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

  function updateAbsoluteErrorLegend(state) {
    const errorLegend = state.elements.errorLegend;
    if (!errorLegend) {
      return;
    }
    const depthLevel = selectedDepthLevel(state);
    const activeConfig = activeVariableConfig(state);
    const showUncertainty = Boolean(
      state.elements.uncertaintyToggle &&
        state.elements.uncertaintyToggle.checked &&
        hasUncertaintyLayer(activeConfig)
    );
    const showAbsoluteError = Boolean(
      state.elements.absoluteErrorToggle &&
        state.elements.absoluteErrorToggle.checked &&
        hasAbsoluteErrorLayer(depthLevel)
    );
    const unitLabel = showUncertainty
      ? activeConfig.uncertainty_value_unit_label || valueUnitLabel(activeConfig)
      : depthLevel.absolute_error_value_unit_label || valueUnitLabel(activeConfig);
    const visible = showUncertainty || showAbsoluteError;
    errorLegend.hidden = !visible;
    if (visible && state.elements.errorLegendTitle) {
      state.elements.errorLegendTitle.textContent = showUncertainty ? "Uncertainty" : "Absolute Error";
    }
    if (visible && state.elements.errorLegendMin) {
      const legendMin = showUncertainty
        ? firstFiniteNumber([activeConfig.uncertainty_legend_min], 0.0)
        : firstFiniteNumber(
            [depthLevel.absolute_error_legend_min, depthLevel.absolute_error_legend_min_c],
            0.0
          );
      state.elements.errorLegendMin.textContent = formatLegendValue(legendMin, unitLabel);
    }
    if (visible && state.elements.errorLegendMax) {
      const legendMax = showUncertainty
        ? firstFiniteNumber([activeConfig.uncertainty_legend_max], NaN)
        : firstFiniteNumber(
            [depthLevel.absolute_error_legend_max, depthLevel.absolute_error_legend_max_c],
            NaN
          );
      const colorScaleMax = showUncertainty
        ? firstFiniteNumber([activeConfig.uncertainty_color_scale_max], NaN)
        : firstFiniteNumber(
            [depthLevel.absolute_error_color_scale_max, depthLevel.absolute_error_color_scale_max_c],
            NaN
          );
      state.elements.errorLegendMax.textContent = formatLegendValue(
        Number.isFinite(legendMax) ? legendMax : colorScaleMax,
        unitLabel
      );
    }
  }

  function cleanedClosedPatchPositions(positions) {
    const cleanedPositions = [];
    positions.forEach(function (position) {
      const previousPosition = cleanedPositions[cleanedPositions.length - 1];
      if (
        position &&
        (!previousPosition ||
          !Cesium.Cartesian3.equalsEpsilon(
            position,
            previousPosition,
            Cesium.Math.EPSILON7
          ))
      ) {
        cleanedPositions.push(position);
      }
    });
    if (cleanedPositions.length < 3) {
      return null;
    }
    const firstPosition = cleanedPositions[0];
    const lastPosition = cleanedPositions[cleanedPositions.length - 1];
    if (
      !Cesium.Cartesian3.equalsEpsilon(
        firstPosition,
        lastPosition,
        Cesium.Math.EPSILON7
      )
    ) {
      cleanedPositions.push(firstPosition);
    }
    return cleanedPositions;
  }

  function stylePatchSplitEntities(dataSource) {
    const fillColor = Cesium.Color.fromCssColorString(PATCH_FILL_COLOR).withAlpha(PATCH_FILL_ALPHA);
    const outlineColor = Cesium.Color.fromCssColorString(PATCH_OUTLINE_COLOR);
    const now = Cesium.JulianDate.now();
    dataSource.entities.values.forEach(function (entity) {
      entity.billboard = null;
      entity.label = null;
      if (!entity.polygon) {
        return;
      }

      // Keep patch overlays readable above imagery without hiding the raster values.
      entity.polygon.material = fillColor;
      entity.polygon.outline = false;
      const hierarchy = entity.polygon.hierarchy
        ? entity.polygon.hierarchy.getValue(now)
        : null;
      const positions = hierarchy && hierarchy.positions ? hierarchy.positions : null;
      if (positions && positions.length > 1) {
        const borderPositions = cleanedClosedPatchPositions(positions);
        if (!borderPositions) {
          return;
        }
        entity.polyline = new Cesium.PolylineGraphics({
          positions: borderPositions,
          material: outlineColor,
          width: PATCH_OUTLINE_WIDTH,
          arcType: Cesium.ArcType.GEODESIC,
          clampToGround: false,
        });
      }
    });
  }

  function clearProfilePopupCloseTimer(state) {
    if (state.profilePopupCloseTimer !== null) {
      window.clearTimeout(state.profilePopupCloseTimer);
      state.profilePopupCloseTimer = null;
    }
  }

  function clearToolbarCollapseTimer(state) {
    if (typeof state.toolbarCollapseTimer !== "undefined" && state.toolbarCollapseTimer !== null) {
      window.clearTimeout(state.toolbarCollapseTimer);
      state.toolbarCollapseTimer = null;
    }
  }

  function setToolbarCollapsed(elements, collapsed) {
    if (!elements.toolbar || !elements.toolbarToggle || !elements.toolbarContent) {
      return;
    }

    elements.toolbar.classList.toggle("is-collapsed", collapsed);
    elements.toolbarToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    elements.toolbarToggle.textContent = collapsed ? "Show settings" : "Hide settings";
  }

  function setCinematicMode(state, enabled) {
    state.cinematicEnabled = Boolean(enabled);
    if (state.elements.stage) {
      state.elements.stage.classList.toggle("is-cinematic", state.cinematicEnabled);
    }
    if (state.elements.cinematicToggle) {
      state.elements.cinematicToggle.setAttribute(
        "aria-pressed",
        state.cinematicEnabled ? "true" : "false"
      );
      state.elements.cinematicToggle.textContent = state.cinematicEnabled
        ? "Exit cinematic"
        : "Cinematic";
    }
    if (state.elements.cinematicExit) {
      state.elements.cinematicExit.hidden = !state.cinematicEnabled;
    }
    if (state.cinematicEnabled) {
      closeProfilePopup(state);
    }
    forceViewerResize(state);
    requestRender(state);
  }

  function finalizeProfilePopupClose(state) {
    const profilePopup = state.elements.profilePopup;
    const profilePopupImage = state.elements.profilePopupImage;
    if (!profilePopup) {
      return;
    }
    clearProfilePopupCloseTimer(state);
    profilePopup.classList.remove("is-open", "is-closing");
    profilePopup.hidden = true;
    if (profilePopupImage) {
      // Clear the image only after the fade-out so the close animation stays visible.
      profilePopupImage.removeAttribute("src");
      profilePopupImage.alt = "";
    }
    requestRender(state);
  }

  function closeProfilePopup(state) {
    const profilePopup = state.elements.profilePopup;
    if (!profilePopup) {
      return;
    }
    if (profilePopup.hidden && !profilePopup.classList.contains("is-open")) {
      finalizeProfilePopupClose(state);
      return;
    }
    clearProfilePopupCloseTimer(state);
    profilePopup.classList.remove("is-open");
    profilePopup.classList.add("is-closing");
    state.profilePopupCloseTimer = window.setTimeout(function () {
      finalizeProfilePopupClose(state);
    }, PROFILE_POPUP_CLOSE_DELAY_MS);
    requestRender(state);
  }

  function openProfilePopup(state) {
    const elements = state.elements;
    clearProfilePopupCloseTimer(state);
    elements.profilePopup.hidden = false;
    elements.profilePopup.classList.remove("is-closing");
    window.requestAnimationFrame(function () {
      elements.profilePopup.classList.add("is-open");
      requestRender(state);
    });
  }

  function showProfilePopup(state, entity) {
    const elements = state.elements;
    if (!elements.profilePopup || !elements.profilePopupImage) {
      return;
    }
    const now = Cesium.JulianDate.now();
    const properties = entity.properties;
    if (!properties || !properties.graph_png_path) {
      return;
    }
    const graphPath = properties.graph_png_path.getValue(now);
    if (!graphPath) {
      return;
    }
    const locationId = properties.location_id ? properties.location_id.getValue(now) : "Full Sample";
    const activeConfig = activeVariableConfig(state);
    const dateValue = properties.date ? properties.date.getValue(now) : activeConfig.selected_date;
    const patchId = properties.patch_id ? properties.patch_id.getValue(now) : "";
    const pixelRow = properties.pixel_row ? properties.pixel_row.getValue(now) : null;
    const pixelCol = properties.pixel_col ? properties.pixel_col.getValue(now) : null;

    if (elements.profilePopupTitle) {
      elements.profilePopupTitle.textContent = formatFullSampleTitle(locationId);
    }
    if (elements.profilePopupSubtitle) {
      elements.profilePopupSubtitle.textContent =
        "Week: " +
        formatCompactIsoWeek(dateValue) +
        "\nPatch " +
        String(patchId || "") +
        ", pixel (" +
        String(pixelRow) +
        ", " +
        String(pixelCol) +
        ")";
    }
    elements.profilePopupImage.src = new URL(String(graphPath), state.configUrl).toString();
    elements.profilePopupImage.alt = formatFullSampleTitle(locationId) + " profile comparison";
    elements.profilePopupImage.hidden = false;
    openProfilePopup(state);
  }

  function showArgoPointPopup(state, entity) {
    const elements = state.elements;
    if (!elements.profilePopup || !elements.profilePopupImage) {
      return;
    }
    const now = Cesium.JulianDate.now();
    const properties = entity.properties;
    const lonLat = positionToLonLat(entity, now);
    const dateValue =
      properties && properties.date
        ? properties.date.getValue(now)
        : activeVariableConfig(state).selected_date;
    const patchId = properties && properties.patch_id ? properties.patch_id.getValue(now) : "";
    const pixelRow = properties && properties.pixel_row ? properties.pixel_row.getValue(now) : null;
    const pixelCol = properties && properties.pixel_col ? properties.pixel_col.getValue(now) : null;
    const locationText = lonLat
      ? formatCoordinate(lonLat.lat, "N", "S") +
        ", " +
        formatCoordinate(lonLat.lon, "E", "W")
      : "Unknown location";

    if (elements.profilePopupTitle) {
      elements.profilePopupTitle.textContent = "Argo observation";
    }
    if (elements.profilePopupSubtitle) {
      // Prefer observation metadata from the GeoJSON, but keep the location visible even
      // for older point files that were exported before these properties were retained.
      elements.profilePopupSubtitle.textContent =
        "Location: " +
        locationText +
        "\nWeek: " +
        formatCompactIsoWeek(dateValue) +
        (patchId || pixelRow !== null || pixelCol !== null
          ? "\nPatch " +
            String(patchId || "") +
            ", pixel (" +
            String(pixelRow) +
            ", " +
            String(pixelCol) +
            ")"
          : "");
    }
    elements.profilePopupImage.hidden = true;
    elements.profilePopupImage.removeAttribute("src");
    elements.profilePopupImage.alt = "";
    openProfilePopup(state);
  }

  function enforceOverlayOrder(state) {
    const imageryLayers = state.viewer.imageryLayers;
    if (state.groundTruthLayer) {
      // Keep GLORYS above prediction without moving prediction below the basemap.
      imageryLayers.raise(state.groundTruthLayer);
    }
    if (state.absoluteErrorLayer) {
      // Error tiles use a different color scale, so keep them above temperature rasters.
      imageryLayers.raise(state.absoluteErrorLayer);
    }
    if (state.uncertaintyLayer) {
      imageryLayers.raise(state.uncertaintyLayer);
    }

    const dataSources = state.viewer.dataSources;
    if (state.patchSplitsDataSource) {
      dataSources.lowerToBottom(state.patchSplitsDataSource);
    }
    if (state.pointsDataSource) {
      dataSources.raiseToTop(state.pointsDataSource);
    }
  }

  function addBaseMap(viewer, config, configUrl) {
    const naturalEarthUrl = Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII");

    function addProvider(url, credit, fallback) {
      Cesium.TileMapServiceImageryProvider.fromUrl(url, { credit: credit })
        .then(function (provider) {
          if (!viewer.isDestroyed()) {
            const baseLayer = viewer.imageryLayers.addImageryProvider(provider);
            // The basemap may resolve after overlay layers, so pin it to the bottom
            // of the stack to keep prediction and GLORYS imagery visible above it.
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
      const hostedCredit =
        config.base_map_credit || (config.credits && config.credits.base_map) || "Natural Earth II";
      addProvider(hostedBaseMapUrl, hostedCredit, addBundledNaturalEarthFallback);
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

    // The square stage height is derived from layout, so watch the actual container
    // rather than only the window. Otherwise Cesium can keep an older, shorter size.
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
    if (
      !Number.isFinite(destination.lon) ||
      !Number.isFinite(destination.lat) ||
      !Number.isFinite(destination.height)
    ) {
      return;
    }

    state.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        destination.lon,
        destination.lat,
        destination.height
      ),
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

  function setSpinEnabled(state, enabled) {
    state.spinEnabled = enabled;
    state.lastSpinTime = null;
    // Cesium's onTick fires every frame, but clock.currentTime only advances
    // while animation is enabled, so keep the viewer clock in sync with the UI.
    state.viewer.clock.shouldAnimate = enabled;
    if (state.elements.spinToggle) {
      state.elements.spinToggle.setAttribute("aria-pressed", enabled ? "true" : "false");
      state.elements.spinToggle.textContent = enabled ? "Stop Spin" : "Spin Globe";
    }
    requestRender(state);
  }

  function activeRasterLayerKey(state) {
    if (state.elements.uncertaintyToggle && state.elements.uncertaintyToggle.checked) {
      return "uncertainty";
    }
    if (state.elements.absoluteErrorToggle && state.elements.absoluteErrorToggle.checked) {
      return "error";
    }
    if (state.elements.groundTruthToggle && state.elements.groundTruthToggle.checked) {
      return "glorys";
    }
    if (state.elements.predictionToggle && state.elements.predictionToggle.checked) {
      return "prediction";
    }
    return "off";
  }

  function temperatureExportColorStops() {
    const firstStop = TEMPERATURE_COLOR_STOPS[0];
    const lastStop = TEMPERATURE_COLOR_STOPS[TEMPERATURE_COLOR_STOPS.length - 1];
    const span = Math.max(1.0, lastStop.value - firstStop.value);
    return TEMPERATURE_COLOR_STOPS.map(function (stop) {
      return {
        offset: clamp((stop.value - firstStop.value) / span, 0.0, 1.0),
        rgb: stop.rgb,
      };
    });
  }

  function buildPictureLegendModel(state) {
    const depthLevel = selectedDepthLevel(state);
    const activeConfig = activeVariableConfig(state);
    const rasterKey = activeRasterLayerKey(state);
    if (rasterKey === "uncertainty" && hasUncertaintyLayer(activeConfig)) {
      const unitLabel = activeConfig.uncertainty_value_unit_label || valueUnitLabel(activeConfig);
      const legendMin = firstFiniteNumber([activeConfig.uncertainty_legend_min], 0.0);
      const legendMax = firstFiniteNumber([activeConfig.uncertainty_legend_max], NaN);
      const colorScaleMax = firstFiniteNumber([activeConfig.uncertainty_color_scale_max], NaN);
      return {
        colorStops: ERROR_EXPORT_COLOR_STOPS,
        element: state.elements.errorLegend,
        maxLabel: formatLegendValue(Number.isFinite(legendMax) ? legendMax : colorScaleMax, unitLabel),
        minLabel: formatLegendValue(legendMin, unitLabel),
        title: "Uncertainty",
      };
    }

    if (rasterKey === "error" && hasAbsoluteErrorLayer(depthLevel)) {
      const unitLabel = depthLevel.absolute_error_value_unit_label || valueUnitLabel(activeConfig);
      const legendMin = firstFiniteNumber(
        [depthLevel.absolute_error_legend_min, depthLevel.absolute_error_legend_min_c],
        0.0
      );
      const legendMax = firstFiniteNumber(
        [depthLevel.absolute_error_legend_max, depthLevel.absolute_error_legend_max_c],
        NaN
      );
      const colorScaleMax = firstFiniteNumber(
        [depthLevel.absolute_error_color_scale_max, depthLevel.absolute_error_color_scale_max_c],
        NaN
      );
      return {
        colorStops: ERROR_EXPORT_COLOR_STOPS,
        element: state.elements.errorLegend,
        maxLabel: formatLegendValue(Number.isFinite(legendMax) ? legendMax : colorScaleMax, unitLabel),
        minLabel: formatLegendValue(legendMin, unitLabel),
        title: "Absolute Error",
      };
    }

    const colorScale = resolveColorScale(activeConfig);
    const unitLabel = valueUnitLabel(activeConfig);
    return {
      colorStops: activeVariableKey(state) === "salinity"
        ? SALINITY_EXPORT_COLOR_STOPS
        : temperatureExportColorStops(),
      element: state.elements.valueLegend,
      maxLabel: formatLegendValue(colorScale.max, unitLabel),
      minLabel: formatLegendValue(colorScale.min, unitLabel),
      title: String(activeConfig.variable_label || activeConfig.variable || "Temperature"),
    };
  }

  function shouldExportPointLegend(state) {
    const activeConfig = activeVariableConfig(state);
    return Boolean(
      activeConfig &&
        activeConfig.argo_sample_locations_url &&
        state.elements.pointsToggle &&
        state.elements.pointsToggle.checked
    );
  }

  function addCanvasGradientStops(gradient, colorStops) {
    colorStops.forEach(function (stop) {
      gradient.addColorStop(
        clamp(Number(stop.offset), 0.0, 1.0),
        "rgb(" + stop.rgb.join(", ") + ")"
      );
    });
  }

  function drawPicturePointLegend(ctx, x, y, scale) {
    const markerSize = 13 * scale;
    ctx.save();
    ctx.lineWidth = 2 * scale;
    ctx.strokeStyle = "#7cf5ff";
    ctx.fillStyle = "rgba(6, 23, 38, 0.82)";
    ctx.beginPath();
    ctx.arc(x, y, markerSize * 0.55, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#dffcff";
    ctx.beginPath();
    ctx.arc(x, y, markerSize * 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(234, 248, 255, 0.92)";
    ctx.fillText("ARGO", x + 18 * scale, y + 4 * scale);

    const diamondY = y + 29 * scale;
    ctx.strokeStyle = "#ffd166";
    ctx.fillStyle = "rgba(9, 34, 56, 0.9)";
    ctx.beginPath();
    ctx.moveTo(x, diamondY - markerSize * 0.7);
    ctx.lineTo(x + markerSize * 0.7, diamondY - markerSize * 0.2);
    ctx.lineTo(x + markerSize * 0.45, diamondY + markerSize * 0.68);
    ctx.lineTo(x, diamondY + markerSize * 0.92);
    ctx.lineTo(x - markerSize * 0.45, diamondY + markerSize * 0.68);
    ctx.lineTo(x - markerSize * 0.7, diamondY - markerSize * 0.2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#fff4cf";
    ctx.beginPath();
    ctx.arc(x, diamondY, markerSize * 0.28, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(234, 248, 255, 0.92)";
    ctx.fillText("Full-depth ARGO", x + 18 * scale, diamondY + 4 * scale);
    ctx.restore();
  }

  function drawRoundedRectPath(ctx, x, y, width, height, radius) {
    const clampedRadius = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + clampedRadius, y);
    ctx.lineTo(x + width - clampedRadius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + clampedRadius);
    ctx.lineTo(x + width, y + height - clampedRadius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - clampedRadius, y + height);
    ctx.lineTo(x + clampedRadius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - clampedRadius);
    ctx.lineTo(x, y + clampedRadius);
    ctx.quadraticCurveTo(x, y, x + clampedRadius, y);
    ctx.closePath();
  }

  function scaledElementDimension(element, scale, dimensionName, fallbackCssPixels) {
    const rect = element && element.getBoundingClientRect ? element.getBoundingClientRect() : null;
    const elementPixels = rect ? rect[dimensionName] : 0;
    return Math.round((elementPixels > 0 ? elementPixels : fallbackCssPixels) * scale);
  }

  function drawPictureLegendCard(ctx, x, y, width, height, legend, scale) {
    const borderWidth = Math.max(1, scale);
    const radius = 16 * scale;
    const paddingX = 13.6 * scale;
    const paddingY = 12 * scale;
    const gap = 7.2 * scale;
    const titleFontSize = 11.52 * scale;
    const labelFontSize = 13.12 * scale;
    const barHeight = 11.2 * scale;
    const contentLeft = x + paddingX;
    const contentRight = x + width - paddingX;
    const barY = y + paddingY + titleFontSize * 1.2 + gap;
    const labelY = barY + barHeight + gap;

    ctx.save();
    drawRoundedRectPath(ctx, x, y, width, height, radius);
    ctx.fillStyle = "rgba(5, 20, 32, 0.86)";
    ctx.fill();
    ctx.lineWidth = borderWidth;
    ctx.strokeStyle = "rgba(124, 200, 255, 0.16)";
    ctx.stroke();

    ctx.fillStyle = "#7cc8ff";
    ctx.font = "700 " + Math.round(titleFontSize) + "px Roboto, Arial, sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText(legend.title.toUpperCase(), contentLeft, y + paddingY);

    const gradient = ctx.createLinearGradient(contentLeft, barY, contentRight, barY);
    addCanvasGradientStops(gradient, legend.colorStops);
    drawRoundedRectPath(ctx, contentLeft, barY, contentRight - contentLeft, barHeight, barHeight / 2);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.lineWidth = borderWidth;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.stroke();

    ctx.fillStyle = "rgba(234, 248, 255, 0.92)";
    ctx.font = "600 " + Math.round(labelFontSize) + "px Roboto, Arial, sans-serif";
    ctx.fillText(legend.minLabel, contentLeft, labelY);
    ctx.textAlign = "right";
    ctx.fillText(legend.maxLabel, contentRight, labelY);
    ctx.restore();
  }

  function drawPicturePointLegendCard(ctx, x, y, width, height, scale) {
    const borderWidth = Math.max(1, scale);
    const radius = 16 * scale;
    const paddingX = 13.6 * scale;
    const paddingY = 12 * scale;

    ctx.save();
    drawRoundedRectPath(ctx, x, y, width, height, radius);
    ctx.fillStyle = "rgba(5, 20, 32, 0.86)";
    ctx.fill();
    ctx.lineWidth = borderWidth;
    ctx.strokeStyle = "rgba(124, 200, 255, 0.16)";
    ctx.stroke();

    ctx.fillStyle = "#7cc8ff";
    ctx.font = "700 " + Math.round(11.52 * scale) + "px Roboto, Arial, sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText("ARGO SAMPLES", x + paddingX, y + paddingY);
    ctx.font = "600 " + Math.round(13.12 * scale) + "px Roboto, Arial, sans-serif";
    drawPicturePointLegend(ctx, x + paddingX + 10 * scale, y + paddingY + 36 * scale, scale);
    ctx.restore();
  }

  function drawPictureLegend(ctx, outputCanvas, state, scale) {
    if (!hasActiveRasterSelection(state)) {
      return;
    }
    const legend = buildPictureLegendModel(state);
    const margin = 16 * scale;
    const gap = 12 * scale;
    const legendWidth = scaledElementDimension(legend.element, scale, "width", 235.2);
    const legendHeight = scaledElementDimension(legend.element, scale, "height", 74.4);
    const includePointLegend = shouldExportPointLegend(state) && outputCanvas.width >= 540 * scale;
    const pointLegendWidth = includePointLegend
      ? scaledElementDimension(state.elements.argoLegend, scale, "width", 235.2)
      : 0;
    const pointLegendHeight = includePointLegend
      ? scaledElementDimension(state.elements.argoLegend, scale, "height", 96)
      : 0;
    const totalWidth = legendWidth + (includePointLegend ? gap + pointLegendWidth : 0);
    const legendX = Math.max(margin, outputCanvas.width - margin - legendWidth);
    const legendY = Math.max(margin, outputCanvas.height - margin - legendHeight);

    if (includePointLegend) {
      // Match the website legend stack by keeping any point legend as its own compact card.
      const pointX = Math.max(margin, outputCanvas.width - margin - totalWidth);
      const pointY = Math.max(margin, outputCanvas.height - margin - pointLegendHeight);
      drawPicturePointLegendCard(ctx, pointX, pointY, pointLegendWidth, pointLegendHeight, scale);
    }
    drawPictureLegendCard(ctx, legendX, legendY, legendWidth, legendHeight, legend, scale);
  }

  function buildPictureCanvas(state) {
    const sourceCanvas = state.viewer && state.viewer.scene ? state.viewer.scene.canvas : null;
    if (!sourceCanvas || !sourceCanvas.width || !sourceCanvas.height) {
      throw new Error("Cesium canvas is not ready for export.");
    }
    const scale = Math.max(
      1.0,
      sourceCanvas.width / Math.max(1, sourceCanvas.clientWidth || sourceCanvas.width)
    );
    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = sourceCanvas.width;
    outputCanvas.height = sourceCanvas.height;
    const ctx = outputCanvas.getContext("2d");
    if (!ctx) {
      throw new Error("Could not create PNG export canvas.");
    }
    ctx.fillStyle = "#04131f";
    ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    ctx.drawImage(sourceCanvas, 0, 0, sourceCanvas.width, sourceCanvas.height);
    drawPictureLegend(ctx, outputCanvas, state, scale);
    return outputCanvas;
  }

  function waitForAnimationFrame() {
    return new Promise(function (resolve) {
      window.requestAnimationFrame(resolve);
    });
  }

  async function waitForPictureRender(state) {
    forceViewerResize(state);
    requestRender(state);
    await waitForAnimationFrame();
    requestRender(state);
    await waitForAnimationFrame();
  }

  function sanitizeFilenamePart(value) {
    const sanitized = String(value || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return sanitized || "view";
  }

  function buildPictureFilename(state) {
    const activeConfig = activeVariableConfig(state);
    const depthLevel = selectedDepthLevel(state);
    const parts = [
      "depthdif",
      activeVariableKey(state),
      activeRasterLayerKey(state),
      formatDepthMeters(depthLevel),
      activeConfig.selected_date || state.config.selected_date || "current",
    ];
    return parts.map(sanitizeFilenamePart).join("_") + ".png";
  }

  function setPictureExportButtonState(state, label, disabled) {
    const button = state.elements.pictureExportButton;
    if (!button) {
      return;
    }
    button.textContent = label;
    button.disabled = disabled;
  }

  function resetPictureExportButtonLater(state) {
    if (state.pictureExportStatusTimer !== null) {
      window.clearTimeout(state.pictureExportStatusTimer);
    }
    state.pictureExportStatusTimer = window.setTimeout(function () {
      state.pictureExportStatusTimer = null;
      setPictureExportButtonState(state, "Export PNG", false);
    }, 1600);
  }

  function downloadCanvasPng(canvas, filename) {
    return new Promise(function (resolve, reject) {
      try {
        canvas.toBlob(function (blob) {
          if (!blob) {
            reject(new Error("Could not render PNG export."));
            return;
          }
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          window.setTimeout(function () {
            URL.revokeObjectURL(url);
          }, 0);
          resolve();
        }, "image/png");
      } catch (error) {
        reject(error);
      }
    });
  }

  async function handlePictureExport(state) {
    if (state.pictureExportInProgress) {
      return;
    }
    state.pictureExportInProgress = true;
    if (state.pictureExportStatusTimer !== null) {
      window.clearTimeout(state.pictureExportStatusTimer);
      state.pictureExportStatusTimer = null;
    }
    setPictureExportButtonState(state, "Rendering...", true);
    try {
      await waitForPictureRender(state);
      const pictureCanvas = buildPictureCanvas(state);
      await downloadCanvasPng(pictureCanvas, buildPictureFilename(state));
      setPictureExportButtonState(state, "Saved PNG", false);
    } catch (error) {
      console.error(error);
      setPictureExportButtonState(state, "Export failed", false);
    } finally {
      state.pictureExportInProgress = false;
      resetPictureExportButtonLater(state);
    }
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

      const deltaSeconds = Cesium.JulianDate.secondsDifference(
        clock.currentTime,
        state.lastSpinTime
      );
      state.lastSpinTime = Cesium.JulianDate.clone(clock.currentTime);
      if (deltaSeconds <= 0.0) {
        requestRender(state);
        return;
      }

      // Rotate around the Earth's axis to keep the spin centered on the globe
      // rather than panning in screen space.
      state.viewer.scene.camera.rotate(
        Cesium.Cartesian3.UNIT_Z,
        -SPIN_RATE_RADIANS_PER_SECOND * deltaSeconds
      );
      requestRender(state);
    };
    state.viewer.clock.onTick.addEventListener(state.spinTickListener);
  }

  async function addPredictionLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const depthLevel = selectedDepthLevel(state);
    const predictionUrl = resolveAssetUrl(depthLevel.prediction_tiles_url, state.configUrl);
    if (!predictionUrl) {
      markToggleUnavailable(state.elements.predictionToggle);
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(predictionUrl, {
      credit: activeConfig.credits && activeConfig.credits.prediction,
    });
    const layer = state.viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = 1.0;
    layer.show = state.elements.predictionToggle.checked;
    requestRender(state);
    return layer;
  }

  async function addGroundTruthLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const depthLevel = selectedDepthLevel(state);
    const groundTruthUrl = resolveAssetUrl(depthLevel.ground_truth_tiles_url, state.configUrl);
    if (!groundTruthUrl) {
      markToggleUnavailable(state.elements.groundTruthToggle);
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(groundTruthUrl, {
      credit: activeConfig.credits && activeConfig.credits.ground_truth,
    });
    const layer = state.viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = 1.0;
    layer.show = state.elements.groundTruthToggle.checked;
    return layer;
  }

  async function addAbsoluteErrorLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const depthLevel = selectedDepthLevel(state);
    const errorUrl = resolveAssetUrl(depthLevel.absolute_error_tiles_url, state.configUrl);
    if (!errorUrl) {
      markToggleUnavailable(state.elements.absoluteErrorToggle);
      updateAbsoluteErrorLegend(state);
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(errorUrl, {
      credit: activeConfig.credits && activeConfig.credits.absolute_error,
    });
    const layer = state.viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = 1.0;
    layer.show = state.elements.absoluteErrorToggle.checked;
    updateAbsoluteErrorLegend(state);
    return layer;
  }

  async function addUncertaintyLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const uncertaintyUrl = resolveAssetUrl(activeConfig.uncertainty_tiles_url, state.configUrl);
    if (!uncertaintyUrl) {
      markToggleUnavailable(state.elements.uncertaintyToggle);
      setToggleLabelHidden(state.elements.uncertaintyToggle, true);
      updateAbsoluteErrorLegend(state);
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(uncertaintyUrl, {
      credit: activeConfig.credits && activeConfig.credits.uncertainty,
    });
    const layer = state.viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = 1.0;
    layer.show = state.elements.uncertaintyToggle.checked;
    updateAbsoluteErrorLegend(state);
    return layer;
  }

  async function addPointsLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const pointsUrl = resolveAssetUrl(activeConfig.argo_sample_locations_url, state.configUrl);
    if (!pointsUrl) {
      markToggleUnavailable(state.elements.pointsToggle);
      updateArgoLegendVisibility(state);
      return null;
    }
    const dataSource = await Cesium.GeoJsonDataSource.load(pointsUrl, {
      clampToGround: true,
      markerColor: Cesium.Color.fromCssColorString("#7cf5ff"),
      markerSize: 10,
      stroke: Cesium.Color.BLACK,
      strokeWidth: 1,
      credit: activeConfig.credits && activeConfig.credits.points,
    });
    styleArgoSampleEntities(dataSource);
    state.viewer.dataSources.add(dataSource);
    dataSource.show = state.elements.pointsToggle.checked;
    updateArgoLegendVisibility(state);
    return dataSource;
  }

  async function addPatchSplitsLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const patchSplitsUrl = resolveAssetUrl(activeConfig.patch_splits_url, state.configUrl);
    if (!patchSplitsUrl) {
      markToggleUnavailable(state.elements.patchSplitsToggle);
      return null;
    }
    const dataSource = await Cesium.GeoJsonDataSource.load(patchSplitsUrl, {
      clampToGround: true,
      credit: activeConfig.credits && activeConfig.credits.patch_splits,
    });
    stylePatchSplitEntities(dataSource);
    state.viewer.dataSources.add(dataSource);
    dataSource.show = state.elements.patchSplitsToggle.checked;
    return dataSource;
  }

  async function loadOptionalLayer(state, toggle, stateKey, promiseKey, loadLayer) {
    if (state[stateKey]) {
      return state[stateKey];
    }
    if (state[promiseKey]) {
      return state[promiseKey];
    }

    setToggleLoading(toggle, true);
    state[promiseKey] = loadLayer(state)
      .then(function (layerOrSource) {
        state[stateKey] = layerOrSource;
        enforceOverlayOrder(state);
        requestRender(state);
        return layerOrSource;
      })
      .finally(function () {
        state[promiseKey] = null;
        setToggleLoading(toggle, false);
      });

    return state[promiseKey];
  }

  async function ensureGroundTruthLayer(state) {
    if (state.groundTruthLayer) {
      return state.groundTruthLayer;
    }
    if (state.groundTruthLayerLoadPromise) {
      return state.groundTruthLayerLoadPromise;
    }

    const rasterDepthReloadToken = state.rasterDepthReloadToken;
    setToggleLoading(state.elements.groundTruthToggle, true);
    const loadPromise = addGroundTruthLayer(state)
      .then(function (layer) {
        if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
          if (layer) {
            // Drop late-arriving imagery from an older depth so it cannot remain
            // visible underneath the currently selected depth layer.
            state.viewer.imageryLayers.remove(layer, true);
          }
          requestRender(state);
          return null;
        }
        state.groundTruthLayer = layer;
        enforceOverlayOrder(state);
        requestRender(state);
        return layer;
      })
      .finally(function () {
        if (state.groundTruthLayerLoadPromise === loadPromise) {
          state.groundTruthLayerLoadPromise = null;
          setToggleLoading(state.elements.groundTruthToggle, false);
        }
      });

    state.groundTruthLayerLoadPromise = loadPromise;
    return state.groundTruthLayerLoadPromise;
  }

  async function ensureAbsoluteErrorLayer(state) {
    if (state.absoluteErrorLayer) {
      return state.absoluteErrorLayer;
    }
    if (state.absoluteErrorLayerLoadPromise) {
      return state.absoluteErrorLayerLoadPromise;
    }

    const rasterDepthReloadToken = state.rasterDepthReloadToken;
    setToggleLoading(state.elements.absoluteErrorToggle, true);
    const loadPromise = addAbsoluteErrorLayer(state)
      .then(function (layer) {
        if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
          if (layer) {
            state.viewer.imageryLayers.remove(layer, true);
          }
          requestRender(state);
          return null;
        }
        state.absoluteErrorLayer = layer;
        enforceOverlayOrder(state);
        updateAbsoluteErrorLegend(state);
        requestRender(state);
        return layer;
      })
      .finally(function () {
        if (state.absoluteErrorLayerLoadPromise === loadPromise) {
          state.absoluteErrorLayerLoadPromise = null;
          setToggleLoading(state.elements.absoluteErrorToggle, false);
        }
      });

    state.absoluteErrorLayerLoadPromise = loadPromise;
    return state.absoluteErrorLayerLoadPromise;
  }

  async function ensureUncertaintyLayer(state) {
    if (state.uncertaintyLayer) {
      return state.uncertaintyLayer;
    }
    if (state.uncertaintyLayerLoadPromise) {
      return state.uncertaintyLayerLoadPromise;
    }

    const rasterDepthReloadToken = state.rasterDepthReloadToken;
    setToggleLoading(state.elements.uncertaintyToggle, true);
    const loadPromise = addUncertaintyLayer(state)
      .then(function (layer) {
        if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
          if (layer) {
            state.viewer.imageryLayers.remove(layer, true);
          }
          requestRender(state);
          return null;
        }
        state.uncertaintyLayer = layer;
        enforceOverlayOrder(state);
        updateAbsoluteErrorLegend(state);
        requestRender(state);
        return layer;
      })
      .finally(function () {
        if (state.uncertaintyLayerLoadPromise === loadPromise) {
          state.uncertaintyLayerLoadPromise = null;
          setToggleLoading(state.elements.uncertaintyToggle, false);
        }
      });

    state.uncertaintyLayerLoadPromise = loadPromise;
    return state.uncertaintyLayerLoadPromise;
  }

  async function ensurePointsLayer(state) {
    return loadOptionalLayer(
      state,
      state.elements.pointsToggle,
      "pointsDataSource",
      "pointsDataSourceLoadPromise",
      addPointsLayer
    );
  }

  async function ensurePatchSplitsLayer(state) {
    return loadOptionalLayer(
      state,
      state.elements.patchSplitsToggle,
      "patchSplitsDataSource",
      "patchSplitsDataSourceLoadPromise",
      addPatchSplitsLayer
    );
  }

  function preloadOptionalLayers(state) {
    if (!state || state.preloadTaskId !== null) {
      return;
    }

    state.preloadTaskId = scheduleBackgroundTask(function () {
      state.preloadTaskId = null;
      const loaders = [
        ensureGroundTruthLayer,
        ensureAbsoluteErrorLayer,
        ensureUncertaintyLayer,
        ensurePointsLayer,
        ensurePatchSplitsLayer,
      ];

      // Warm hidden layers in the background so toggle changes become mostly
      // instant without competing with the first visible prediction render.
      loaders.reduce(function (chain, loadLayer) {
        return chain.then(function () {
          if (!window.__depthdifCesiumGlobeState || window.__depthdifCesiumGlobeState !== state) {
            return null;
          }
          return loadLayer(state).catch(function (error) {
            console.error(error);
            return null;
          });
        });
      }, Promise.resolve());
    });
  }

  async function handleRasterLayerToggle(state, toggle, ensureLayer) {
    if (!toggle) {
      return;
    }
    if (!toggle.checked) {
      syncRasterLayerVisibility(state);
      return;
    }

    syncRasterLayerVisibility(state);
    if (!ensureLayer) {
      return;
    }

    try {
      const layer = await ensureLayer(state);
      if (!layer) {
        ensureRasterSelection(state);
      }
      syncRasterLayerVisibility(state);
      enforceOverlayOrder(state);
    } catch (error) {
      toggle.checked = false;
      setToggleLoading(toggle, false);
      ensureRasterSelection(state);
      syncRasterLayerVisibility(state);
      console.error(error);
    }
  }

  async function handleOptionalLayerToggle(state, toggle, stateKey, ensureLayer) {
    if (!toggle) {
      return;
    }

    if (!toggle.checked) {
      if (state[stateKey]) {
        state[stateKey].show = false;
      }
      if (stateKey === "pointsDataSource") {
        updateArgoLegendVisibility(state);
      }
      if (stateKey === "absoluteErrorLayer") {
        updateAbsoluteErrorLegend(state);
      }
      requestRender(state);
      return;
    }

    try {
      const layerOrSource = await ensureLayer(state);
      if (layerOrSource) {
        layerOrSource.show = toggle.checked;
        if (stateKey === "pointsDataSource") {
          updateArgoLegendVisibility(state);
        }
        if (stateKey === "absoluteErrorLayer") {
          updateAbsoluteErrorLegend(state);
        }
        requestRender(state);
      }
    } catch (error) {
      toggle.checked = false;
      selectOffRadioForToggle(toggle);
      setToggleLoading(toggle, false);
      if (stateKey === "absoluteErrorLayer") {
        updateAbsoluteErrorLegend(state);
      }
      console.error(error);
    }
  }

  async function reloadRasterDepthLayers(state) {
    const imageryLayers = state.viewer.imageryLayers;
    const showPrediction = state.elements.predictionToggle.checked;
    const showGroundTruth = state.elements.groundTruthToggle.checked;
    const showAbsoluteError = state.elements.absoluteErrorToggle.checked;
    const showUncertainty = state.elements.uncertaintyToggle.checked;
    const rasterDepthReloadToken = state.rasterDepthReloadToken + 1;
    state.rasterDepthReloadToken = rasterDepthReloadToken;

    if (state.predictionLayer) {
      imageryLayers.remove(state.predictionLayer, true);
      state.predictionLayer = null;
    }
    if (state.groundTruthLayer) {
      imageryLayers.remove(state.groundTruthLayer, true);
      state.groundTruthLayer = null;
    }
    if (state.absoluteErrorLayer) {
      imageryLayers.remove(state.absoluteErrorLayer, true);
      state.absoluteErrorLayer = null;
    }
    if (state.uncertaintyLayer) {
      imageryLayers.remove(state.uncertaintyLayer, true);
      state.uncertaintyLayer = null;
    }
    state.predictionLayerLoadPromise = null;
    state.groundTruthLayerLoadPromise = null;
    state.absoluteErrorLayerLoadPromise = null;
    state.uncertaintyLayerLoadPromise = null;

    try {
      const predictionLayer = await addPredictionLayer(state);
      if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
        if (predictionLayer) {
          imageryLayers.remove(predictionLayer, true);
        }
        return;
      }
      state.predictionLayer = predictionLayer;
      if (predictionLayer) {
        predictionLayer.show = showPrediction;
      }
      if (showGroundTruth) {
        const groundTruthLayer = await addGroundTruthLayer(state);
        if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
          if (groundTruthLayer) {
            imageryLayers.remove(groundTruthLayer, true);
          }
          return;
        }
        state.groundTruthLayer = groundTruthLayer;
        if (groundTruthLayer) {
          groundTruthLayer.show = true;
        }
      }
      if (showAbsoluteError) {
        const absoluteErrorLayer = await addAbsoluteErrorLayer(state);
        if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
          if (absoluteErrorLayer) {
            imageryLayers.remove(absoluteErrorLayer, true);
          }
          return;
        }
        state.absoluteErrorLayer = absoluteErrorLayer;
        if (absoluteErrorLayer) {
          absoluteErrorLayer.show = true;
        }
      }
      if (showUncertainty) {
        const uncertaintyLayer = await addUncertaintyLayer(state);
        if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
          if (uncertaintyLayer) {
            imageryLayers.remove(uncertaintyLayer, true);
          }
          return;
        }
        state.uncertaintyLayer = uncertaintyLayer;
        if (uncertaintyLayer) {
          uncertaintyLayer.show = true;
        }
      }
      enforceOverlayOrder(state);
      updateAbsoluteErrorLegend(state);
      requestRender(state);
    } catch (error) {
      console.error(error);
    }
  }

  function removeDataSourceIfPresent(state, stateKey, promiseKey) {
    if (state[stateKey]) {
      state.viewer.dataSources.remove(state[stateKey], true);
      state[stateKey] = null;
    }
    state[promiseKey] = null;
  }

  async function reloadVariableLayers(state) {
    state.selectedDepthIndex = 0;
    updateVariableControl(state);
    updateDepthControl(state);
    updateAbsoluteErrorLegend(state);
    closeProfilePopup(state);
    removeDataSourceIfPresent(state, "pointsDataSource", "pointsDataSourceLoadPromise");
    removeDataSourceIfPresent(state, "patchSplitsDataSource", "patchSplitsDataSourceLoadPromise");
    await reloadRasterDepthLayers(state);
    if (state.elements.pointsToggle && state.elements.pointsToggle.checked) {
      ensurePointsLayer(state).catch(function (error) {
        console.error(error);
      });
    }
    if (state.elements.patchSplitsToggle && state.elements.patchSplitsToggle.checked) {
      ensurePatchSplitsLayer(state).catch(function (error) {
        console.error(error);
      });
    }
    updateArgoLegendVisibility(state);
    requestRender(state);
  }

  function wireUi(state) {
    const elements = state.elements;
    if (elements.toolbarToggle) {
      elements.toolbarToggle.addEventListener("click", function () {
        const collapsed = elements.toolbar
          ? elements.toolbar.classList.contains("is-collapsed")
          : false;
        setToolbarCollapsed(elements, !collapsed);
        requestRender(state);
      });
    }

    function wireRasterRadio(toggle, ensureLayer) {
      const pointerTarget = toggle.closest("label") || toggle;
      const rememberCheckedState = function () {
        toggle.dataset.globeWasChecked = toggle.checked ? "true" : "false";
      };
      pointerTarget.addEventListener("pointerdown", rememberCheckedState);
      toggle.addEventListener("keydown", function (event) {
        if ((event.key === " " || event.key === "Enter") && toggle.checked && !toggle.disabled) {
          event.preventDefault();
          toggle.checked = false;
          syncRasterLayerVisibility(state);
        }
      });
      toggle.addEventListener("click", function (event) {
        if (toggle.dataset.globeWasChecked === "true" && toggle.checked && !toggle.disabled) {
          event.preventDefault();
          toggle.checked = false;
          syncRasterLayerVisibility(state);
        }
        delete toggle.dataset.globeWasChecked;
      });
      toggle.addEventListener("change", function () {
        handleRasterLayerToggle(state, toggle, ensureLayer);
      });
    }

    wireRasterRadio(elements.predictionToggle, null);
    wireRasterRadio(elements.groundTruthToggle, ensureGroundTruthLayer);
    wireRasterRadio(elements.absoluteErrorToggle, ensureAbsoluteErrorLayer);
    wireRasterRadio(elements.uncertaintyToggle, ensureUncertaintyLayer);

    if (elements.pointsRadios) {
      elements.pointsRadios.forEach(function (radio) {
        radio.addEventListener("change", function () {
          if (!radio.checked) {
            return;
          }
          handleOptionalLayerToggle(
            state,
            elements.pointsToggle,
            "pointsDataSource",
            ensurePointsLayer
          );
        });
      });
    }

    if (elements.patchSplitsRadios) {
      elements.patchSplitsRadios.forEach(function (radio) {
        radio.addEventListener("change", function () {
          if (!radio.checked) {
            return;
          }
          handleOptionalLayerToggle(
            state,
            elements.patchSplitsToggle,
            "patchSplitsDataSource",
            ensurePatchSplitsLayer
          );
        });
      });
    }

    if (elements.variableRadios) {
      elements.variableRadios.forEach(function (radio) {
        radio.addEventListener("change", function () {
          if (!radio.checked || radio.disabled || radio.value === state.selectedVariable) {
            return;
          }
          state.selectedVariable = radio.value;
          reloadVariableLayers(state);
        });
      });
    }

    if (elements.depthSlider) {
      elements.depthSlider.addEventListener("input", function () {
        state.selectedDepthIndex = Number(elements.depthSlider.value);
        updateDepthControl(state);
        updateAbsoluteErrorLegend(state);
        reloadRasterDepthLayers(state);
      });
    }

    if (elements.spinToggle) {
      elements.spinToggle.addEventListener("click", function () {
        setSpinEnabled(state, !state.spinEnabled);
      });
    }

    elements.resetButton.addEventListener("click", function () {
      if (state.viewer && state.config) {
        flyToConfig(state);
      }
    });

    if (elements.pictureExportButton) {
      elements.pictureExportButton.addEventListener("click", function () {
        handlePictureExport(state);
      });
    }

    if (elements.cinematicToggle) {
      elements.cinematicToggle.addEventListener("click", function () {
        setCinematicMode(state, !state.cinematicEnabled);
      });
    }

    if (elements.cinematicExit) {
      elements.cinematicExit.addEventListener("click", function () {
        setCinematicMode(state, false);
      });
    }

    if (elements.profilePopupClose) {
      elements.profilePopupClose.addEventListener("click", function () {
        closeProfilePopup(state);
      });
    }
    if (elements.profilePopup) {
      elements.profilePopup.addEventListener("click", function (event) {
        if (event.target === elements.profilePopup) {
          closeProfilePopup(state);
        }
      });
    }
  }

  function cleanupState(state) {
    if (!state) {
      return;
    }
    clearProfilePopupCloseTimer(state);
    clearToolbarCollapseTimer(state);
    if (state.pictureExportStatusTimer !== null) {
      window.clearTimeout(state.pictureExportStatusTimer);
      state.pictureExportStatusTimer = null;
    }
    if (state.elements && state.elements.stage) {
      state.elements.stage.classList.remove("is-cinematic");
    }
    if (state.elements && state.elements.cinematicExit) {
      state.elements.cinematicExit.hidden = true;
    }
    if (state.stopWatchingResize) {
      state.stopWatchingResize();
      state.stopWatchingResize = null;
    }
    if (state.handleWindowResize) {
      window.removeEventListener("resize", state.handleWindowResize);
      state.handleWindowResize = null;
    }
    cancelBackgroundTask(state.preloadTaskId);
    state.preloadTaskId = null;
    if (state.viewer && !state.viewer.isDestroyed()) {
      if (state.spinTickListener) {
        state.viewer.clock.onTick.removeEventListener(state.spinTickListener);
      }
      state.viewer.destroy();
    }
    if (state.elements && state.elements.container) {
      delete state.elements.container.dataset.globeInitialized;
    }
  }

  function destroyDepthDifCesiumGlobe() {
    nextInitToken();
    cleanupState(window.__depthdifCesiumGlobeState || null);
    window.__depthdifCesiumGlobeState = null;
  }

  async function initDepthDifCesiumGlobe() {
    const elements = getGlobeElements();
    if (!elements || typeof window.Cesium === "undefined") {
      return false;
    }

    const activeState = window.__depthdifCesiumGlobeState || null;
    if (activeState && activeState.elements.container === elements.container) {
      return true;
    }

    destroyDepthDifCesiumGlobe();
    const initToken = getCurrentInitToken();

    try {
      const loaded = await loadConfig();
      if (initToken !== getCurrentInitToken()) {
        return false;
      }

      const viewer = buildViewer(elements.container, loaded.config, loaded.configUrl);
      const state = {
        config: loaded.config,
        configUrl: loaded.configUrl,
        viewer: viewer,
        elements: elements,
        predictionLayer: null,
        groundTruthLayer: null,
        absoluteErrorLayer: null,
        uncertaintyLayer: null,
        pointsDataSource: null,
        patchSplitsDataSource: null,
        predictionLayerLoadPromise: null,
        groundTruthLayerLoadPromise: null,
        absoluteErrorLayerLoadPromise: null,
        uncertaintyLayerLoadPromise: null,
        pointsDataSourceLoadPromise: null,
        patchSplitsDataSourceLoadPromise: null,
        selectedVariable: resolveDefaultVariable(loaded.config),
        selectedDepthIndex: 0,
        rasterDepthReloadToken: 0,
        spinEnabled: false,
        cinematicEnabled: false,
        lastSpinTime: null,
        profilePopupCloseTimer: null,
        pictureExportInProgress: false,
        pictureExportStatusTimer: null,
        stopWatchingResize: null,
        handleWindowResize: null,
        spinTickListener: null,
        preloadTaskId: null,
      };
      window.__depthdifCesiumGlobeState = state;
      updatePageHeader(elements, loaded.config);
      updateVariableControl(state);
      updateDepthControl(state);
      updateAbsoluteErrorLegend(state);

      try {
        state.predictionLayer = await addPredictionLayer(state);
      } catch (error) {
        // Keep the base globe interactive even when the hosted overlay tiles are
        // unavailable. Otherwise one failing asset host tears down the whole page.
        markToggleUnavailable(elements.predictionToggle);
        console.error(error);
        state.predictionLayer = null;
      }
      if (initToken !== getCurrentInitToken()) {
        cleanupState(state);
        return false;
      }

      enforceOverlayOrder(state);
      attachSpinLoop(state);
      setSpinEnabled(state, false);
      wireUi(state);
      setToolbarCollapsed(elements, false);
      setCinematicMode(state, false);
      updateArgoLegendVisibility(state);
      updateAbsoluteErrorLegend(state);
      viewer.screenSpaceEventHandler.setInputAction(function (movement) {
        const picked = viewer.scene.pick(movement.position);
        if (!picked || !picked.id) {
          return;
        }
        if (state.pointsDataSource && state.pointsDataSource.entities.contains(picked.id)) {
          if (entityHasFullDepthGraph(picked.id, Cesium.JulianDate.now())) {
            showProfilePopup(state, picked.id);
            return;
          }
          showArgoPointPopup(state, picked.id);
        }
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
      flyToConfig(state);
      forceViewerResize(state);
      state.stopWatchingResize = watchContainerResize(state, elements.container);
      state.handleWindowResize = function () {
        forceViewerResize(state);
      };
      window.addEventListener("resize", state.handleWindowResize);
      elements.container.dataset.globeInitialized = "true";
      preloadOptionalLayers(state);
      requestRender(state);
      return true;
    } catch (error) {
      if (initToken === getCurrentInitToken()) {
        destroyDepthDifCesiumGlobe();
      }
      console.error(error);
      return false;
    }
  }

  window.initDepthDifCesiumGlobe = initDepthDifCesiumGlobe;
  window.destroyDepthDifCesiumGlobe = destroyDepthDifCesiumGlobe;
})();
