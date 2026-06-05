(function () {
  const DEFAULT_COMPARISON_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/comparison-globe/comparison-globe-config.json";
  const DEFAULT_CAMERA_DESTINATION = {
    lon: -38.56452881619089,
    lat: 34.53988238358822,
    height: 9500000.0,
  };
  const SPIN_RATE_RADIANS_PER_SECOND = 5.0 * (Math.PI / 180.0);
  const DEFAULT_COLOR_SCALE = { min: 0.0, max: 30.0 };
  const PATCH_FILL_ALPHA = 0.18;
  const PATCH_FILL_COLOR = "#f97316";
  const PATCH_OUTLINE_COLOR = "#fb923c";
  const PATCH_OUTLINE_WIDTH = 2.75;
  const PROFILE_POPUP_CLOSE_DELAY_MS = 180;

  const RASTER_LAYERS = [
    { key: "glorys", label: "GLORYS", toggleId: "globe-toggle-glorys" },
    { key: "depthdif", label: "DepthDif", toggleId: "globe-toggle-depthdif" },
    { key: "idw", label: "IDW", toggleId: "globe-toggle-idw" },
    { key: "lstm", label: "LSTM", toggleId: "globe-toggle-lstm" },
  ];

  const RASTER_LAYER_FIELD_ALIASES = {
    glorys: ["glorys_tiles_url", "ground_truth_tiles_url", "target_tiles_url"],
    depthdif: ["depthdif_tiles_url", "prediction_tiles_url"],
    idw: ["idw_tiles_url", "idw_prediction_tiles_url"],
    lstm: ["lstm_tiles_url", "lstm_prediction_tiles_url"],
  };

  const TEMPERATURE_EXPORT_COLOR_STOPS = [
    { offset: 0.0, rgb: [18, 38, 140] },
    { offset: 0.13, rgb: [30, 86, 196] },
    { offset: 0.27, rgb: [44, 140, 255] },
    { offset: 0.4, rgb: [58, 212, 255] },
    { offset: 0.53, rgb: [255, 238, 98] },
    { offset: 0.67, rgb: [255, 172, 54] },
    { offset: 0.8, rgb: [240, 84, 32] },
    { offset: 1.0, rgb: [180, 16, 26] },
  ];
  const SALINITY_EXPORT_COLOR_STOPS = [
    { offset: 0.0, rgb: [49, 54, 149] },
    { offset: 0.2, rgb: [69, 117, 180] },
    { offset: 0.4, rgb: [116, 173, 209] },
    { offset: 0.6, rgb: [171, 221, 164] },
    { offset: 0.8, rgb: [253, 224, 71] },
    { offset: 1.0, rgb: [168, 85, 36] },
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

  function getComparisonElements() {
    const container = document.getElementById("depthdif-cesium-globe");
    if (!container) {
      return null;
    }

    const layerToggles = {};
    RASTER_LAYERS.forEach(function (definition) {
      layerToggles[definition.key] = document.getElementById(definition.toggleId);
    });

    return {
      container: container,
      stage: container.closest(".globe-stage"),
      layerToggles: layerToggles,
      rasterRadios: document.querySelectorAll('input[name="globe-raster-layer"]'),
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
      profilePopup: document.getElementById("globe-profile-popup"),
      profilePopupTitle: document.getElementById("globe-profile-popup-title"),
      profilePopupSubtitle: document.getElementById("globe-profile-popup-subtitle"),
      profilePopupImage: document.getElementById("globe-profile-popup-image"),
      profilePopupClose: document.getElementById("globe-profile-popup-close"),
    };
  }

  function nextInitToken() {
    const nextToken = Number(window.__depthdifComparisonGlobeInitToken || 0) + 1;
    window.__depthdifComparisonGlobeInitToken = nextToken;
    return nextToken;
  }

  function getCurrentInitToken() {
    return Number(window.__depthdifComparisonGlobeInitToken || 0);
  }

  function requestRender(state) {
    if (!state || !state.viewer || state.viewer.isDestroyed()) {
      return;
    }
    state.viewer.scene.requestRender();
  }

  function resolveConfigUrl() {
    const params = new URLSearchParams(window.location.search);
    const explicitConfig = params.get("config");
    return explicitConfig || DEFAULT_COMPARISON_CONFIG_URL;
  }

  async function loadConfig() {
    const configUrl = resolveConfigUrl();
    const response = await fetch(configUrl);
    if (!response.ok) {
      throw new Error(
        "Failed to load comparison globe config: " +
          response.status +
          " " +
          response.statusText
      );
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
      return state.config || {};
    }
    return variables[state.selectedVariable] || variables[resolveDefaultVariable(state.config)] || {};
  }

  function activeVariableKey(state) {
    const activeConfig = activeVariableConfig(state);
    return String(activeConfig.variable || state.selectedVariable || "temperature");
  }

  function firstFiniteNumber(values, fallbackValue) {
    for (let index = 0; index < values.length; index += 1) {
      const value = Number(values[index]);
      if (Number.isFinite(value)) {
        return value;
      }
    }
    return fallbackValue;
  }

  function resolveColorScale(config) {
    const minValue = firstFiniteNumber([config.color_scale_min, config.color_scale_min_c], NaN);
    const maxValue = firstFiniteNumber([config.color_scale_max, config.color_scale_max_c], NaN);
    if (Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {
      return { min: minValue, max: maxValue };
    }
    return DEFAULT_COLOR_SCALE;
  }

  function valueUnitLabel(config) {
    return String(config.value_unit_label || config.value_units || "degC");
  }

  function formatLegendValue(value, unitLabel) {
    const numericValue = Number(value);
    const unit = String(unitLabel || "");
    const rounded = Number.isFinite(numericValue)
      ? Math.abs(numericValue - Math.round(numericValue)) < 0.05
        ? String(Math.round(numericValue))
        : String(Number(numericValue.toFixed(1)))
      : "0";
    return unit === "PSU" ? rounded + " PSU" : rounded + "\u00b0C";
  }

  function getDepthLevels(config) {
    if (config && Array.isArray(config.depth_levels) && config.depth_levels.length > 0) {
      return config.depth_levels;
    }
    return [
      {
        label: "10 m",
        requested_depth_m: 10.0,
        actual_depth_m: 10.0,
        layers: config && config.layers ? config.layers : null,
        glorys_tiles_url: config ? config.glorys_tiles_url || config.ground_truth_tiles_url : null,
        depthdif_tiles_url: config ? config.depthdif_tiles_url || config.prediction_tiles_url : null,
        idw_tiles_url: config ? config.idw_tiles_url : null,
        lstm_tiles_url: config ? config.lstm_tiles_url : null,
      },
    ];
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(maxValue, Math.max(minValue, value));
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
    if (Math.abs(depthM - Math.round(depthM)) < 0.05) {
      return String(Math.round(depthM)) + " m";
    }
    return String(Number(depthM.toFixed(depthM < 10.0 ? 1 : 0))) + " m";
  }

  function layerDefinitionForKey(key) {
    return (
      RASTER_LAYERS.find(function (definition) {
        return definition.key === key;
      }) || null
    );
  }

  function selectedRasterKey(state) {
    for (let index = 0; index < RASTER_LAYERS.length; index += 1) {
      const key = RASTER_LAYERS[index].key;
      const toggle = state.elements.layerToggles[key];
      if (toggle && toggle.checked && !toggle.disabled) {
        return key;
      }
    }
    return "off";
  }

  function layerConfigFromDepthLevel(depthLevel, key) {
    if (!depthLevel || !depthLevel.layers || typeof depthLevel.layers !== "object") {
      return null;
    }
    return depthLevel.layers[key] || null;
  }

  function layerTilesUrl(depthLevel, key) {
    const layerConfig = layerConfigFromDepthLevel(depthLevel, key);
    if (typeof layerConfig === "string") {
      return layerConfig;
    }
    if (layerConfig && typeof layerConfig === "object") {
      const configuredUrl =
        layerConfig.tiles_url || layerConfig.tile_url || layerConfig.tms_url || layerConfig.url;
      if (configuredUrl) {
        return configuredUrl;
      }
    }

    const aliases = RASTER_LAYER_FIELD_ALIASES[key] || [];
    for (let index = 0; index < aliases.length; index += 1) {
      const value = depthLevel ? depthLevel[aliases[index]] : null;
      if (value) {
        return value;
      }
    }
    return null;
  }

  function layerLabel(depthLevel, key) {
    const layerConfig = layerConfigFromDepthLevel(depthLevel, key);
    if (layerConfig && typeof layerConfig === "object" && layerConfig.label) {
      return String(layerConfig.label);
    }
    const definition = layerDefinitionForKey(key);
    return definition ? definition.label : key;
  }

  function layerCredit(activeConfig, depthLevel, key) {
    const layerConfig = layerConfigFromDepthLevel(depthLevel, key);
    if (layerConfig && typeof layerConfig === "object" && layerConfig.credit) {
      return layerConfig.credit;
    }
    if (activeConfig.credits && activeConfig.credits[key]) {
      return activeConfig.credits[key];
    }
    return undefined;
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

  function setToggleLoading(toggle, loading) {
    if (!toggle || toggle.dataset.globeUnavailable === "true") {
      return;
    }
    toggle.disabled = loading;
  }

  function ensureRasterSelection(state) {
    let hasValidSelection = false;
    RASTER_LAYERS.forEach(function (definition) {
      const toggle = state.elements.layerToggles[definition.key];
      if (!toggle) {
        return;
      }
      if (toggle.disabled) {
        toggle.checked = false;
        return;
      }
      hasValidSelection = hasValidSelection || toggle.checked;
    });
    if (hasValidSelection) {
      return;
    }

    const preferredKeys = ["depthdif", "glorys", "idw", "lstm"];
    for (let index = 0; index < preferredKeys.length; index += 1) {
      const toggle = state.elements.layerToggles[preferredKeys[index]];
      if (toggle && !toggle.disabled) {
        toggle.checked = true;
        return;
      }
    }
  }

  function hasActiveRasterSelection(state) {
    return selectedRasterKey(state) !== "off";
  }

  function syncRasterLayerVisibility(state) {
    RASTER_LAYERS.forEach(function (definition) {
      const layer = state.rasterLayers[definition.key];
      const toggle = state.elements.layerToggles[definition.key];
      if (layer) {
        layer.show = Boolean(toggle && toggle.checked && !toggle.disabled);
      }
    });
    updateValueLegend(state);
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
    const depthLevel = selectedDepthLevel(state);
    const rasterKey = selectedRasterKey(state);
    const colorScale = resolveColorScale(activeConfig);
    const unitLabel = valueUnitLabel(activeConfig);
    const variableLabel = String(activeConfig.variable_label || activeConfig.variable || "Temperature");
    const selectedLayerLabel = layerLabel(depthLevel, rasterKey);
    if (elements.valueLegendTitle) {
      elements.valueLegendTitle.textContent = selectedLayerLabel + " " + variableLabel;
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
      tick.textContent = formatDepthMeters(depthLevel).replace(/\s*m$/, "");
      depthTicks.appendChild(tick);
    });
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
      state.elements.depthLabel.textContent = String(depthLevel.label || formatDepthMeters(depthLevel));
    }

    RASTER_LAYERS.forEach(function (definition) {
      setToggleAvailable(
        state.elements.layerToggles[definition.key],
        Boolean(layerTilesUrl(depthLevel, definition.key))
      );
    });
    ensureRasterSelection(state);
    updateValueLegend(state);
    updateDepthTicks(state, depthLevels);
  }

  function resolveVectorUrl(config, primaryKey, fallbackKey) {
    if (!config) {
      return null;
    }
    return config[primaryKey] || config[fallbackKey] || null;
  }

  function resolveActiveVectorUrl(state, primaryKey, fallbackKey) {
    return (
      resolveVectorUrl(activeVariableConfig(state), primaryKey, fallbackKey) ||
      resolveVectorUrl(state.config, primaryKey, fallbackKey)
    );
  }

  function updateVectorControls(state) {
    const pointsUrl = resolveActiveVectorUrl(
      state,
      "argo_sample_locations_url",
      "argo_points_url"
    );
    const patchSplitsUrl = resolveActiveVectorUrl(
      state,
      "patch_splits_url",
      "inference_patches_url"
    );
    setToggleAvailable(state.elements.pointsToggle, Boolean(pointsUrl));
    setToggleAvailable(state.elements.patchSplitsToggle, Boolean(patchSplitsUrl));
    if (!pointsUrl) {
      selectOffRadioForToggle(state.elements.pointsToggle);
    }
    if (!patchSplitsUrl) {
      selectOffRadioForToggle(state.elements.patchSplitsToggle);
    }
    updateArgoLegendVisibility(state);
  }

  function updateArgoLegendVisibility(state) {
    const argoLegend = state.elements.argoLegend;
    if (!argoLegend) {
      return;
    }
    const hasPoints = Boolean(
      resolveActiveVectorUrl(state, "argo_sample_locations_url", "argo_points_url")
    );
    argoLegend.hidden = !hasPoints || !state.elements.pointsToggle || !state.elements.pointsToggle.checked;
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
    entity.billboard.image = options.image;
    entity.billboard.width = options.width;
    entity.billboard.height = options.height;
    entity.billboard.scale = 1;
    entity.billboard.verticalOrigin = Cesium.VerticalOrigin.BOTTOM;
    entity.billboard.horizontalOrigin = Cesium.HorizontalOrigin.CENTER;
    entity.billboard.heightReference = Cesium.HeightReference.CLAMP_TO_GROUND;
    entity.billboard.pixelOffset = new Cesium.Cartesian2(0, options.pixelOffsetY);
    entity.billboard.eyeOffset = Cesium.Cartesian3.ZERO;
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
    if (properties.has_full_depth_graph && properties.has_full_depth_graph.getValue(now)) {
      return "full_depth_profile";
    }
    if (properties.graph_png_path && properties.graph_png_path.getValue(now)) {
      return "full_depth_profile";
    }
    return "argo";
  }

  function styleArgoSampleEntities(dataSource) {
    const now = Cesium.JulianDate.now();
    dataSource.entities.values.forEach(function (entity) {
      if (!entity.billboard) {
        return;
      }
      const fullDepthProfile = markerKindForEntity(entity, now) === "full_depth_profile";
      stylePointEntity(entity, {
        image: fullDepthProfile ? FULL_SAMPLE_MARKER_IMAGE : ARGO_POINT_MARKER_IMAGE,
        width: fullDepthProfile ? 34 : 24,
        height: fullDepthProfile ? 34 : 24,
        pixelOffsetY: fullDepthProfile ? -2 : 0,
      });
    });
  }

  function cleanedClosedPatchPositions(positions) {
    const cleanedPositions = [];
    positions.forEach(function (position) {
      const previousPosition = cleanedPositions[cleanedPositions.length - 1];
      if (
        position &&
        (!previousPosition ||
          !Cesium.Cartesian3.equalsEpsilon(position, previousPosition, Cesium.Math.EPSILON7))
      ) {
        cleanedPositions.push(position);
      }
    });
    if (cleanedPositions.length < 3) {
      return null;
    }
    const firstPosition = cleanedPositions[0];
    const lastPosition = cleanedPositions[cleanedPositions.length - 1];
    if (!Cesium.Cartesian3.equalsEpsilon(firstPosition, lastPosition, Cesium.Math.EPSILON7)) {
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

      entity.polygon.material = fillColor;
      entity.polygon.outline = false;
      const hierarchy = entity.polygon.hierarchy ? entity.polygon.hierarchy.getValue(now) : null;
      const positions = hierarchy && hierarchy.positions ? hierarchy.positions : null;
      if (!positions || positions.length <= 1) {
        return;
      }
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
    });
  }

  function clearProfilePopupCloseTimer(state) {
    if (state.profilePopupCloseTimer !== null) {
      window.clearTimeout(state.profilePopupCloseTimer);
      state.profilePopupCloseTimer = null;
    }
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
      profilePopupImage.removeAttribute("src");
      profilePopupImage.alt = "";
      profilePopupImage.hidden = true;
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
    if (!elements.profilePopup) {
      return;
    }
    clearProfilePopupCloseTimer(state);
    elements.profilePopup.hidden = false;
    elements.profilePopup.classList.remove("is-closing");
    window.requestAnimationFrame(function () {
      elements.profilePopup.classList.add("is-open");
      requestRender(state);
    });
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

  function showArgoPointPopup(state, entity) {
    const elements = state.elements;
    if (!elements.profilePopup) {
      return;
    }
    const now = Cesium.JulianDate.now();
    const properties = entity.properties;
    const lonLat = positionToLonLat(entity, now);
    const patchId = properties && properties.patch_id ? properties.patch_id.getValue(now) : "";
    const pixelRow = properties && properties.pixel_row ? properties.pixel_row.getValue(now) : null;
    const pixelCol = properties && properties.pixel_col ? properties.pixel_col.getValue(now) : null;
    const locationText = lonLat
      ? formatCoordinate(lonLat.lat, "N", "S") + ", " + formatCoordinate(lonLat.lon, "E", "W")
      : "Unknown location";

    if (elements.profilePopupTitle) {
      elements.profilePopupTitle.textContent = "ARGO observation";
    }
    if (elements.profilePopupSubtitle) {
      elements.profilePopupSubtitle.textContent =
        "Location: " +
        locationText +
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
    if (elements.profilePopupImage) {
      elements.profilePopupImage.hidden = true;
      elements.profilePopupImage.removeAttribute("src");
      elements.profilePopupImage.alt = "";
    }
    openProfilePopup(state);
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
    state.viewer.clock.shouldAnimate = enabled;
    if (state.elements.spinToggle) {
      state.elements.spinToggle.setAttribute("aria-pressed", enabled ? "true" : "false");
      state.elements.spinToggle.textContent = enabled ? "Stop Spin" : "Spin Globe";
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

      const deltaSeconds = Cesium.JulianDate.secondsDifference(
        clock.currentTime,
        state.lastSpinTime
      );
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

  function enforceOverlayOrder(state) {
    RASTER_LAYERS.forEach(function (definition) {
      const layer = state.rasterLayers[definition.key];
      if (layer) {
        state.viewer.imageryLayers.raise(layer);
      }
    });
    if (state.patchSplitsDataSource) {
      state.viewer.dataSources.lowerToBottom(state.patchSplitsDataSource);
    }
    if (state.pointsDataSource) {
      state.viewer.dataSources.raiseToTop(state.pointsDataSource);
    }
  }

  async function addComparisonLayer(state, key) {
    const activeConfig = activeVariableConfig(state);
    const depthLevel = selectedDepthLevel(state);
    const tilesUrl = resolveAssetUrl(layerTilesUrl(depthLevel, key), state.configUrl);
    if (!tilesUrl) {
      markToggleUnavailable(state.elements.layerToggles[key]);
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(tilesUrl, {
      credit: layerCredit(activeConfig, depthLevel, key),
    });
    const layer = state.viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = 1.0;
    layer.show = Boolean(state.elements.layerToggles[key] && state.elements.layerToggles[key].checked);
    return layer;
  }

  async function ensureComparisonLayer(state, key) {
    if (state.rasterLayers[key]) {
      return state.rasterLayers[key];
    }
    if (state.rasterLayerLoadPromises[key]) {
      return state.rasterLayerLoadPromises[key];
    }

    const rasterDepthReloadToken = state.rasterDepthReloadToken;
    const toggle = state.elements.layerToggles[key];
    setToggleLoading(toggle, true);
    const loadPromise = addComparisonLayer(state, key)
      .then(function (layer) {
        if (rasterDepthReloadToken !== state.rasterDepthReloadToken) {
          if (layer) {
            state.viewer.imageryLayers.remove(layer, true);
          }
          requestRender(state);
          return null;
        }
        state.rasterLayers[key] = layer;
        enforceOverlayOrder(state);
        syncRasterLayerVisibility(state);
        return layer;
      })
      .finally(function () {
        if (state.rasterLayerLoadPromises[key] === loadPromise) {
          state.rasterLayerLoadPromises[key] = null;
          setToggleLoading(toggle, false);
        }
      });

    state.rasterLayerLoadPromises[key] = loadPromise;
    return loadPromise;
  }

  function removeRasterLayers(state) {
    RASTER_LAYERS.forEach(function (definition) {
      const layer = state.rasterLayers[definition.key];
      if (layer) {
        state.viewer.imageryLayers.remove(layer, true);
      }
      state.rasterLayers[definition.key] = null;
      state.rasterLayerLoadPromises[definition.key] = null;
    });
  }

  async function reloadRasterDepthLayers(state) {
    state.rasterDepthReloadToken += 1;
    const rasterKey = selectedRasterKey(state);
    removeRasterLayers(state);
    if (rasterKey === "off") {
      syncRasterLayerVisibility(state);
      return;
    }
    try {
      await ensureComparisonLayer(state, rasterKey);
      enforceOverlayOrder(state);
      syncRasterLayerVisibility(state);
    } catch (error) {
      const toggle = state.elements.layerToggles[rasterKey];
      if (toggle) {
        toggle.checked = false;
      }
      ensureRasterSelection(state);
      syncRasterLayerVisibility(state);
      console.error(error);
    }
  }

  async function addPointsLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const pointsUrl = resolveAssetUrl(
      resolveActiveVectorUrl(state, "argo_sample_locations_url", "argo_points_url"),
      state.configUrl
    );
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
    dataSource.show = Boolean(state.elements.pointsToggle && state.elements.pointsToggle.checked);
    updateArgoLegendVisibility(state);
    return dataSource;
  }

  async function addPatchSplitsLayer(state) {
    const activeConfig = activeVariableConfig(state);
    const patchSplitsUrl = resolveAssetUrl(
      resolveActiveVectorUrl(state, "patch_splits_url", "inference_patches_url"),
      state.configUrl
    );
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
    dataSource.show = Boolean(
      state.elements.patchSplitsToggle && state.elements.patchSplitsToggle.checked
    );
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
        requestRender(state);
      }
    } catch (error) {
      toggle.checked = false;
      selectOffRadioForToggle(toggle);
      setToggleLoading(toggle, false);
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
    updateVectorControls(state);
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

    if (elements.rasterRadios) {
      elements.rasterRadios.forEach(function (radio) {
        radio.addEventListener("change", function () {
          if (!radio.checked || radio.disabled) {
            syncRasterLayerVisibility(state);
            return;
          }
          ensureComparisonLayer(state, radio.value)
            .then(function () {
              enforceOverlayOrder(state);
              syncRasterLayerVisibility(state);
            })
            .catch(function (error) {
              radio.checked = false;
              ensureRasterSelection(state);
              syncRasterLayerVisibility(state);
              console.error(error);
            });
        });
      });
    }

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
        reloadRasterDepthLayers(state);
      });
    }

    if (elements.spinToggle) {
      elements.spinToggle.addEventListener("click", function () {
        setSpinEnabled(state, !state.spinEnabled);
      });
    }
    if (elements.resetButton) {
      elements.resetButton.addEventListener("click", function () {
        if (state.viewer && state.config) {
          flyToConfig(state);
        }
      });
    }
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
      "depthdif-comparison",
      activeVariableKey(state),
      selectedRasterKey(state),
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

  function addCanvasGradientStops(gradient, colorStops) {
    colorStops.forEach(function (stop) {
      gradient.addColorStop(
        clamp(Number(stop.offset), 0.0, 1.0),
        "rgb(" + stop.rgb.join(", ") + ")"
      );
    });
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

  function drawPictureLegend(ctx, outputCanvas, state, scale) {
    if (!hasActiveRasterSelection(state)) {
      return;
    }

    const activeConfig = activeVariableConfig(state);
    const depthLevel = selectedDepthLevel(state);
    const colorScale = resolveColorScale(activeConfig);
    const unitLabel = valueUnitLabel(activeConfig);
    const variableLabel = String(activeConfig.variable_label || activeConfig.variable || "Temperature");
    const title = layerLabel(depthLevel, selectedRasterKey(state)) + " " + variableLabel;
    const colorStops =
      activeVariableKey(state) === "salinity"
        ? SALINITY_EXPORT_COLOR_STOPS
        : TEMPERATURE_EXPORT_COLOR_STOPS;
    const margin = 16 * scale;
    const width = 235 * scale;
    const height = 74 * scale;
    const x = Math.max(margin, outputCanvas.width - margin - width);
    const y = Math.max(margin, outputCanvas.height - margin - height);
    const paddingX = 14 * scale;
    const paddingY = 12 * scale;
    const barY = y + paddingY + 22 * scale;
    const labelY = barY + 24 * scale;
    const contentLeft = x + paddingX;
    const contentRight = x + width - paddingX;

    ctx.save();
    drawRoundedRectPath(ctx, x, y, width, height, 16 * scale);
    ctx.fillStyle = "rgba(5, 20, 32, 0.86)";
    ctx.fill();
    ctx.lineWidth = Math.max(1, scale);
    ctx.strokeStyle = "rgba(124, 200, 255, 0.16)";
    ctx.stroke();

    ctx.fillStyle = "#7cc8ff";
    ctx.font = "700 " + Math.round(12 * scale) + "px Roboto, Arial, sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText(title.toUpperCase(), contentLeft, y + paddingY);

    const gradient = ctx.createLinearGradient(contentLeft, barY, contentRight, barY);
    addCanvasGradientStops(gradient, colorStops);
    drawRoundedRectPath(ctx, contentLeft, barY, contentRight - contentLeft, 11 * scale, 6 * scale);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.stroke();

    ctx.fillStyle = "rgba(234, 248, 255, 0.92)";
    ctx.font = "600 " + Math.round(13 * scale) + "px Roboto, Arial, sans-serif";
    ctx.fillText(formatLegendValue(colorScale.min, unitLabel), contentLeft, labelY);
    ctx.textAlign = "right";
    ctx.fillText(formatLegendValue(colorScale.max, unitLabel), contentRight, labelY);
    ctx.restore();
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

  function resolveComparisonWeekLabel(config) {
    const activeVariables = getVariableConfigs(config);
    const firstVariable = activeVariables
      ? activeVariables[resolveDefaultVariable(config)]
      : config || {};
    const label =
      config.comparison_label ||
      config.week_label ||
      firstVariable.comparison_label ||
      firstVariable.week_label;
    if (label) {
      return String(label);
    }
    const isoYear = firstFiniteNumber([config.iso_year, firstVariable.iso_year], NaN);
    const isoWeek = firstFiniteNumber([config.iso_week, firstVariable.iso_week], NaN);
    if (Number.isFinite(isoYear) && Number.isFinite(isoWeek)) {
      return String(Math.round(isoYear)) + "-W" + String(Math.round(isoWeek)).padStart(2, "0");
    }
    return "";
  }

  function updatePageHeader(elements, config) {
    const weekLabel = resolveComparisonWeekLabel(config);
    const titleText = "Model Comparison Globe";
    const descriptionText = weekLabel
      ? "Compare GLORYS, DepthDif, and baseline reconstructions at the 10 m depth level for " + weekLabel + "."
      : "Compare GLORYS, DepthDif, and baseline reconstructions at the 10 m depth level for the selected ocean variable.";

    document.title = weekLabel ? titleText + " " + weekLabel : titleText;
    const metaDescription = document.querySelector('meta[name="description"]');
    if (metaDescription) {
      metaDescription.setAttribute("content", descriptionText);
    }
    if (elements.pageEyebrow) {
      elements.pageEyebrow.textContent = weekLabel
        ? "Comparison Viewer " + weekLabel
        : "Comparison Viewer";
    }
    if (elements.pageTitle) {
      elements.pageTitle.textContent = titleText;
    }
    if (elements.pageDescription) {
      elements.pageDescription.textContent = descriptionText;
    }
    if (elements.mobileBlockTitle) {
      elements.mobileBlockTitle.textContent = weekLabel
        ? "Comparison globe visualization for " + weekLabel + " is disabled on mobile"
        : "Comparison globe visualization is disabled on mobile";
    }
    if (elements.mobileBlockText) {
      elements.mobileBlockText.textContent = weekLabel
        ? "Open the page on a desktop or laptop to load Cesium and inspect the " +
          weekLabel +
          " comparison globe."
        : "Open the page on a desktop or laptop to load Cesium and inspect the comparison globe.";
    }
  }

  function cleanupState(state) {
    if (!state) {
      return;
    }
    clearProfilePopupCloseTimer(state);
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

  function destroyDepthDifComparisonGlobe() {
    nextInitToken();
    cleanupState(window.__depthdifComparisonGlobeState || null);
    window.__depthdifComparisonGlobeState = null;
  }

  async function initDepthDifComparisonGlobe() {
    const elements = getComparisonElements();
    if (!elements || typeof window.Cesium === "undefined") {
      return false;
    }

    const activeState = window.__depthdifComparisonGlobeState || null;
    if (activeState && activeState.elements.container === elements.container) {
      return true;
    }

    destroyDepthDifComparisonGlobe();
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
        rasterLayers: {},
        rasterLayerLoadPromises: {},
        pointsDataSource: null,
        patchSplitsDataSource: null,
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
      };
      RASTER_LAYERS.forEach(function (definition) {
        state.rasterLayers[definition.key] = null;
        state.rasterLayerLoadPromises[definition.key] = null;
      });

      window.__depthdifComparisonGlobeState = state;
      updatePageHeader(elements, loaded.config);
      updateVariableControl(state);
      updateDepthControl(state);
      updateVectorControls(state);
      if (elements.errorLegend) {
        elements.errorLegend.hidden = true;
      }

      try {
        const initialRasterKey = selectedRasterKey(state);
        if (initialRasterKey !== "off") {
          await ensureComparisonLayer(state, initialRasterKey);
        }
      } catch (error) {
        console.error(error);
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
      viewer.screenSpaceEventHandler.setInputAction(function (movement) {
        const picked = viewer.scene.pick(movement.position);
        if (!picked || !picked.id) {
          return;
        }
        if (state.pointsDataSource && state.pointsDataSource.entities.contains(picked.id)) {
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
      syncRasterLayerVisibility(state);
      requestRender(state);
      return true;
    } catch (error) {
      if (initToken === getCurrentInitToken()) {
        destroyDepthDifComparisonGlobe();
      }
      console.error(error);
      return false;
    }
  }

  window.initDepthDifComparisonGlobe = initDepthDifComparisonGlobe;
  window.destroyDepthDifComparisonGlobe = destroyDepthDifComparisonGlobe;
})();
