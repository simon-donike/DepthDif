(function () {
  const DEFAULT_GLOBE_CONFIG_URL =
    "https://pub-a0d604187e144d18a52f7c9e679577dc.r2.dev/inference_production/globe/globe-config.json";
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
  const DEFAULT_COLOR_SCALE = { min: 0.0, max: 30.0 };
  const PATCH_SPLIT_ALPHA = 0.5;
  const PROFILE_POPUP_CLOSE_DELAY_MS = 180;
  const BACKGROUND_PRELOAD_DELAY_MS = 180;
  const PATCH_SPLIT_COLORS = {
    train: "#1f9d55",
    val: "#d64545",
  };
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

    return {
      container: container,
      predictionToggle: document.getElementById("globe-toggle-prediction"),
      groundTruthToggle: document.getElementById("globe-toggle-ground-truth"),
      pointsToggle: document.getElementById("globe-toggle-points"),
      fullSampleToggle: document.getElementById("globe-toggle-full-samples"),
      patchSplitsToggle: document.getElementById("globe-toggle-patch-splits"),
      toolbar: document.querySelector(".globe-toolbar"),
      toolbarContent: document.getElementById("globe-toolbar-content"),
      toolbarToggle: document.getElementById("globe-toggle-toolbar"),
      depthSlider: document.getElementById("globe-depth-level"),
      depthLabel: document.getElementById("globe-depth-level-label"),
      spinToggle: document.getElementById("globe-toggle-spin"),
      resetButton: document.getElementById("globe-reset-camera"),
      pageTitle: document.getElementById("globe-page-title"),
      pageDescription: document.getElementById("globe-page-description"),
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

  function resolveConfigUrl() {
    const params = new URLSearchParams(window.location.search);
    const configParam = params.get("config");
    if (configParam && configParam.trim() !== "") {
      return new URL(configParam, window.location.href).toString();
    }
    return DEFAULT_GLOBE_CONFIG_URL;
  }

  function resolveSelectedDateParts(selectedDate) {
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

    const isoDate = new Date(date.getTime());
    const dayOfWeek = isoDate.getUTCDay() || 7;
    isoDate.setUTCDate(isoDate.getUTCDate() + 4 - dayOfWeek);
    const isoYear = isoDate.getUTCFullYear();
    const yearStart = new Date(Date.UTC(isoYear, 0, 1));
    const isoWeek = Math.ceil((((isoDate - yearStart) / 86400000) + 1) / 7);
    return { isoYear, isoWeek };
  }

  function updatePageHeader(elements, config) {
    if (!elements.pageTitle && !elements.pageDescription) {
      return;
    }

    const selectedDateParts = resolveSelectedDateParts(config.selected_date);
    if (elements.pageTitle) {
      elements.pageTitle.textContent = "DepthDif";
    }

    if (elements.pageDescription) {
      if (selectedDateParts) {
        elements.pageDescription.textContent =
          "Densifies deep sea measurements using diffusion. This 3D globe represents a weekly aggregate from week " +
          String(selectedDateParts.isoWeek) +
          " of year 2015.";
      } else {
        elements.pageDescription.textContent =
          "Densifies deep sea measurements using diffusion.";
      }
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

  function markToggleUnavailable(toggle) {
    if (!toggle) {
      return;
    }
    toggle.checked = false;
    toggle.disabled = true;
    toggle.dataset.globeUnavailable = "true";
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
    const minValue = Number(config.color_scale_min_c);
    const maxValue = Number(config.color_scale_max_c);
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
      },
    ];
  }

  function selectedDepthLevel(state) {
    const depthLevels = getDepthLevels(state.config);
    const index = clamp(Number(state.selectedDepthIndex || 0), 0, depthLevels.length - 1);
    return depthLevels[index] || depthLevels[0];
  }

  function updateDepthControl(state) {
    const depthLevels = getDepthLevels(state.config);
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
      const depthLevel = selectedDepthLevel(state);
      state.elements.depthLabel.textContent = String(depthLevel.label || "Surface");
    }
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

  function splitColor(splitValue) {
    const normalized = String(splitValue || "").trim().toLowerCase();
    return Cesium.Color.fromCssColorString(PATCH_SPLIT_COLORS[normalized] || "#ffffff");
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

  function stylePointEntities(dataSource, options) {
    dataSource.entities.values.forEach(function (entity) {
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
    });
  }

  function stylePatchSplitEntities(dataSource) {
    const now = Cesium.JulianDate.now();
    dataSource.entities.values.forEach(function (entity) {
      const splitValue =
        entity.properties && entity.properties.split
          ? entity.properties.split.getValue(now)
          : null;
      const fillColor = splitColor(splitValue).withAlpha(PATCH_SPLIT_ALPHA);

      entity.billboard = null;
      entity.label = null;
      if (!entity.polygon) {
        return;
      }

      // Keep the split overlay simple and stable: solid train/val squares with fixed alpha.
      entity.polygon.material = fillColor;
      entity.polygon.outline = false;
      entity.polyline = null;
    });
  }

  function clearProfilePopupCloseTimer(state) {
    if (state.profilePopupCloseTimer !== null) {
      window.clearTimeout(state.profilePopupCloseTimer);
      state.profilePopupCloseTimer = null;
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
    const locationId = properties.location_id ? properties.location_id.getValue(now) : "Full sample";
    const patchId = properties.patch_id ? properties.patch_id.getValue(now) : "";
    const pixelRow = properties.pixel_row ? properties.pixel_row.getValue(now) : null;
    const pixelCol = properties.pixel_col ? properties.pixel_col.getValue(now) : null;

    if (elements.profilePopupTitle) {
      elements.profilePopupTitle.textContent = String(locationId);
    }
    if (elements.profilePopupSubtitle) {
      elements.profilePopupSubtitle.textContent =
        "Patch " +
        String(patchId || "") +
        ", pixel (" +
        String(pixelRow) +
        ", " +
        String(pixelCol) +
        ")";
    }
    elements.profilePopupImage.src = new URL(String(graphPath), state.configUrl).toString();
    elements.profilePopupImage.alt = String(locationId) + " profile comparison";
    clearProfilePopupCloseTimer(state);
    elements.profilePopup.hidden = false;
    elements.profilePopup.classList.remove("is-closing");
    window.requestAnimationFrame(function () {
      elements.profilePopup.classList.add("is-open");
      requestRender(state);
    });
  }

  function enforceOverlayOrder(state) {
    const imageryLayers = state.viewer.imageryLayers;
    if (state.groundTruthLayer) {
      // Keep GLORYS above prediction without moving prediction below the basemap.
      imageryLayers.raise(state.groundTruthLayer);
    }

    const dataSources = state.viewer.dataSources;
    if (state.patchSplitsDataSource) {
      dataSources.lowerToBottom(state.patchSplitsDataSource);
    }
    if (state.pointsDataSource) {
      dataSources.raiseToTop(state.pointsDataSource);
    }
    if (state.fullSampleDataSource) {
      dataSources.raiseToTop(state.fullSampleDataSource);
    }
  }

  function addBaseMap(viewer) {
    const naturalEarthUrl = Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII");
    // Prefer Cesium's bundled Natural Earth relief tiles for a lighter free basemap.
    Cesium.TileMapServiceImageryProvider.fromUrl(naturalEarthUrl, {
      credit: "Natural Earth II",
    })
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
      });
  }

  function buildViewer(container) {
    const viewer = new Cesium.Viewer(container, {
      animation: false,
      baseLayer: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
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
    addBaseMap(viewer);
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
    const depthLevel = selectedDepthLevel(state);
    const predictionUrl = resolveAssetUrl(depthLevel.prediction_tiles_url, state.configUrl);
    if (!predictionUrl) {
      markToggleUnavailable(state.elements.predictionToggle);
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(predictionUrl, {
      credit: state.config.credits && state.config.credits.prediction,
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
    const depthLevel = selectedDepthLevel(state);
    const groundTruthUrl = resolveAssetUrl(depthLevel.ground_truth_tiles_url, state.configUrl);
    if (!groundTruthUrl) {
      markToggleUnavailable(state.elements.groundTruthToggle);
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(groundTruthUrl, {
      credit: state.config.credits && state.config.credits.ground_truth,
    });
    const layer = state.viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = 1.0;
    layer.show = state.elements.groundTruthToggle.checked;
    return layer;
  }

  async function addPointsLayer(state) {
    const pointsUrl = resolveAssetUrl(state.config.argo_points_url, state.configUrl);
    if (!pointsUrl) {
      markToggleUnavailable(state.elements.pointsToggle);
      return null;
    }
    const dataSource = await Cesium.GeoJsonDataSource.load(pointsUrl, {
      clampToGround: true,
      markerColor: Cesium.Color.fromCssColorString("#7cf5ff"),
      markerSize: 10,
      stroke: Cesium.Color.BLACK,
      strokeWidth: 1,
      credit: state.config.credits && state.config.credits.points,
    });
    stylePointEntities(dataSource, {
      image: ARGO_POINT_MARKER_IMAGE,
      width: 24,
      height: 24,
      pixelOffsetY: 0,
    });
    state.viewer.dataSources.add(dataSource);
    dataSource.show = state.elements.pointsToggle.checked;
    return dataSource;
  }

  async function addFullSampleLayer(state) {
    const fullSampleUrl = resolveAssetUrl(state.config.full_sample_points_url, state.configUrl);
    if (!fullSampleUrl) {
      markToggleUnavailable(state.elements.fullSampleToggle);
      return null;
    }
    const dataSource = await Cesium.GeoJsonDataSource.load(fullSampleUrl, {
      clampToGround: true,
      markerColor: Cesium.Color.fromCssColorString("#ffd166"),
      markerSize: 21, // Bump the full-sample profile markers to ~1.5x their previous size.
      stroke: Cesium.Color.BLACK,
      strokeWidth: 1,
      credit: state.config.credits && state.config.credits.full_sample_points,
    });
    stylePointEntities(dataSource, {
      image: FULL_SAMPLE_MARKER_IMAGE,
      width: 34,
      height: 34,
      pixelOffsetY: -2,
    });
    state.viewer.dataSources.add(dataSource);
    dataSource.show = state.elements.fullSampleToggle.checked;
    return dataSource;
  }

  async function addPatchSplitsLayer(state) {
    const patchSplitsUrl = resolveAssetUrl(state.config.patch_splits_url, state.configUrl);
    if (!patchSplitsUrl) {
      markToggleUnavailable(state.elements.patchSplitsToggle);
      return null;
    }
    const dataSource = await Cesium.GeoJsonDataSource.load(patchSplitsUrl, {
      clampToGround: true,
      credit: state.config.credits && state.config.credits.patch_splits,
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
    return loadOptionalLayer(
      state,
      state.elements.groundTruthToggle,
      "groundTruthLayer",
      "groundTruthLayerLoadPromise",
      addGroundTruthLayer
    );
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

  async function ensureFullSampleLayer(state) {
    return loadOptionalLayer(
      state,
      state.elements.fullSampleToggle,
      "fullSampleDataSource",
      "fullSampleDataSourceLoadPromise",
      addFullSampleLayer
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
        ensurePointsLayer,
        ensureFullSampleLayer,
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

  async function handleOptionalLayerToggle(state, toggle, stateKey, ensureLayer) {
    if (!toggle) {
      return;
    }

    if (!toggle.checked) {
      if (state[stateKey]) {
        state[stateKey].show = false;
      }
      requestRender(state);
      return;
    }

    try {
      const layerOrSource = await ensureLayer(state);
      if (layerOrSource) {
        layerOrSource.show = toggle.checked;
        requestRender(state);
      }
    } catch (error) {
      toggle.checked = false;
      setToggleLoading(toggle, false);
      console.error(error);
    }
  }

  async function reloadRasterDepthLayers(state) {
    const imageryLayers = state.viewer.imageryLayers;
    const showPrediction = state.elements.predictionToggle.checked;
    const showGroundTruth = state.elements.groundTruthToggle.checked;

    if (state.predictionLayer) {
      imageryLayers.remove(state.predictionLayer, true);
      state.predictionLayer = null;
    }
    if (state.groundTruthLayer) {
      imageryLayers.remove(state.groundTruthLayer, true);
      state.groundTruthLayer = null;
    }
    state.predictionLayerLoadPromise = null;
    state.groundTruthLayerLoadPromise = null;

    try {
      state.predictionLayer = await addPredictionLayer(state);
      if (state.predictionLayer) {
        state.predictionLayer.show = showPrediction;
      }
      if (showGroundTruth) {
        state.groundTruthLayer = await addGroundTruthLayer(state);
        if (state.groundTruthLayer) {
          state.groundTruthLayer.show = true;
        }
      }
      enforceOverlayOrder(state);
      requestRender(state);
    } catch (error) {
      console.error(error);
    }
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

    elements.predictionToggle.addEventListener("change", function () {
      if (state.predictionLayer) {
        state.predictionLayer.show = elements.predictionToggle.checked;
      }
      requestRender(state);
    });

    elements.groundTruthToggle.addEventListener("change", function () {
      handleOptionalLayerToggle(
        state,
        elements.groundTruthToggle,
        "groundTruthLayer",
        ensureGroundTruthLayer
      );
    });

    elements.pointsToggle.addEventListener("change", function () {
      handleOptionalLayerToggle(
        state,
        elements.pointsToggle,
        "pointsDataSource",
        ensurePointsLayer
      );
    });

    if (elements.fullSampleToggle) {
      elements.fullSampleToggle.addEventListener("change", function () {
        handleOptionalLayerToggle(
          state,
          elements.fullSampleToggle,
          "fullSampleDataSource",
          ensureFullSampleLayer
        );
        if (!elements.fullSampleToggle.checked) {
          closeProfilePopup(state);
        }
      });
    }

    if (elements.patchSplitsToggle) {
      elements.patchSplitsToggle.addEventListener("change", function () {
        handleOptionalLayerToggle(
          state,
          elements.patchSplitsToggle,
          "patchSplitsDataSource",
          ensurePatchSplitsLayer
        );
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

    elements.resetButton.addEventListener("click", function () {
      if (state.viewer && state.config) {
        flyToConfig(state);
      }
    });

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

      const viewer = buildViewer(elements.container);
      const state = {
        config: loaded.config,
        configUrl: loaded.configUrl,
        viewer: viewer,
        elements: elements,
        predictionLayer: null,
        groundTruthLayer: null,
        pointsDataSource: null,
        fullSampleDataSource: null,
        patchSplitsDataSource: null,
        predictionLayerLoadPromise: null,
        groundTruthLayerLoadPromise: null,
        pointsDataSourceLoadPromise: null,
        fullSampleDataSourceLoadPromise: null,
        patchSplitsDataSourceLoadPromise: null,
        selectedDepthIndex: 0,
        spinEnabled: false,
        lastSpinTime: null,
        profilePopupCloseTimer: null,
        stopWatchingResize: null,
        handleWindowResize: null,
        spinTickListener: null,
        preloadTaskId: null,
      };
      window.__depthdifCesiumGlobeState = state;
      updatePageHeader(elements, loaded.config);
      updateDepthControl(state);

      state.predictionLayer = await addPredictionLayer(state);
      if (initToken !== getCurrentInitToken()) {
        cleanupState(state);
        return false;
      }

      enforceOverlayOrder(state);
      attachSpinLoop(state);
      setSpinEnabled(state, false);
      wireUi(state);
      setToolbarCollapsed(elements, false);
      viewer.screenSpaceEventHandler.setInputAction(function (movement) {
        const picked = viewer.scene.pick(movement.position);
        if (
          !picked ||
          !picked.id ||
          !state.fullSampleDataSource ||
          !state.fullSampleDataSource.entities.contains(picked.id)
        ) {
          return;
        }
        showProfilePopup(state, picked.id);
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
