(function () {
  const DEFAULT_GLOBE_CONFIG_URL =
    "https://pub-a0d604187e144d18a52f7c9e679577dc.r2.dev/inference_production/globe/globe-config.json";
  const container = document.getElementById("depthdif-cesium-globe");
  if (!container || typeof window.Cesium === "undefined") {
    return;
  }

  const predictionToggle = document.getElementById("globe-toggle-prediction");
  const groundTruthToggle = document.getElementById("globe-toggle-ground-truth");
  const pointsToggle = document.getElementById("globe-toggle-points");
  const patchSplitsToggle = document.getElementById("globe-toggle-patch-splits");
  const opacitySlider = document.getElementById("globe-overlay-opacity");
  const spinToggle = document.getElementById("globe-toggle-spin");
  const resetButton = document.getElementById("globe-reset-camera");
  const DEFAULT_CAMERA_DESTINATION = {
    lon: -38.56452881619089,
    lat: 34.53988238358822,
    height: 9500000.0,
  };
  const SPIN_RATE_RADIANS_PER_SECOND = Cesium.Math.toRadians(2.5);
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
  const POINT_PIXEL_SIZE = 6;
  const POINT_OUTLINE_WIDTH = 1.5;
  const PATCH_SPLIT_DASH_LENGTH = 18;
  const PATCH_SPLIT_DASH_PATTERN = 255;
  const PATCH_SPLIT_OUTLINE_WIDTH = 2.0;
  const PATCH_SPLIT_COLORS = {
    train: Cesium.Color.fromCssColorString("#1f9d55"),
    val: Cesium.Color.fromCssColorString("#d64545"),
  };

  function resolveConfigUrl() {
    const params = new URLSearchParams(window.location.search);
    const configParam = params.get("config");
    if (configParam && configParam.trim() !== "") {
      return new URL(configParam, window.location.href).toString();
    }
    return DEFAULT_GLOBE_CONFIG_URL;
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

  function stylePointEntities(dataSource, config) {
    const colorScale = resolveColorScale(config);
    const now = Cesium.JulianDate.now();
    dataSource.entities.values.forEach(function (entity) {
      if (!entity.position || !entity.properties || !entity.properties.observed_temp_c) {
        return;
      }

      const observedTempC = Number(entity.properties.observed_temp_c.getValue(now));
      entity.billboard = null;
      entity.label = null;
      entity.point = new Cesium.PointGraphics({
        color: colorForTemperature(observedTempC, colorScale),
        pixelSize: POINT_PIXEL_SIZE,
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: POINT_OUTLINE_WIDTH,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      });
    });
  }

  function splitColor(splitValue) {
    const normalized = String(splitValue || "").trim().toLowerCase();
    return PATCH_SPLIT_COLORS[normalized] || Cesium.Color.WHITE;
  }

  function polygonRingToDegreesArray(hierarchy) {
    if (!hierarchy || !Array.isArray(hierarchy.positions) || hierarchy.positions.length < 2) {
      return null;
    }
    const degrees = [];
    hierarchy.positions.forEach(function (position) {
      const cartographic = Cesium.Cartographic.fromCartesian(position);
      degrees.push(Cesium.Math.toDegrees(cartographic.longitude));
      degrees.push(Cesium.Math.toDegrees(cartographic.latitude));
    });
    return degrees;
  }

  function stylePatchSplitEntities(dataSource) {
    const now = Cesium.JulianDate.now();
    dataSource.entities.values.forEach(function (entity) {
      const splitValue =
        entity.properties && entity.properties.split
          ? entity.properties.split.getValue(now)
          : null;
      const outlineColor = splitColor(splitValue);
      const hierarchy = entity.polygon && entity.polygon.hierarchy
        ? entity.polygon.hierarchy.getValue(now)
        : null;
      const positionsDegrees = polygonRingToDegreesArray(hierarchy);

      entity.billboard = null;
      entity.label = null;
      entity.polygon = null;
      if (!positionsDegrees) {
        return;
      }

      // Replace the GeoJSON polygon fill with a clamped dashed outline so the
      // grid reads as a boundary layer instead of obscuring the temperature tiles.
      entity.polyline = new Cesium.PolylineGraphics({
        positions: Cesium.Cartesian3.fromDegreesArray(positionsDegrees),
        clampToGround: true,
        width: PATCH_SPLIT_OUTLINE_WIDTH,
        material: new Cesium.PolylineDashMaterialProperty({
          color: outlineColor,
          dashLength: PATCH_SPLIT_DASH_LENGTH,
          dashPattern: PATCH_SPLIT_DASH_PATTERN,
        }),
      });
    });
  }

  function buildViewer() {
    const viewer = new Cesium.Viewer("depthdif-cesium-globe", {
      animation: false,
      baseLayer: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      navigationHelpButton: false,
      sceneModePicker: false,
      selectionIndicator: false,
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      timeline: false,
    });
    viewer.useBrowserRecommendedResolution = false;
    viewer.resolutionScale = window.devicePixelRatio || 1;

    viewer.imageryLayers.addImageryProvider(
      new Cesium.OpenStreetMapImageryProvider({
        url: "https://tile.openstreetmap.org/",
        credit: "OpenStreetMap contributors",
      })
    );
    viewer.scene.globe.enableLighting = false;
    return viewer;
  }

  function forceViewerResize(viewer) {
    viewer.resolutionScale = window.devicePixelRatio || 1;
    viewer.resize();
    window.requestAnimationFrame(function () {
      viewer.resize();
    });
  }

  function watchContainerResize(viewer, element) {
    if (typeof window.ResizeObserver === "undefined") {
      return function () {};
    }

    // The square stage height is derived from layout, so watch the actual container
    // rather than only the window. Otherwise Cesium can keep an older, shorter size.
    const observer = new window.ResizeObserver(function () {
      window.requestAnimationFrame(function () {
        forceViewerResize(viewer);
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

  function flyToConfig(viewer, config) {
    const destination = resolveCameraDestination(config);
    if (
      Number.isFinite(destination.lon) &&
      Number.isFinite(destination.lat) &&
      Number.isFinite(destination.height)
    ) {
      viewer.camera.flyTo({
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
      });
      return;
    }
  }

  function setSpinEnabled(state, enabled) {
    state.spinEnabled = enabled;
    state.lastSpinTime = null;
    if (spinToggle) {
      spinToggle.setAttribute("aria-pressed", enabled ? "true" : "false");
      spinToggle.textContent = enabled ? "Stop Spin" : "Spin Globe";
    }
  }

  function attachSpinLoop(state) {
    state.viewer.clock.onTick.addEventListener(function (clock) {
      if (!state.spinEnabled) {
        state.lastSpinTime = null;
        return;
      }

      if (state.lastSpinTime === null) {
        state.lastSpinTime = Cesium.JulianDate.clone(clock.currentTime);
        return;
      }

      const deltaSeconds = Cesium.JulianDate.secondsDifference(
        clock.currentTime,
        state.lastSpinTime
      );
      state.lastSpinTime = Cesium.JulianDate.clone(clock.currentTime);
      if (deltaSeconds <= 0.0) {
        return;
      }

      // Rotate around the Earth's axis to keep the spin centered on the globe
      // rather than panning in screen space.
      state.viewer.scene.camera.rotate(Cesium.Cartesian3.UNIT_Z, -SPIN_RATE_RADIANS_PER_SECOND * deltaSeconds);
    });
  }

  function wireUi(state) {
    predictionToggle.addEventListener("change", function () {
      if (state.predictionLayer) {
        state.predictionLayer.show = predictionToggle.checked;
      }
    });

    groundTruthToggle.addEventListener("change", function () {
      if (state.groundTruthLayer) {
        state.groundTruthLayer.show = groundTruthToggle.checked;
      }
    });

    pointsToggle.addEventListener("change", function () {
      if (state.pointsDataSource) {
        state.pointsDataSource.show = pointsToggle.checked;
      }
    });

    if (patchSplitsToggle) {
      patchSplitsToggle.addEventListener("change", function () {
        if (state.patchSplitsDataSource) {
          state.patchSplitsDataSource.show = patchSplitsToggle.checked;
        }
      });
    }

    opacitySlider.addEventListener("input", function () {
      const alpha = Number(opacitySlider.value);
      if (state.predictionLayer) {
        state.predictionLayer.alpha = alpha;
      }
      if (state.groundTruthLayer) {
        state.groundTruthLayer.alpha = alpha;
      }
    });

    if (spinToggle) {
      spinToggle.addEventListener("click", function () {
        setSpinEnabled(state, !state.spinEnabled);
      });
    }

    resetButton.addEventListener("click", function () {
      if (state.viewer && state.config) {
        flyToConfig(state.viewer, state.config);
      }
    });
  }

  async function addPredictionLayer(viewer, config, configUrl) {
    const predictionUrl = resolveAssetUrl(config.prediction_tiles_url, configUrl);
    if (!predictionUrl) {
      predictionToggle.checked = false;
      predictionToggle.disabled = true;
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(predictionUrl, {
      credit: config.credits && config.credits.prediction,
    });
    const layer = viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = Number(opacitySlider.value);
    layer.show = predictionToggle.checked;
    return layer;
  }

  async function addGroundTruthLayer(viewer, config, configUrl) {
    const groundTruthUrl = resolveAssetUrl(config.ground_truth_tiles_url, configUrl);
    if (!groundTruthUrl) {
      groundTruthToggle.checked = false;
      groundTruthToggle.disabled = true;
      return null;
    }
    const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(groundTruthUrl, {
      credit: config.credits && config.credits.ground_truth,
    });
    const layer = viewer.imageryLayers.addImageryProvider(provider);
    layer.minificationFilter = Cesium.TextureMinificationFilter.NEAREST;
    layer.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST;
    layer.alpha = Number(opacitySlider.value);
    layer.show = groundTruthToggle.checked;
    return layer;
  }

  async function addPointsLayer(viewer, config, configUrl) {
    const pointsUrl = resolveAssetUrl(config.argo_points_url, configUrl);
    if (!pointsUrl) {
      pointsToggle.checked = false;
      pointsToggle.disabled = true;
      return null;
    }
    const dataSource = await Cesium.GeoJsonDataSource.load(pointsUrl, {
      clampToGround: true,
      credit: config.credits && config.credits.points,
    });
    stylePointEntities(dataSource, config);
    viewer.dataSources.add(dataSource);
    dataSource.show = pointsToggle.checked;
    return dataSource;
  }

  async function addPatchSplitsLayer(viewer, config, configUrl) {
    const patchSplitsUrl = resolveAssetUrl(config.patch_splits_url, configUrl);
    if (!patchSplitsUrl) {
      if (patchSplitsToggle) {
        patchSplitsToggle.checked = false;
        patchSplitsToggle.disabled = true;
      }
      return null;
    }
    const dataSource = await Cesium.GeoJsonDataSource.load(patchSplitsUrl, {
      clampToGround: true,
      credit: config.credits && config.credits.patch_splits,
    });
    stylePatchSplitEntities(dataSource);
    viewer.dataSources.add(dataSource);
    dataSource.show = patchSplitsToggle ? patchSplitsToggle.checked : false;
    return dataSource;
  }

  (async function init() {
    try {
      const loaded = await loadConfig();
      const config = loaded.config;
      const configUrl = loaded.configUrl;
      const viewer = buildViewer();
      const state = {
        config: config,
        viewer: viewer,
        predictionLayer: null,
        groundTruthLayer: null,
        pointsDataSource: null,
        patchSplitsDataSource: null,
        spinEnabled: false,
        lastSpinTime: null,
      };

      state.predictionLayer = await addPredictionLayer(viewer, config, configUrl);
      state.groundTruthLayer = await addGroundTruthLayer(viewer, config, configUrl);
      state.pointsDataSource = await addPointsLayer(viewer, config, configUrl);
      state.patchSplitsDataSource = await addPatchSplitsLayer(viewer, config, configUrl);
      attachSpinLoop(state);
      setSpinEnabled(state, false);
      wireUi(state);
      flyToConfig(viewer, config);
      forceViewerResize(viewer);
      const stopWatchingResize = watchContainerResize(viewer, container);
      window.addEventListener("resize", function () {
        forceViewerResize(viewer);
      });
      window.addEventListener("beforeunload", stopWatchingResize, { once: true });
    } catch (error) {
      console.error(error);
    }
  })();
})();
