(function () {
  const DEFAULT_GLOBE_CONFIG_URL =
    "https://pub-a0d604187e144d18a52f7c9e679577dc.r2.dev/global_top_band_20150615/globe/globe-config.json";
  const container = document.getElementById("depthdif-cesium-globe");
  if (!container || typeof window.Cesium === "undefined") {
    return;
  }

  const predictionToggle = document.getElementById("globe-toggle-prediction");
  const groundTruthToggle = document.getElementById("globe-toggle-ground-truth");
  const pointsToggle = document.getElementById("globe-toggle-points");
  const opacitySlider = document.getElementById("globe-overlay-opacity");
  const resetButton = document.getElementById("globe-reset-camera");

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

  function flyToConfig(viewer, config) {
    if (
      Number.isFinite(config.west) &&
      Number.isFinite(config.south) &&
      Number.isFinite(config.east) &&
      Number.isFinite(config.north)
    ) {
      const rectangle = Cesium.Rectangle.fromDegrees(
        config.west,
        config.south,
        config.east,
        config.north
      );
      viewer.camera.flyTo({ destination: rectangle, duration: 1.8 });
      return;
    }

    const destination = config.default_camera_destination || {};
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
        duration: 1.8,
      });
    }
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

    opacitySlider.addEventListener("input", function () {
      const alpha = Number(opacitySlider.value);
      if (state.predictionLayer) {
        state.predictionLayer.alpha = alpha;
      }
      if (state.groundTruthLayer) {
        state.groundTruthLayer.alpha = alpha;
      }
    });

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
      markerColor: Cesium.Color.fromCssColorString("#7cf5ff"),
      markerSize: 10,
      stroke: Cesium.Color.BLACK,
      strokeWidth: 1,
      credit: config.credits && config.credits.points,
    });
    viewer.dataSources.add(dataSource);
    dataSource.show = pointsToggle.checked;
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
      };

      state.predictionLayer = await addPredictionLayer(viewer, config, configUrl);
      state.groundTruthLayer = await addGroundTruthLayer(viewer, config, configUrl);
      state.pointsDataSource = await addPointsLayer(viewer, config, configUrl);
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
