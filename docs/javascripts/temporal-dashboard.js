(function () {
  const DEFAULT_TEMPORAL_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/temporal/temporal-config.json";
  const MAP_BOUNDS = [
    [-85, -180],
    [85, 180],
  ];
  const MAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png";
  const MAP_TILE_ATTRIBUTION =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';
  const PLOTLY_CONFIG = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  };
  const state = {
    config: null,
    basinMap: null,
    basinDataCache: {},
    activeBasin: null,
    activeVariable: null,
    activeDepth: "all",
    map: null,
    basinLayer: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function configUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get("config") || DEFAULT_TEMPORAL_CONFIG_URL;
  }

  function requireDashboardLibraries() {
    const missing = [];
    if (!window.L) {
      missing.push("Leaflet");
    }
    if (!window.Plotly) {
      missing.push("Plotly");
    }
    if (missing.length > 0) {
      throw new Error(`${missing.join(" and ")} did not load.`);
    }
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
  }

  function formatNumber(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "n/a";
    }
    return number.toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
  }

  function activeVariableConfig() {
    return (state.config.variables && state.config.variables[state.activeVariable]) || {};
  }

  function unitLabel() {
    return activeVariableConfig().value_unit_label || "";
  }

  function basinLabel(name) {
    const basin = (state.config.basins || []).find((item) => item.name === name);
    return (basin && basin.label) || name;
  }

  function activeBasinPayload() {
    return state.basinDataCache[state.activeBasin] || null;
  }

  function activeDepthErrors() {
    const payload = activeBasinPayload();
    if (!payload || !payload.variables || !payload.variables[state.activeVariable]) {
      return [];
    }
    return payload.variables[state.activeVariable].depth_errors || [];
  }

  function depthKey(row, index) {
    return String(row && row.suffix ? row.suffix : index);
  }

  function selectedDepthErrors() {
    const rows = activeDepthErrors();
    if (state.activeDepth === "all") {
      return rows;
    }
    return rows.filter((row, index) => depthKey(row, index) === state.activeDepth);
  }

  function setControlsDisabled(disabled) {
    document
      .querySelectorAll("#temporal-dashboard-select, #temporal-basin-select, #temporal-variable-select, #temporal-depth-select")
      .forEach((element) => {
        element.disabled = Boolean(disabled);
      });
  }

  function renderLoadFailure(error, sourceUrl) {
    document.body.classList.add("temporal-load-failed");
    setControlsDisabled(true);
    $("temporal-run-label").textContent = "Temporal analysis data is unavailable";
    const existing = document.querySelector(".temporal-error-state");
    if (existing) {
      existing.remove();
    }
    const panel = document.createElement("section");
    panel.className = "temporal-error-state";
    panel.innerHTML = [
      "<h2>Could not load the temporal dashboard data</h2>",
      `<p><strong>Manifest URL:</strong> <code>${escapeHtml(sourceUrl)}</code></p>`,
      `<p><strong>Error:</strong> ${escapeHtml(error && error.message ? error.message : error)}</p>`,
    ].join("");
    document.querySelector(".temporal-shell").appendChild(panel);
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`${url} returned HTTP ${response.status}`);
    }
    return response.json();
  }

  function validateConfig(config) {
    if (!config || Number(config.schema_version) !== 2) {
      throw new Error("temporal-config.json is not schema version 2");
    }
    if (!config.basin_data_urls || !config.basin_map_geojson_url) {
      throw new Error("temporal-config.json is missing basin data URLs");
    }
  }

  function setupDashboardSelect() {
    const dashboardSelect = $("temporal-dashboard-select");
    dashboardSelect.value = "temporal";
    dashboardSelect.addEventListener("change", function () {
      if (dashboardSelect.value === "analysis") {
        window.location.href = "../analysis/";
        return;
      }
      dashboardSelect.value = "temporal";
    });
  }

  function setupControls() {
    const basinSelect = $("temporal-basin-select");
    basinSelect.innerHTML = (state.config.basins || [])
      .map((basin) => `<option value="${escapeHtml(basin.name)}">${escapeHtml(basin.label || basin.name)}</option>`)
      .join("");
    basinSelect.value = state.activeBasin;
    basinSelect.addEventListener("change", async function () {
      state.activeBasin = basinSelect.value;
      await loadActiveBasinData();
      populateDepthSelect();
      render();
    });

    const variableSelect = $("temporal-variable-select");
    variableSelect.innerHTML = (state.config.available_variables || [])
      .map((variable) => {
        const variableConfig = state.config.variables[variable] || {};
        return `<option value="${escapeHtml(variable)}">${escapeHtml(variableConfig.variable_label || variable)}</option>`;
      })
      .join("");
    variableSelect.value = state.activeVariable;
    variableSelect.addEventListener("change", function () {
      state.activeVariable = variableSelect.value;
      populateDepthSelect();
      render();
    });

    populateDepthSelect();
    $("temporal-depth-select").addEventListener("change", function () {
      state.activeDepth = this.value;
      render();
    });
  }

  function populateDepthSelect() {
    const depthSelect = $("temporal-depth-select");
    const rows = activeDepthErrors();
    const options = ['<option value="all">All depths</option>'];
    rows.forEach((row, index) => {
      const label = row.label || `Depth ${index + 1}`;
      options.push(`<option value="${escapeHtml(depthKey(row, index))}">${escapeHtml(label)}</option>`);
    });
    depthSelect.innerHTML = options.join("");
    if (state.activeDepth !== "all" && !rows.some((row, index) => depthKey(row, index) === state.activeDepth)) {
      state.activeDepth = "all";
    }
    depthSelect.value = state.activeDepth;
  }

  function setRunLabel() {
    const variableConfig = activeVariableConfig();
    const year = state.config.validation_year || "unknown";
    $("temporal-run-label").textContent = `${variableConfig.variable_label || state.activeVariable} validation year ${year} | ${variableConfig.run_count || 0} weekly exports`;
  }

  function createMap() {
    const map = window.L.map("temporal-map", {
      attributionControl: true,
      maxBounds: MAP_BOUNDS,
      maxBoundsViscosity: 0.65,
      minZoom: 1,
      preferCanvas: true,
      worldCopyJump: false,
      zoomSnap: 0.25,
    });
    window.L.tileLayer(MAP_TILE_URL, {
      attribution: MAP_TILE_ATTRIBUTION,
      maxZoom: 8,
      noWrap: true,
      subdomains: "abcd",
    }).addTo(map);
    map.fitBounds(MAP_BOUNDS, { animate: false, padding: [8, 8] });
    state.map = map;
    return map;
  }

  function basinStyle(feature) {
    const name = feature.properties && feature.properties.name;
    const active = name === state.activeBasin;
    return {
      color: active ? cssVar("--temporal-amber", "#ffd166") : "rgba(124, 200, 255, 0.48)",
      fillColor: active ? cssVar("--temporal-teal", "#7cc8ff") : "rgba(124, 200, 255, 0.12)",
      fillOpacity: active ? 0.48 : 0.18,
      opacity: active ? 0.96 : 0.62,
      weight: active ? 2 : 1,
    };
  }

  function renderMap() {
    const map = state.map || createMap();
    if (state.basinLayer) {
      state.basinLayer.remove();
    }
    state.basinLayer = window.L.geoJSON(state.basinMap, {
      style: basinStyle,
      onEachFeature(feature, layer) {
        const name = feature.properties && feature.properties.name;
        const label = feature.properties && feature.properties.label ? feature.properties.label : name;
        layer.bindTooltip(escapeHtml(label), {
          className: "temporal-leaflet-tooltip",
          direction: "auto",
          sticky: true,
        });
        layer.on("click", async function () {
          state.activeBasin = name;
          $("temporal-basin-select").value = name;
          await loadActiveBasinData();
          populateDepthSelect();
          render();
        });
      },
    }).addTo(map);
    requestAnimationFrame(() => map.invalidateSize(false));
  }

  function plotlyLayout() {
    return {
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: cssVar("--temporal-text", "#d7e9f7"), family: "Roboto, sans-serif", size: 12 },
      margin: { l: 68, r: 20, t: 18, b: 52 },
      hoverlabel: {
        bgcolor: "#061726",
        bordercolor: cssVar("--temporal-border", "rgba(124,200,255,0.18)"),
        font: { color: cssVar("--temporal-text", "#d7e9f7") },
      },
      xaxis: {
        automargin: true,
        color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(124,200,255,0.14)",
        title: { text: `Mean absolute error${unitLabel() ? ` (${unitLabel()})` : ""}`, standoff: 10 },
      },
      yaxis: {
        autorange: "reversed",
        automargin: true,
        color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(124,200,255,0.14)",
        title: { text: "Depth", standoff: 10 },
      },
    };
  }

  function renderDepthErrorGraph() {
    const chart = $("temporal-depth-error");
    const rows = selectedDepthErrors();
    const trace = {
      customdata: rows.map((row) => [row.label, row.count ? row.count.toLocaleString() : "0"]),
      hovertemplate:
        "%{customdata[0]}<br>Mean absolute error: %{x:.3f} " +
        unitLabel() +
        "<br>Count: %{customdata[1]}<extra></extra>",
      marker: { color: cssVar("--temporal-amber", "#ffd166"), line: { color: "#061726", width: 1 }, size: 8 },
      orientation: "h",
      type: "bar",
      name: basinLabel(state.activeBasin),
      x: rows.map((row) => row.mean_absolute_error),
      y: rows.map((row, index) => row.label || `Depth ${index + 1}`),
    };
    window.Plotly.react(chart, [trace], plotlyLayout(), PLOTLY_CONFIG);
    const depthLabel = state.activeDepth === "all" ? "all depths" : rows[0] && rows[0].label ? rows[0].label : "selected depth";
    $("temporal-depth-caption").textContent = `${basinLabel(state.activeBasin)} ${activeVariableConfig().variable_label || state.activeVariable} mean absolute error across the validation year for ${depthLabel}`;
  }

  function render() {
    setRunLabel();
    $("temporal-selection-pill").textContent = basinLabel(state.activeBasin);
    $("temporal-map-caption").textContent = `${basinLabel(state.activeBasin)} | validation year ${state.config.validation_year}`;
    renderMap();
    renderDepthErrorGraph();
  }

  async function loadActiveBasinData() {
    if (state.basinDataCache[state.activeBasin]) {
      return;
    }
    const sourceUrl = configUrl();
    const basinUrl = state.config.basin_data_urls[state.activeBasin];
    if (!basinUrl) {
      throw new Error(`No basin data URL configured for ${state.activeBasin}`);
    }
    state.basinDataCache[state.activeBasin] = await fetchJson(new URL(basinUrl, sourceUrl).toString());
  }

  async function loadDashboard(sourceUrl) {
    state.config = await fetchJson(sourceUrl);
    validateConfig(state.config);
    state.activeBasin = state.config.default_basin || (state.config.basins && state.config.basins[0] && state.config.basins[0].name);
    state.activeVariable =
      (state.config.default_variable && state.config.variables[state.config.default_variable] && state.config.default_variable) ||
      (state.config.available_variables && state.config.available_variables[0]);
    state.basinMap = await fetchJson(new URL(state.config.basin_map_geojson_url, sourceUrl).toString());
    await loadActiveBasinData();
  }

  function resizeVisuals() {
    if (state.map) {
      state.map.invalidateSize(false);
    }
    const chart = $("temporal-depth-error");
    if (chart && window.Plotly) {
      window.Plotly.Plots.resize(chart);
    }
  }

  async function init() {
    const sourceUrl = configUrl();
    setupDashboardSelect();
    try {
      requireDashboardLibraries();
      await loadDashboard(sourceUrl);
      setupControls();
      setControlsDisabled(false);
      window.addEventListener("resize", resizeVisuals);
      render();
    } catch (error) {
      console.error(error);
      renderLoadFailure(error, sourceUrl);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
