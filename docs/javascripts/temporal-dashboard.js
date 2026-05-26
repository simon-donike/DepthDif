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
  const TEMPORAL_FIELD_KEY = "change_error";
  const state = {
    config: null,
    datasets: {},
    gridGeometries: {},
    variables: [],
    activeArea: null,
    activeVariable: null,
    activeDepth: "all_depths",
    map: null,
    areaLayer: null,
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

  function formatPixelCount(value) {
    return formatNumber(value, 0);
  }

  function data() {
    return state.datasets[state.activeVariable] || null;
  }

  function activeGridGeometryIndex() {
    return state.gridGeometries[state.activeVariable] || null;
  }

  function indexAnalysisGridGeoJson(payload) {
    if (!payload || !Array.isArray(payload.features)) {
      return null;
    }
    const featuresById = {};
    for (const feature of payload.features) {
      const properties = feature && feature.properties;
      const geometry = feature && feature.geometry;
      const id = properties && properties.id;
      if (!id || !geometry) {
        continue;
      }
      featuresById[String(id)] = {
        type: "Feature",
        geometry,
        properties,
      };
    }
    return featuresById;
  }

  function activeVariableConfig() {
    const payload = data();
    return (
      (payload && payload.variable) ||
      (state.config && state.config.variables && state.config.variables[state.activeVariable]) ||
      {}
    );
  }

  function unitLabel() {
    return activeVariableConfig().value_unit_label || "";
  }

  function formatDateLabel(value) {
    const raw = String(value || "").trim();
    if (/^\d{8}$/.test(raw)) {
      return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
    }
    return raw || "unknown date";
  }

  function formatIsoWeek(year, week) {
    if (!year || !week) {
      return "";
    }
    return `ISO ${year}-W${String(week).padStart(2, "0")}`;
  }

  function activeDepthLevels() {
    return data() && Array.isArray(data().depth_levels) ? data().depth_levels : [];
  }

  function depthKey(row, index) {
    return String(row && row.suffix ? row.suffix : index);
  }

  function defaultDepthKey() {
    const levels = activeDepthLevels();
    const aggregate = levels.find((level) => level.is_aggregate);
    return depthKey(aggregate || levels[0], 0);
  }

  function activeDepthLevel() {
    const levels = activeDepthLevels();
    return levels.find((level, index) => depthKey(level, index) === state.activeDepth) || levels[0] || null;
  }

  function activeTemporalField() {
    const depth = activeDepthLevel();
    return depth && depth.fields ? depth.fields[TEMPORAL_FIELD_KEY] : null;
  }

  function selectedDepthErrors() {
    const field = activeTemporalField();
    if (!field || !Array.isArray(field.periods)) {
      return [];
    }
    return field.periods;
  }

  function temporalSourcesFromConfig(config) {
    if (config.depth_levels) {
      return [];
    }
    const variables = config.variables && typeof config.variables === "object" ? config.variables : null;
    if (variables) {
      return Object.entries(variables)
        .map(([key, variableConfig]) => ({
          key,
          label: variableConfig.variable_label || variableConfig.variable || key,
          url: variableConfig.temporal_analysis_data_url,
          gridUrl: variableConfig.analysis_grid_geojson_url,
        }))
        .filter((item) => item.url);
    }
    if (config.temporal_analysis_data_url) {
      return [
        {
          key: config.variable || "default",
          label: config.variable_label || config.variable || "Default",
          url: config.temporal_analysis_data_url,
          gridUrl: config.analysis_grid_geojson_url,
        },
      ];
    }
    return [];
  }

  function validateTemporalPayload(payload) {
    if (!payload || typeof payload !== "object") {
      throw new Error("temporal analysis payload is empty or invalid");
    }
    if (payload.schema_version === undefined) {
      throw new Error("temporal analysis payload is missing schema_version");
    }
    if (!Array.isArray(payload.depth_levels) || payload.depth_levels.length === 0) {
      throw new Error("temporal analysis payload has no depth levels");
    }
  }

  function validateConfig(config) {
    if (!config || typeof config !== "object" || config.schema_version === undefined) {
      throw new Error("temporal-config.json is empty or missing schema_version");
    }
    if (config.depth_levels) {
      validateTemporalPayload(config);
      return;
    }
    if (temporalSourcesFromConfig(config).length === 0) {
      throw new Error("temporal-config.json does not list any temporal analysis datasets");
    }
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

  async function fetchOptionalJson(url) {
    if (!url) {
      return null;
    }
    try {
      return await fetchJson(url);
    } catch (error) {
      console.warn(`Optional temporal dashboard asset failed to load: ${url}`, error);
      return null;
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
    const areaSelect = $("temporal-basin-select");
    areaSelect.addEventListener("change", function () {
      state.activeArea = areaSelect.value || null;
      render();
    });

    const variableSelect = $("temporal-variable-select");
    variableSelect.innerHTML = state.variables
      .map((variable) => `<option value="${escapeHtml(variable.key)}">${escapeHtml(variable.label)}</option>`)
      .join("");
    variableSelect.value = state.activeVariable;
    variableSelect.addEventListener("change", function () {
      state.activeVariable = variableSelect.value;
      state.activeDepth = defaultDepthKey();
      state.activeArea = null;
      populateDepthSelect();
      populateAreaSelect();
      render();
    });

    const depthSelect = $("temporal-depth-select");
    depthSelect.addEventListener("change", function () {
      state.activeDepth = depthSelect.value;
      populateAreaSelect();
      render();
    });

    populateDepthSelect();
    populateAreaSelect();
  }

  function populateDepthSelect() {
    const depthSelect = $("temporal-depth-select");
    const levels = activeDepthLevels();
    depthSelect.innerHTML = levels
      .map((row, index) => `<option value="${escapeHtml(depthKey(row, index))}">${escapeHtml(row.label || `Depth ${index + 1}`)}</option>`)
      .join("");
    if (!levels.some((row, index) => depthKey(row, index) === state.activeDepth)) {
      state.activeDepth = defaultDepthKey();
    }
    depthSelect.value = state.activeDepth;
  }

  function populateAreaSelect() {
    const areaSelect = $("temporal-basin-select");
    const cells = mapAreaRows();
    areaSelect.innerHTML = [
      '<option value="">No area selected</option>',
      ...cells.map((cell) => `<option value="${escapeHtml(cell.id)}">${escapeHtml(cell.label || cell.id)}</option>`),
    ].join("");
    if (state.activeArea && !cells.some((cell) => cell.id === state.activeArea)) {
      state.activeArea = null;
    }
    areaSelect.value = state.activeArea || "";
  }

  function syncAreaSelect() {
    const areaSelect = $("temporal-basin-select");
    if (areaSelect) {
      areaSelect.value = state.activeArea || "";
    }
  }

  function setRunLabel() {
    const variableConfig = activeVariableConfig();
    const run = (data() && data().run) || {};
    const dateRange = `${formatDateLabel(run.start_date)} to ${formatDateLabel(run.end_date)}`;
    $("temporal-run-label").textContent = `${variableConfig.label || variableConfig.variable_label || state.activeVariable} temporal consistency | ${dateRange} | ${run.run_count || 0} weekly exports`;
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
    state.areaLayer = window.L.layerGroup().addTo(map);
    state.map = map;
    return map;
  }

  function cellBounds(cell) {
    return [
      [Number(cell.south), Number(cell.west)],
      [Number(cell.north), Number(cell.east)],
    ];
  }

  function mapAreaRows() {
    const cellsById = {};
    for (const period of selectedDepthErrors()) {
      for (const cell of period.grid_cells || []) {
        const id = String(cell.id);
        if (!cellsById[id]) {
          cellsById[id] = {
            ...cell,
            id,
            weightedError: 0,
            weight: 0,
            count: 0,
          };
        }
        const value = Number(cell.mean);
        if (!Number.isFinite(value)) {
          continue;
        }
        // Weight week-level cell means by pixel count so sparse cells do not dominate the map color.
        const count = Number(cell.count);
        const weight = Number.isFinite(count) && count > 0 ? count : 1;
        cellsById[id].weightedError += value * weight;
        cellsById[id].weight += weight;
        cellsById[id].count += Number.isFinite(count) ? count : 0;
      }
    }
    return Object.values(cellsById)
      .map((cell) => ({
        ...cell,
        mean: cell.weight > 0 ? cell.weightedError / cell.weight : null,
      }))
      .sort((left, right) => Number(left.south) - Number(right.south) || Number(left.west) - Number(right.west));
  }

  function selectedAreaRow() {
    if (!state.activeArea) {
      return null;
    }
    return mapAreaRows().find((cell) => cell.id === state.activeArea) || null;
  }

  function activeAreaLabel() {
    const cell = selectedAreaRow();
    return cell ? cell.label || cell.id : "No area selected";
  }

  function chartAreaLabel() {
    return selectedAreaRow() ? activeAreaLabel() : "Global";
  }

  function mapColorDomain(cells) {
    const values = cells.map((cell) => Number(cell.mean)).filter(Number.isFinite);
    if (values.length === 0) {
      return [0, 1];
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    return min === max ? [min, min + 1] : [min, max];
  }

  function normalizedMapValue(value, domain) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return 0;
    }
    return Math.max(0, Math.min(1, (number - domain[0]) / (domain[1] - domain[0])));
  }

  function areaStyle(cell, domain) {
    const active = state.activeArea === cell.id;
    const intensity = normalizedMapValue(cell.mean, domain);
    return {
      color: active ? cssVar("--temporal-amber", "#ffd166") : "rgba(124, 200, 255, 0.48)",
      fillColor: active ? cssVar("--temporal-amber", "#ffd166") : cssVar("--temporal-teal", "#7cc8ff"),
      fillOpacity: active ? 0.72 : 0.14 + intensity * 0.5,
      opacity: active ? 1 : 0.58,
      weight: active ? 2 : 0.7,
    };
  }

  function areaTooltipHtml(cell) {
    const lines = [`<strong>${escapeHtml(cell.label || cell.id)}</strong>`];
    if (cell.basin) {
      lines.push(`Basin: ${escapeHtml(cell.basin)}`);
    }
    lines.push(`Average error: ${escapeHtml(formatNumber(cell.mean, 3))} ${escapeHtml(unitLabel())}`);
    lines.push(`Count: ${escapeHtml(formatPixelCount(cell.count))}`);
    return lines.join("<br>");
  }

  function mapAreaLayer(cell, colorDomain, geometryIndex) {
    const geometryFeature = geometryIndex ? geometryIndex[cell.id] : null;
    if (geometryIndex && !geometryFeature) {
      return null;
    }
    if (geometryFeature) {
      return window.L.geoJSON(geometryFeature, {
        style: areaStyle(cell, colorDomain),
      });
    }
    return window.L.rectangle(cellBounds(cell), areaStyle(cell, colorDomain));
  }

  function renderMap() {
    const map = state.map || createMap();
    const cells = mapAreaRows();
    const colorDomain = mapColorDomain(cells);
    const geometryIndex = activeGridGeometryIndex();
    state.areaLayer.clearLayers();
    for (const cell of cells) {
      const layer = mapAreaLayer(cell, colorDomain, geometryIndex);
      if (!layer) {
        continue;
      }
      layer.bindTooltip(areaTooltipHtml(cell), {
        className: "temporal-leaflet-tooltip",
        direction: "auto",
        sticky: true,
      });
      layer.on("click", function () {
        const active = state.activeArea === cell.id;
        state.activeArea = active ? null : cell.id;
        syncAreaSelect();
        render();
      });
      state.areaLayer.addLayer(layer);
    }
    requestAnimationFrame(() => map.invalidateSize(false));
  }

  function plotlyLayout() {
    const unit = unitLabel();
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
        title: { text: "Date (weekly export)", standoff: 10 },
        type: "date",
      },
      yaxis: {
        automargin: true,
        color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(124,200,255,0.14)",
        title: { text: `Average error${unit ? ` (${unit})` : ""}`, standoff: 10 },
      },
    };
  }

  function periodDate(period) {
    return formatDateLabel(period.center_date || period.end_date || period.start_date || period.label);
  }

  function periodWeek(period) {
    return (
      formatIsoWeek(period.center_iso_year || period.end_iso_year || period.start_iso_year, period.center_iso_week || period.end_iso_week || period.start_iso_week) ||
      "Week unavailable"
    );
  }

  function periodStats(period) {
    if (!state.activeArea) {
      return period.global || null;
    }
    return (period.grid_cells || []).find((cell) => String(cell.id) === state.activeArea) || null;
  }

  function renderDepthErrorGraph() {
    const chart = $("temporal-depth-error");
    const rows = selectedDepthErrors()
      .map((period) => ({
        period,
        stats: periodStats(period),
      }))
      .filter((row) => Number.isFinite(Number(row.stats && row.stats.mean)));
    const trace = {
      customdata: rows.map((row) => [
        row.period.label || periodDate(row.period),
        periodWeek(row.period),
        row.stats.count ? row.stats.count.toLocaleString() : "0",
        chartAreaLabel(),
      ]),
      hovertemplate:
        "%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[3]}<br>Average error: %{y:.3f} " +
        unitLabel() +
        "<br>Count: %{customdata[2]}<extra></extra>",
      line: { color: cssVar("--temporal-amber", "#ffd166"), width: 2 },
      marker: { color: cssVar("--temporal-teal", "#7cc8ff"), line: { color: "#061726", width: 1 }, size: 8 },
      mode: "lines+markers",
      name: chartAreaLabel(),
      type: "scatter",
      x: rows.map((row) => periodDate(row.period)),
      y: rows.map((row) => row.stats.mean),
    };
    window.Plotly.react(chart, [trace], plotlyLayout(), PLOTLY_CONFIG);
    const depth = activeDepthLevel();
    const depthLabel = depth && depth.label ? depth.label : "selected depth";
    const variableLabel = activeVariableConfig().label || activeVariableConfig().variable_label || state.activeVariable;
    $("temporal-depth-caption").textContent = `${chartAreaLabel()} ${variableLabel} average change error by week for ${depthLabel}`;
  }

  function render() {
    setRunLabel();
    syncAreaSelect();
    $("temporal-selection-pill").textContent = activeAreaLabel();
    $("temporal-map-caption").textContent = `${activeAreaLabel()} | click an area to select or deselect it`;
    renderMap();
    renderDepthErrorGraph();
  }

  async function loadDashboard(sourceUrl) {
    state.config = await fetchJson(sourceUrl);
    validateConfig(state.config);
    if (state.config.depth_levels) {
      const key = (state.config.variable && state.config.variable.name) || state.config.variable || "default";
      const label = (state.config.variable && state.config.variable.label) || key;
      state.variables = [{ key, label }];
      state.datasets = { [key]: state.config };
      state.gridGeometries = {};
      state.activeVariable = key;
      state.activeDepth = defaultDepthKey();
      state.activeArea = null;
      return;
    }

    const sources = temporalSourcesFromConfig(state.config);
    const loaded = await Promise.all(
      sources.map(async (source) => {
        const payloadUrl = new URL(source.url, sourceUrl).toString();
        const gridUrl = source.gridUrl ? new URL(source.gridUrl, sourceUrl).toString() : null;
        const [payload, gridPayload] = await Promise.all([fetchJson(payloadUrl), fetchOptionalJson(gridUrl)]);
        validateTemporalPayload(payload);
        return {
          key: source.key,
          label: (payload.variable && payload.variable.label) || source.label,
          payload,
          gridGeometryIndex: indexAnalysisGridGeoJson(gridPayload),
        };
      })
    );
    state.variables = loaded.map((item) => ({ key: item.key, label: item.label }));
    state.datasets = Object.fromEntries(loaded.map((item) => [item.key, item.payload]));
    state.gridGeometries = Object.fromEntries(
      loaded.filter((item) => item.gridGeometryIndex).map((item) => [item.key, item.gridGeometryIndex])
    );
    state.activeVariable =
      (state.config.default_variable && state.datasets[state.config.default_variable] && state.config.default_variable) || loaded[0].key;
    state.activeDepth = defaultDepthKey();
    state.activeArea = null;
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
    const sourceUrl = new URL(configUrl(), window.location.href).toString();
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
