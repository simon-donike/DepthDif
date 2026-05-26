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
    globalDataLoaded: false,
    activeBasin: null,
    activeVariable: null,
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

  function formatDateLabel(value) {
    const raw = String(value || "").trim();
    const match = raw.match(/^(\d{4})(\d{2})(\d{2})$/);
    return match ? `${match[1]}-${match[2]}-${match[3]}` : raw;
  }

  function activeVariableConfig() {
    return (state.config.variables && state.config.variables[state.activeVariable]) || {};
  }

  function unitLabel() {
    return activeVariableConfig().value_unit_label || "";
  }

  function basinLabel(name) {
    if (!name) {
      return "Global";
    }
    const basin = (state.config.basins || []).find((item) => item.name === name);
    return (basin && basin.label) || name;
  }

  function activeBasinPayload() {
    if (!state.activeBasin) {
      return null;
    }
    return state.basinDataCache[state.activeBasin] || null;
  }

  function activeWeeklyErrors() {
    if (!state.activeBasin) {
      return globalWeeklyErrors();
    }
    const payload = activeBasinPayload();
    if (!payload || !payload.variables || !payload.variables[state.activeVariable]) {
      return [];
    }
    return payload.variables[state.activeVariable].weekly_errors || [];
  }

  function globalWeeklyErrors() {
    const rowsByIndex = new Map();
    Object.values(state.basinDataCache).forEach((payload) => {
      const variablePayload = payload && payload.variables && payload.variables[state.activeVariable];
      (variablePayload ? variablePayload.weekly_errors || [] : []).forEach((row) => {
        const index = Number(row.index);
        if (!Number.isFinite(index)) {
          return;
        }
        const current = rowsByIndex.get(index) || {
          index,
          selected_date: row.selected_date,
          target_date: row.target_date,
          iso_year: row.iso_year,
          iso_week: row.iso_week,
          count: 0,
          sum_absolute_error: 0,
        };
        const count = Number(row.count || 0);
        const total = Number(row.sum_absolute_error || 0);
        current.count += Number.isFinite(count) ? count : 0;
        current.sum_absolute_error += Number.isFinite(total) ? total : 0;
        rowsByIndex.set(index, current);
      });
    });
    return Array.from(rowsByIndex.values())
      .sort((left, right) => left.index - right.index)
      .map((row) => ({
        ...row,
        mean_absolute_error: row.count > 0 ? row.sum_absolute_error / row.count : null,
      }));
  }

  function activeDepthErrors() {
    if (!state.activeBasin) {
      return globalDepthErrors();
    }
    const payload = activeBasinPayload();
    if (!payload || !payload.variables || !payload.variables[state.activeVariable]) {
      return [];
    }
    return payload.variables[state.activeVariable].depth_errors || [];
  }

  function globalDepthErrors() {
    const rowsByIndex = new Map();
    Object.values(state.basinDataCache).forEach((payload) => {
      const variablePayload = payload && payload.variables && payload.variables[state.activeVariable];
      (variablePayload ? variablePayload.depth_errors || [] : []).forEach((row) => {
        const index = Number(row.index);
        if (!Number.isFinite(index)) {
          return;
        }
        const current = rowsByIndex.get(index) || {
          index,
          label: row.label,
          requested_depth_m: row.requested_depth_m,
          actual_depth_m: row.actual_depth_m,
          count: 0,
          sum_absolute_error: 0,
        };
        const count = Number(row.count || 0);
        const total = Number(row.sum_absolute_error || 0);
        current.count += Number.isFinite(count) ? count : 0;
        current.sum_absolute_error += Number.isFinite(total) ? total : 0;
        rowsByIndex.set(index, current);
      });
    });
    return Array.from(rowsByIndex.values())
      .sort((left, right) => left.index - right.index)
      .map((row) => ({
        ...row,
        mean_absolute_error: row.count > 0 ? row.sum_absolute_error / row.count : null,
      }));
  }

  function setControlsDisabled(disabled) {
    document
      .querySelectorAll("#temporal-dashboard-select, #temporal-variable-select")
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
        window.location.href = "../spatial-dashboard/";
        return;
      }
      dashboardSelect.value = "temporal";
    });
  }

  function setupControls() {
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
      render();
    });
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
      color: active ? cssVar("--temporal-amber", "#ffd166") : "rgba(124, 200, 255, 0.38)",
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
        layer.on("click", async function (event) {
          if (event.originalEvent) {
            window.L.DomEvent.stop(event.originalEvent);
          }
          state.activeBasin = state.activeBasin === name ? null : name;
          if (state.activeBasin) {
            await loadActiveBasinData();
          } else {
            await loadGlobalBasinData();
          }
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
      font: { color: cssVar("--temporal-text", "#17212b"), family: "Roboto, sans-serif", size: 12 },
      margin: { l: 68, r: 20, t: 18, b: 52 },
      hoverlabel: {
        bgcolor: "#071d2d",
        bordercolor: cssVar("--temporal-border", "rgba(124,200,255,0.32)"),
        font: { color: cssVar("--temporal-text", "#d7e9f7") },
      },
      xaxis: {
        automargin: true,
        color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        tickformat: "%b %d",
        title: { text: "Date", standoff: 10 },
        type: "date",
      },
      yaxis: {
        automargin: true,
        color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        title: { text: `Mean absolute error${unitLabel() ? ` (${unitLabel()})` : ""}`, standoff: 10 },
      },
    };
  }

  function depthProfileLayout() {
    const layout = plotlyLayout();
    layout.margin = { l: 72, r: 20, t: 18, b: 48 };
    layout.xaxis = {
      automargin: true,
      color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
      gridcolor: "rgba(223,244,239,0.1)",
      title: { text: `Mean absolute error${unitLabel() ? ` (${unitLabel()})` : ""}`, standoff: 10 },
      type: "linear",
      zerolinecolor: "rgba(223,244,239,0.18)",
    };
    layout.yaxis = {
      automargin: true,
      autorange: "reversed",
      color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
      gridcolor: "rgba(223,244,239,0.1)",
      title: { text: "Depth (m)", standoff: 10 },
      zerolinecolor: "rgba(223,244,239,0.18)",
    };
    return layout;
  }

  function renderEmptyChart(chartId, layout, message) {
    layout.annotations = [
      {
        font: { color: cssVar("--temporal-muted", "#5f6f7c"), size: 13 },
        showarrow: false,
        text: escapeHtml(message),
        x: 0.5,
        xref: "paper",
        y: 0.5,
        yref: "paper",
      },
    ];
    window.Plotly.react($(chartId), [], layout, PLOTLY_CONFIG);
  }

  function renderEmptyTemporalGraph(message) {
    renderEmptyChart("temporal-depth-error", plotlyLayout(), message);
  }

  function renderDepthErrorGraph() {
    const chart = $("temporal-depth-error");
    const rows = activeWeeklyErrors().filter((row) => Number.isFinite(Number(row.mean_absolute_error)));
    if (rows.length === 0) {
      $("temporal-depth-caption").textContent = `${basinLabel(state.activeBasin)} has no weekly error data`;
      renderEmptyTemporalGraph("No weekly error data for the selected area");
      return;
    }
    const trace = {
      customdata: rows.map((row) => [
        formatDateLabel(row.selected_date),
        row.iso_year && row.iso_week ? `${row.iso_year}-W${String(row.iso_week).padStart(2, "0")}` : "n/a",
        row.count ? row.count.toLocaleString() : "0",
      ]),
      hovertemplate:
        "%{customdata[0]}<br>Mean absolute error: %{y:.3f} " +
        unitLabel() +
        "<br>ISO week: %{customdata[1]}<br>Count: %{customdata[2]}<extra></extra>",
      line: { color: cssVar("--temporal-teal", "#7cc8ff"), width: 3 },
      marker: { color: cssVar("--temporal-amber", "#ffd166"), line: { color: "#071d2d", width: 1 }, size: 7 },
      mode: "lines+markers",
      name: basinLabel(state.activeBasin),
      x: rows.map((row) => formatDateLabel(row.selected_date)),
      y: rows.map((row) => row.mean_absolute_error),
    };
    window.Plotly.react(chart, [trace], plotlyLayout(), PLOTLY_CONFIG);
    $("temporal-depth-caption").textContent = `${basinLabel(state.activeBasin)} ${activeVariableConfig().variable_label || state.activeVariable} mean absolute error by date`;
  }

  function renderDepthProfileGraph() {
    const chart = $("temporal-depth-profile");
    if (!chart) {
      return;
    }
    const rows = activeDepthErrors().filter(
      (row) => Number.isFinite(Number(row.mean_absolute_error)) && Number.isFinite(Number(row.actual_depth_m))
    );
    if (rows.length === 0) {
      $("temporal-profile-caption").textContent = `${basinLabel(state.activeBasin)} has no depth-profile data`;
      renderEmptyChart("temporal-depth-profile", depthProfileLayout(), "No depth-profile data for the selected area");
      return;
    }
    const trace = {
      customdata: rows.map((row) => [
        row.label || `${formatNumber(row.actual_depth_m, 1)} m`,
        row.count ? row.count.toLocaleString() : "0",
      ]),
      hovertemplate:
        "%{customdata[0]}<br>Mean absolute error: %{x:.3f} " +
        unitLabel() +
        "<br>Count: %{customdata[1]}<extra></extra>",
      line: { color: cssVar("--temporal-amber", "#ffd166"), width: 3 },
      marker: { color: cssVar("--temporal-teal", "#7cc8ff"), line: { color: "#071d2d", width: 1 }, size: 6 },
      mode: "lines+markers",
      name: basinLabel(state.activeBasin),
      x: rows.map((row) => row.mean_absolute_error),
      y: rows.map((row) => row.actual_depth_m),
    };
    window.Plotly.react(chart, [trace], depthProfileLayout(), PLOTLY_CONFIG);
    $("temporal-profile-caption").textContent = `${basinLabel(state.activeBasin)} ${activeVariableConfig().variable_label || state.activeVariable} year-average error by depth`;
  }

  function render() {
    setRunLabel();
    const selectedLabel = basinLabel(state.activeBasin);
    $("temporal-selection-pill").textContent = selectedLabel;
    $("temporal-map-caption").textContent = `${selectedLabel} | validation year ${state.config.validation_year}`;
    renderMap();
    renderDepthErrorGraph();
    renderDepthProfileGraph();
  }

  async function loadBasinData(basinName) {
    if (!basinName || state.basinDataCache[basinName]) {
      return;
    }
    const sourceUrl = configUrl();
    const basinUrl = state.config.basin_data_urls[basinName];
    if (!basinUrl) {
      throw new Error(`No basin data URL configured for ${basinName}`);
    }
    state.basinDataCache[basinName] = await fetchJson(new URL(basinUrl, sourceUrl).toString());
  }

  async function loadActiveBasinData() {
    await loadBasinData(state.activeBasin);
  }

  async function loadGlobalBasinData() {
    if (state.globalDataLoaded) {
      return;
    }
    await Promise.all((state.config.basins || []).map((basin) => loadBasinData(basin.name)));
    state.globalDataLoaded = true;
  }

  async function loadDashboard(sourceUrl) {
    state.config = await fetchJson(sourceUrl);
    validateConfig(state.config);
    const params = new URLSearchParams(window.location.search);
    const requestedBasin = params.get("basin");
    const basinNames = (state.config.basins || []).map((basin) => basin.name);
    state.activeBasin = basinNames.includes(requestedBasin) ? requestedBasin : null;
    state.activeVariable =
      (state.config.default_variable && state.config.variables[state.config.default_variable] && state.config.default_variable) ||
      (state.config.available_variables && state.config.available_variables[0]);
    state.basinMap = await fetchJson(new URL(state.config.basin_map_geojson_url, sourceUrl).toString());
    if (state.activeBasin) {
      await loadActiveBasinData();
    } else {
      await loadGlobalBasinData();
    }
  }

  function resizeVisuals() {
    if (state.map) {
      state.map.invalidateSize(false);
    }
    ["temporal-depth-error", "temporal-depth-profile"].forEach((id) => {
      const chart = $(id);
      if (chart && window.Plotly) {
        window.Plotly.Plots.resize(chart);
      }
    });
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
