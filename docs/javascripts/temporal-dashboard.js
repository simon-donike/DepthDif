(function () {
  const DEFAULT_TEMPORAL_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/temporal/temporal-config.json";
  const METRIC_LABELS = { median: "Median", mean: "Mean", p90: "P90", p95: "P95" };
  const BASIN_LABELS = {
    Pacific: "Pacific Ocean",
    Atlantic: "Atlantic Ocean",
    Indian: "Indian Ocean",
    Southern: "Southern Ocean",
    Arctic: "Arctic Ocean",
    Other: "Other Waters",
  };
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
    datasets: {},
    gridGeometries: {},
    variables: [],
    activeVariable: null,
    field: "change_error",
    periodIndex: 0,
    depthIndex: 0,
    metric: "median",
    focus: { type: "global", id: "global", label: "Global" },
    map: null,
    mapCellLayer: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
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

  function data() {
    return state.datasets[state.activeVariable];
  }

  function activeGridGeometryIndex() {
    return state.gridGeometries[state.activeVariable] || null;
  }

  function validatePayload(payload) {
    if (!payload || typeof payload !== "object") {
      throw new Error("temporal payload is empty or invalid");
    }
    if (!Array.isArray(payload.depth_levels) || payload.depth_levels.length === 0) {
      throw new Error("temporal payload has no depth levels");
    }
    if (!Array.isArray(payload.temporal_fields) || payload.temporal_fields.length === 0) {
      throw new Error("temporal payload has no temporal field definitions");
    }
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

  function setControlsDisabled(disabled) {
    document
      .querySelectorAll(
        "#temporal-modality-select, #temporal-field-select, #temporal-period-select, #temporal-depth-select, #temporal-metric-toggle button, #temporal-reset-focus"
      )
      .forEach((element) => {
        element.disabled = Boolean(disabled);
      });
  }

  function clearDashboardContent() {
    ["temporal-basin-ranking", "temporal-summary"].forEach((id) => {
      const element = $(id);
      if (element) {
        element.innerHTML = "";
      }
    });
    ["temporal-time-series", "temporal-depth-profile", "temporal-basin-chart"].forEach((id) => {
      const element = $(id);
      if (element && window.Plotly) {
        window.Plotly.purge(element);
      } else if (element) {
        element.innerHTML = "";
      }
    });
    if (state.map) {
      state.map.remove();
      state.map = null;
      state.mapCellLayer = null;
    }
  }

  function renderLoadFailure(error, sourceUrl) {
    document.body.classList.add("temporal-load-failed");
    clearDashboardContent();
    setControlsDisabled(true);
    $("temporal-run-label").textContent = "Temporal analysis data is unavailable";
    $("temporal-map-caption").textContent = "No temporal data loaded";
    $("temporal-summary-caption").textContent = "No temporal summary loaded";
    $("temporal-selection-pill").textContent = "Unavailable";
    const existing = document.querySelector(".temporal-error-state");
    if (existing) {
      existing.remove();
    }
    const panel = document.createElement("section");
    panel.className = "temporal-error-state";
    panel.innerHTML = [
      "<h2>Could not load the temporal dashboard data</h2>",
      "<p>The page is working, but the packaged temporal manifest, temporal JSON files, or dashboard libraries could not be loaded.</p>",
      `<p><strong>Manifest URL:</strong> <code>${escapeHtml(sourceUrl)}</code></p>`,
      `<p><strong>Error:</strong> ${escapeHtml(error && error.message ? error.message : error)}</p>`,
      "<p>Run the temporal export command. It writes <code>temporal-config.json</code> plus <code>temporal-analysis.json</code> files for every packaged modality.</p>",
    ].join("");
    document.querySelector(".temporal-shell").appendChild(panel);
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
  }

  function unitLabel() {
    return (data() && data().variable && data().variable.value_unit_label) || "";
  }

  function formatNumber(value, digits) {
    if (value === null || value === undefined || value === "") {
      return "n/a";
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "n/a";
    }
    return number.toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
  }

  function formatMetricValue(value) {
    const formatted = formatNumber(value, 2);
    return formatted === "n/a" ? formatted : `${formatted} ${unitLabel()}`.trim();
  }

  function formatDateLabel(value) {
    const raw = String(value || "").trim();
    return raw.replace(/\b(\d{4})(\d{2})(\d{2})\b/g, "$1-$2-$3");
  }

  function periodDisplayLabel(period) {
    return period && period.label ? formatDateLabel(period.label) : "n/a";
  }

  function formatCount(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? Math.round(number).toLocaleString() : "0";
  }

  function formatPixelCount(value, compact = false) {
    const count = Number(value || 0);
    const label = Math.round(count) === 1 ? "pixel" : "pixels";
    if (!compact || count < 1000) {
      return `${formatCount(count)} ${label}`;
    }
    if (count >= 1000000) {
      return `${formatNumber(count / 1000000, 1)}M ${label}`;
    }
    return `${formatNumber(count / 1000, 1)}K ${label}`;
  }

  function metricLabel(metric) {
    return METRIC_LABELS[metric] || metric;
  }

  function displayBasinName(name) {
    return BASIN_LABELS[name] || name;
  }

  function fieldDefinitions() {
    return data().temporal_fields || [];
  }

  function activeDepth() {
    const levels = data().depth_levels;
    state.depthIndex = Math.max(0, Math.min(state.depthIndex, levels.length - 1));
    return levels[state.depthIndex];
  }

  function fieldPayload(depth = activeDepth()) {
    const fields = depth.fields || {};
    return fields[state.field] || null;
  }

  function periodsForField(depth = activeDepth()) {
    const payload = fieldPayload(depth);
    return payload && Array.isArray(payload.periods) ? payload.periods : [];
  }

  function activePeriod() {
    const periods = periodsForField();
    state.periodIndex = Math.max(0, Math.min(state.periodIndex, periods.length - 1));
    return periods[state.periodIndex] || null;
  }

  function findPeriodForDepth(depth, periodKey) {
    const payload = depth.fields && depth.fields[state.field];
    const periods = payload && Array.isArray(payload.periods) ? payload.periods : [];
    return periods.find((period) => period.period_key === periodKey) || null;
  }

  function fieldLabel() {
    const field = fieldDefinitions().find((item) => item.key === state.field);
    return (field && field.label) || state.field;
  }

  function focusLabel() {
    return state.focus.type === "basin" ? displayBasinName(state.focus.id) : state.focus.label;
  }

  function setRunLabel() {
    const active = data();
    const run = active.run || {};
    const variable = active.variable || {};
    $("temporal-run-label").textContent = `${variable.label || state.activeVariable || "Variable"} temporal consistency | ${formatDateLabel(run.start_date || "unknown")} to ${formatDateLabel(run.end_date || "unknown")} | ${run.run_count || 0} runs`;
  }

  function populateDepthSelect() {
    const depthSelect = $("temporal-depth-select");
    depthSelect.innerHTML = data()
      .depth_levels.map((depth, index) => `<option value="${index}">${escapeHtml(depth.label)}</option>`)
      .join("");
    depthSelect.value = String(state.depthIndex);
  }

  function populateFieldSelect() {
    const fieldSelect = $("temporal-field-select");
    fieldSelect.innerHTML = fieldDefinitions()
      .map((field) => `<option value="${escapeHtml(field.key)}">${escapeHtml(field.label)}</option>`)
      .join("");
    if (!fieldDefinitions().some((field) => field.key === state.field)) {
      state.field = fieldDefinitions()[0].key;
    }
    fieldSelect.value = state.field;
  }

  function populatePeriodSelect() {
    const periodSelect = $("temporal-period-select");
    const periods = periodsForField();
    state.periodIndex = Math.max(0, Math.min(state.periodIndex, periods.length - 1));
    periodSelect.innerHTML = periods
      .map((period, index) => `<option value="${index}">${escapeHtml(periodDisplayLabel(period))}</option>`)
      .join("");
    periodSelect.value = String(state.periodIndex);
  }

  function populateMetricToggle() {
    const metrics = data().metrics || ["median", "mean", "p95"];
    if (!metrics.includes(state.metric)) {
      state.metric = metrics[0];
    }
    $("temporal-metric-toggle").innerHTML = metrics
      .map(
        (metric) =>
          `<button type="button" data-metric="${escapeHtml(metric)}" aria-pressed="${metric === state.metric}">${escapeHtml(metricLabel(metric))}</button>`
      )
      .join("");
  }

  function setupDashboardSelect() {
    const dashboardSelect = $("temporal-dashboard-select");
    if (!dashboardSelect) {
      return;
    }
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
    const modalitySelect = $("temporal-modality-select");
    modalitySelect.innerHTML = state.variables
      .map((variable) => `<option value="${escapeHtml(variable.key)}">${escapeHtml(variable.label)}</option>`)
      .join("");
    modalitySelect.value = state.activeVariable;
    modalitySelect.addEventListener("change", function () {
      state.activeVariable = modalitySelect.value;
      state.depthIndex = 0;
      state.periodIndex = 0;
      state.focus = { type: "global", id: "global", label: "Global" };
      setRunLabel();
      populateDepthSelect();
      populateFieldSelect();
      populatePeriodSelect();
      populateMetricToggle();
      render();
    });

    $("temporal-field-select").addEventListener("change", function (event) {
      state.field = event.target.value;
      state.periodIndex = 0;
      populatePeriodSelect();
      render();
    });

    $("temporal-period-select").addEventListener("change", function (event) {
      state.periodIndex = Number(event.target.value);
      render();
    });

    $("temporal-depth-select").addEventListener("change", function (event) {
      state.depthIndex = Number(event.target.value);
      populatePeriodSelect();
      render();
    });

    $("temporal-reset-focus").addEventListener("click", function () {
      state.focus = { type: "global", id: "global", label: "Global" };
      render();
    });

    $("temporal-metric-toggle").addEventListener("click", function (event) {
      const button = event.target.closest("button[data-metric]");
      if (!button) {
        return;
      }
      state.metric = button.dataset.metric;
      document.querySelectorAll("#temporal-metric-toggle button").forEach((item) => {
        item.setAttribute("aria-pressed", String(item === button));
      });
      render();
    });

    populateDepthSelect();
    populateFieldSelect();
    populatePeriodSelect();
    populateMetricToggle();
  }

  function basinForCoordinate(lon, lat) {
    const lonValue = ((((Number(lon) + 180) % 360) + 360) % 360) - 180;
    const latValue = Number(lat);
    if (!Number.isFinite(lonValue) || !Number.isFinite(latValue)) {
      return "Other";
    }
    if (latValue >= 66) {
      return "Arctic";
    }
    if (latValue <= -60) {
      return "Southern";
    }
    const atlantic =
      (lonValue >= -70 && lonValue < 20) ||
      (lonValue >= -10 && lonValue < 42 && latValue >= 30 && latValue < 48) ||
      (lonValue >= -25 && lonValue < 32 && latValue >= 48);
    if (atlantic && latValue > -60 && latValue < 66) {
      return "Atlantic";
    }
    const indian =
      (lonValue >= 20 && lonValue < 120 && latValue < 32) ||
      (lonValue >= 120 && lonValue < 147 && latValue < 0);
    if (indian && latValue > -60 && latValue < 66) {
      return "Indian";
    }
    const pacific =
      lonValue < -70 ||
      lonValue >= 120 ||
      (lonValue >= 100 && lonValue < 120 && latValue > -15 && latValue < 32);
    if (pacific && latValue > -60 && latValue < 66) {
      return "Pacific";
    }
    return "Other";
  }

  function basinForCell(cell) {
    if (cell.basin) {
      return cell.basin;
    }
    const lon = Number.isFinite(Number(cell.center_lon)) ? Number(cell.center_lon) : (Number(cell.west) + Number(cell.east)) / 2;
    const lat = Number.isFinite(Number(cell.center_lat)) ? Number(cell.center_lat) : (Number(cell.south) + Number(cell.north)) / 2;
    return basinForCoordinate(lon, lat);
  }

  function rankingButton(item, type, label) {
    const id = item.id || item.name;
    const active = state.focus.type === type && state.focus.id === id;
    return `<button type="button" class="temporal-ranking-button ${active ? "is-active" : ""}" data-type="${escapeHtml(type)}" data-id="${escapeHtml(id)}" data-label="${escapeHtml(label)}">
      <span><span class="temporal-ranking-name">${escapeHtml(label)}</span><span class="temporal-ranking-meta">${formatPixelCount(item.count, type === "basin")}</span></span>
      <span class="temporal-ranking-value">${formatNumber(item[state.metric], 2)}</span>
    </button>`;
  }

  function renderRankings() {
    const resetFocus = $("temporal-reset-focus");
    if (resetFocus) {
      resetFocus.disabled = state.focus.type === "global";
    }
    const period = activePeriod();
    const basins = period ? period.basins.filter((basin) => basin[state.metric] !== null) : [];
    $("temporal-basin-ranking").innerHTML = basins
      .map((basin) => rankingButton(basin, "basin", displayBasinName(basin.name)))
      .join("");
    document.querySelectorAll(".temporal-ranking-button").forEach((button) => {
      button.addEventListener("click", function () {
        const active = state.focus.type === button.dataset.type && state.focus.id === button.dataset.id;
        state.focus = active
          ? { type: "global", id: "global", label: "Global" }
          : { type: button.dataset.type, id: button.dataset.id, label: button.dataset.label };
        render();
      });
    });
  }

  function finiteMetricValues(cells) {
    return cells
      .map((cell) => Number(cell[state.metric]))
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b);
  }

  function quantile(sortedValues, fraction) {
    if (sortedValues.length === 0) {
      return null;
    }
    const index = (sortedValues.length - 1) * Math.max(0, Math.min(1, fraction));
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) {
      return sortedValues[lower];
    }
    const weight = index - lower;
    return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
  }

  function mapColorDomain(cells) {
    const values = finiteMetricValues(cells);
    if (values.length === 0) {
      return { lower: 0, upper: 1, clipped: false };
    }
    const min = values[0];
    const max = values[values.length - 1];
    const lower = Math.max(0, quantile(values, 0.05) ?? min);
    let upper = quantile(values, 0.95) ?? max;
    if (!Number.isFinite(upper) || upper <= lower) {
      upper = max > lower ? max : lower + 1;
    }
    return { lower, upper, clipped: max > upper };
  }

  function colorFor(value, domain) {
    const raw = Number(value || 0);
    const span = Math.max(1e-9, Number(domain.upper) - Number(domain.lower));
    const normalized = Math.max(0, Math.min(1, (raw - Number(domain.lower)) / span));
    const t = Math.pow(normalized, 0.72);
    const stops = [
      [44, 123, 182],
      [0, 166, 202],
      [127, 211, 78],
      [253, 174, 97],
      [215, 25, 28],
    ];
    const scaled = t * (stops.length - 1);
    const lowerIndex = Math.min(stops.length - 2, Math.floor(scaled));
    const upperIndex = lowerIndex + 1;
    const local = scaled - lowerIndex;
    const lower = stops[lowerIndex];
    const upper = stops[upperIndex];
    const rgb = lower.map((channel, index) => Math.round(channel + (upper[index] - channel) * local));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  function renderMapLegend(domain) {
    $("temporal-map-legend-min").textContent = `P5 or lower (${formatMetricValue(domain.lower)})`;
    $("temporal-map-legend-max").textContent = `${domain.clipped ? "P95+" : "Higher value"} (${formatMetricValue(domain.upper)})`;
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
    state.mapCellLayer = window.L.layerGroup().addTo(map);
    state.map = map;
    return map;
  }

  function cellBounds(cell) {
    return [
      [Number(cell.south), Number(cell.west)],
      [Number(cell.north), Number(cell.east)],
    ];
  }

  function cellTooltipHtml(cell) {
    return [
      `<strong>${escapeHtml(cell.label)}</strong>`,
      `Basin: ${escapeHtml(displayBasinName(basinForCell(cell)))}`,
      `${escapeHtml(fieldLabel())}: ${escapeHtml(formatMetricValue(cell[state.metric]))}`,
      `${escapeHtml(metricLabel(state.metric))}: ${escapeHtml(formatNumber(cell[state.metric], 2))}`,
      `Count: ${escapeHtml(formatPixelCount(cell.count))}`,
    ].join("<br>");
  }

  function cellMapStyle(cell, domain) {
    const basin = basinForCell(cell);
    const basinIsActive = state.focus.type === "basin" && state.focus.id === basin;
    const cellIsActive = state.focus.type === "cell" && state.focus.id === cell.id;
    let fillOpacity = 0.74;
    let opacity = 0.46;
    let color = "rgba(223,244,239,0.34)";
    let weight = 0.6;
    if (state.focus.type === "basin") {
      fillOpacity = basinIsActive ? 0.92 : 0.17;
      opacity = basinIsActive ? 0.9 : 0.18;
      color = basinIsActive ? "#ffd166" : "rgba(223,244,239,0.18)";
      weight = basinIsActive ? 1.2 : 0.4;
    } else if (state.focus.type === "cell") {
      fillOpacity = cellIsActive ? 0.96 : 0.22;
      opacity = cellIsActive ? 1 : 0.22;
      color = cellIsActive ? "#edf8f4" : "rgba(223,244,239,0.18)";
      weight = cellIsActive ? 2 : 0.4;
    }
    return {
      color,
      fillColor: colorFor(cell[state.metric], domain),
      fillOpacity,
      opacity,
      weight,
    };
  }

  function mapCellLayer(cell, colorDomain, geometryIndex) {
    const geometryFeature = geometryIndex ? geometryIndex[cell.id] : null;
    if (geometryIndex && !geometryFeature) {
      return null;
    }
    if (geometryFeature) {
      return window.L.geoJSON(geometryFeature, {
        style: cellMapStyle(cell, colorDomain),
      });
    }
    return window.L.rectangle(cellBounds(cell), cellMapStyle(cell, colorDomain));
  }

  function renderMap() {
    const map = state.map || createMap();
    const period = activePeriod();
    const cells = period ? period.grid_cells.filter((cell) => cell[state.metric] !== null) : [];
    const colorDomain = mapColorDomain(cells);
    const geometryIndex = activeGridGeometryIndex();
    renderMapLegend(colorDomain);
    state.mapCellLayer.clearLayers();
    for (const cell of cells) {
      const layer = mapCellLayer(cell, colorDomain, geometryIndex);
      if (!layer) {
        continue;
      }
      layer.bindTooltip(cellTooltipHtml(cell), {
        className: "temporal-leaflet-tooltip",
        direction: "auto",
        offset: [8, 8],
        opacity: 0.96,
        sticky: true,
      });
      layer.on("click", function () {
        const active = state.focus.type === "cell" && state.focus.id === cell.id;
        state.focus = active
          ? { type: "global", id: "global", label: "Global" }
          : { type: "cell", id: cell.id, label: cell.label };
        render();
      });
      state.mapCellLayer.addLayer(layer);
    }
    requestAnimationFrame(() => map.invalidateSize(false));
  }

  function basinStatsForPeriod(period, basinName) {
    return period && period.basins ? period.basins.find((basin) => basin.name === basinName) || null : null;
  }

  function cellStatsForPeriod(period, cellId) {
    return period && period.grid_cells ? period.grid_cells.find((cell) => cell.id === cellId) || null : null;
  }

  function focusedStats(period) {
    if (state.focus.type === "basin") {
      return basinStatsForPeriod(period, state.focus.id);
    }
    if (state.focus.type === "cell") {
      return cellStatsForPeriod(period, state.focus.id);
    }
    return period ? period.global : null;
  }

  function focusedMetric(period) {
    const stats = focusedStats(period);
    const value = Number(stats && stats[state.metric]);
    return Number.isFinite(value) ? value : null;
  }

  function plotlyLayout(yAxisTitle) {
    return {
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: cssVar("--temporal-text", "#d7e9f7"), family: "Roboto, sans-serif", size: 11 },
      margin: { l: 58, r: 16, t: 12, b: 44 },
      hoverlabel: {
        bgcolor: "#071d2d",
        bordercolor: cssVar("--temporal-border", "rgba(124,200,255,0.32)"),
        font: { color: cssVar("--temporal-text", "#d7e9f7") },
      },
      legend: { orientation: "h", x: 0, y: 1.12, xanchor: "left", yanchor: "bottom" },
      xaxis: {
        automargin: true,
        color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        linecolor: "rgba(223,244,239,0.24)",
        zerolinecolor: "rgba(223,244,239,0.18)",
      },
      yaxis: {
        automargin: true,
        color: cssVar("--temporal-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        linecolor: "rgba(223,244,239,0.24)",
        title: { text: yAxisTitle, standoff: 8 },
        zerolinecolor: "rgba(223,244,239,0.18)",
      },
    };
  }

  function chartAxisTitle() {
    const unit = unitLabel();
    return `${metricLabel(state.metric)} ${fieldLabel().toLowerCase()}${unit ? ` (${unit})` : ""}`;
  }

  function renderTimeSeries() {
    const chart = $("temporal-time-series");
    const periods = periodsForField();
    const trace = {
      customdata: periods.map((period) => [periodDisplayLabel(period), formatPixelCount((focusedStats(period) || {}).count)]),
      hovertemplate:
        "%{customdata[0]}<br>" +
        `${escapeHtml(metricLabel(state.metric))}: %{y:.2f} ${escapeHtml(unitLabel())}<br>` +
        "Count: %{customdata[1]}<extra></extra>",
      line: { color: state.focus.type === "global" ? "rgba(124,200,255,0.92)" : "#ffd166", width: 3 },
      marker: { color: "#7cc8ff", line: { color: "#f8f2d8", width: 1 }, size: 8 },
      mode: "lines+markers",
      name: focusLabel(),
      x: periods.map((period) => periodDisplayLabel(period)),
      y: periods.map((period) => focusedMetric(period)),
    };
    const layout = plotlyLayout(chartAxisTitle());
    layout.showlegend = false;
    layout.xaxis.tickangle = -18;
    $("temporal-series-caption").textContent = `${focusLabel()} ${fieldLabel().toLowerCase()} over time at ${activeDepth().label}`;
    window.Plotly.react(chart, [trace], layout, PLOTLY_CONFIG);
  }

  function chartDepthLevels() {
    const depthLevels = data().depth_levels.filter((depth) => !depth.is_aggregate);
    return depthLevels.length > 0 ? depthLevels : data().depth_levels;
  }

  function depthXValues(depthLevels) {
    return depthLevels.map((depth, index) => {
      const actualDepth = Number(depth.actual_depth_m);
      return Number.isFinite(actualDepth) ? actualDepth : index;
    });
  }

  function summaryCard(label, value) {
    return `<div class="temporal-summary-card"><span class="temporal-summary-label">${escapeHtml(label)}</span><span class="temporal-summary-value">${escapeHtml(value)}</span></div>`;
  }

  function renderTemporalSummary() {
    const period = activePeriod();
    const stats = focusedStats(period) || {};
    const details = [
      ["Focus", focusLabel()],
      ["Field", fieldLabel()],
      ["Interval", periodDisplayLabel(period)],
      ["Depth", activeDepth().label],
      [metricLabel(state.metric), formatMetricValue(stats[state.metric])],
      ["Count", formatPixelCount(stats.count, true)],
    ];
    $("temporal-summary-caption").textContent = `${focusLabel()} summary for ${period ? periodDisplayLabel(period) : "the active interval"}`;
    $("temporal-summary").innerHTML = details.map((item) => summaryCard(item[0], item[1])).join("");
  }

  function renderDepthProfile() {
    const chart = $("temporal-depth-profile");
    const period = activePeriod();
    const periodKey = period && period.period_key;
    const depthLevels = chartDepthLevels();
    const trace = {
      customdata: depthLevels.map((depth) => depth.label),
      hovertemplate:
        "%{customdata}<br>" +
        `${escapeHtml(metricLabel(state.metric))}: %{y:.2f} ${escapeHtml(unitLabel())}<extra></extra>`,
      line: { color: "rgba(124,200,255,0.92)", width: 3 },
      marker: { color: "#7cc8ff", line: { color: "#f8f2d8", width: 1 }, size: 8 },
      mode: "lines+markers",
      name: focusLabel(),
      x: depthXValues(depthLevels),
      y: depthLevels.map((depth) => {
        const matchingPeriod = periodKey ? findPeriodForDepth(depth, periodKey) : null;
        return focusedMetric(matchingPeriod);
      }),
    };
    const layout = plotlyLayout(chartAxisTitle());
    layout.showlegend = false;
    layout.xaxis.title = { text: "Depth (m)", standoff: 8 };
    $("temporal-depth-caption").textContent = `${focusLabel()} ${fieldLabel().toLowerCase()} across depth for ${period ? periodDisplayLabel(period) : "the active interval"}`;
    window.Plotly.react(chart, [trace], layout, PLOTLY_CONFIG);
  }

  function renderBasinChart() {
    const chart = $("temporal-basin-chart");
    const period = activePeriod();
    const basins = period ? period.basins.filter((basin) => basin[state.metric] !== null) : [];
    const labels = basins.map((basin) => displayBasinName(basin.name));
    const basinIds = basins.map((basin) => basin.name);
    const trace = {
      customdata: basins.map((basin) => [basin.name, formatPixelCount(basin.count, true)]),
      hovertemplate: "%{x}<br>" + metricLabel(state.metric) + ": %{y:.2f} " + unitLabel() + "<br>%{customdata[1]}<extra></extra>",
      marker: {
        color: basins.map((basin) => (state.focus.type === "basin" && state.focus.id === basin.name ? "#ffd166" : "#7cc8ff")),
        line: { color: "rgba(223,244,239,0.36)", width: 1 },
      },
      type: "bar",
      x: labels,
      y: basins.map((basin) => basin[state.metric]),
    };
    const layout = plotlyLayout(chartAxisTitle());
    layout.showlegend = false;
    layout.xaxis.automargin = true;
    layout.xaxis.tickangle = -24;
    $("temporal-basin-caption").textContent = `${metricLabel(state.metric)} ${fieldLabel().toLowerCase()} by basin`;
    Promise.resolve(window.Plotly.react(chart, [trace], layout, PLOTLY_CONFIG)).then(() => {
      if (typeof chart.removeAllListeners === "function") {
        chart.removeAllListeners("plotly_click");
      }
      if (typeof chart.on === "function") {
        chart.on("plotly_click", function (event) {
          const point = event.points && event.points[0];
          if (!point) {
            return;
          }
          const basinId = basinIds[point.pointIndex];
          const active = state.focus.type === "basin" && state.focus.id === basinId;
          state.focus = active
            ? { type: "global", id: "global", label: "Global" }
            : { type: "basin", id: basinId, label: displayBasinName(basinId) };
          render();
        });
      }
    });
  }

  function resizeVisuals() {
    if (state.map) {
      state.map.invalidateSize(false);
    }
    ["temporal-time-series", "temporal-depth-profile", "temporal-basin-chart"].forEach((id) => {
      const chart = $(id);
      if (chart && window.Plotly) {
        window.Plotly.Plots.resize(chart);
      }
    });
  }

  function render() {
    const period = activePeriod();
    renderRankings();
    renderMap();
    renderTimeSeries();
    renderTemporalSummary();
    renderDepthProfile();
    renderBasinChart();
    $("temporal-selection-pill").textContent = focusLabel();
    const geometryLabel = activeGridGeometryIndex() ? "coast-clipped " : "";
    $("temporal-map-caption").textContent = `${activeDepth().label} ${fieldLabel().toLowerCase()} by ${geometryLabel}${data().grouping.grid_size_degrees} degree cell${period ? ` | ${periodDisplayLabel(period)}` : ""}`;
  }

  function temporalSourcesFromConfig(config) {
    const variables = config.variables && typeof config.variables === "object" ? config.variables : null;
    if (!variables) {
      return [];
    }
    return Object.entries(variables)
      .map(([key, variableConfig]) => ({
        key,
        label: variableConfig.variable_label || variableConfig.variable || key,
        url: variableConfig.temporal_analysis_data_url,
        gridUrl: variableConfig.analysis_grid_geojson_url,
      }))
      .filter((item) => item.url);
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

  async function loadAllTemporalData(sourceUrl) {
    const config = await fetchJson(sourceUrl);
    const sources = temporalSourcesFromConfig(config);
    if (sources.length === 0) {
      throw new Error("temporal-config.json does not list any temporal analysis datasets");
    }
    const loaded = await Promise.all(
      sources.map(async (source) => {
        const payloadUrl = new URL(source.url, sourceUrl).toString();
        const gridUrl = source.gridUrl ? new URL(source.gridUrl, sourceUrl).toString() : null;
        const [payload, gridPayload] = await Promise.all([fetchJson(payloadUrl), fetchOptionalJson(gridUrl)]);
        validatePayload(payload);
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
      loaded
        .filter((item) => item.gridGeometryIndex)
        .map((item) => [item.key, item.gridGeometryIndex])
    );
    state.activeVariable =
      (config.default_variable && state.datasets[config.default_variable] && config.default_variable) || loaded[0].key;
  }

  async function init() {
    const sourceUrl = configUrl();
    setupDashboardSelect();
    try {
      requireDashboardLibraries();
      await loadAllTemporalData(sourceUrl);
      document.body.classList.remove("temporal-load-failed");
      setRunLabel();
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
