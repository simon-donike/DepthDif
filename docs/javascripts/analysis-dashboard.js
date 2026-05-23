(function () {
  const DEFAULT_GLOBE_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/globe/globe-config.json";
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
    variables: [],
    activeVariable: null,
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

  function setControlsDisabled(disabled) {
    document
      .querySelectorAll(
        "#analysis-modality-select, #analysis-depth-select, #analysis-metric-toggle button, #analysis-reset-focus"
      )
      .forEach((element) => {
        element.disabled = Boolean(disabled);
      });
  }

  function validateAnalysisPayload(payload) {
    if (!payload || typeof payload !== "object") {
      throw new Error("analysis payload is empty or invalid");
    }
    if (!Array.isArray(payload.depth_levels) || payload.depth_levels.length === 0) {
      throw new Error("analysis payload has no depth levels");
    }
    if (!Array.isArray(payload.metrics) || payload.metrics.length === 0) {
      throw new Error("analysis payload has no metric definitions");
    }
  }

  function clearDashboardContent() {
    ["analysis-kpis", "analysis-basin-ranking"].forEach((id) => {
      const element = $(id);
      if (element) {
        element.innerHTML = "";
      }
    });
    ["analysis-depth-profile", "analysis-basin-chart"].forEach((id) => {
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
    document.body.classList.add("analysis-load-failed");
    clearDashboardContent();
    setControlsDisabled(true);
    $("analysis-run-label").textContent = "Analysis data is unavailable";
    $("analysis-map-caption").textContent = "No analysis data loaded";
    $("analysis-profile-caption").textContent = "No depth profile loaded";
    $("analysis-selection-pill").textContent = "Unavailable";
    const existing = document.querySelector(".analysis-error-state");
    if (existing) {
      existing.remove();
    }
    const panel = document.createElement("section");
    panel.className = "analysis-error-state";
    panel.innerHTML = [
      "<h2>Could not load the analysis dashboard data</h2>",
      "<p>The page is working, but the packaged globe manifest, analysis JSON files, or dashboard libraries could not be loaded.</p>",
      `<p><strong>Manifest URL:</strong> <code>${escapeHtml(sourceUrl)}</code></p>`,
      `<p><strong>Error:</strong> ${escapeHtml(error && error.message ? error.message : error)}</p>`,
      "<p>Run the normal globe packaging/export command. It writes <code>globe-config.json</code> plus <code>error-analysis.json</code> files for every packaged modality.</p>",
    ].join("");
    document.querySelector(".analysis-shell").appendChild(panel);
  }

  function unitLabel() {
    return (data() && data().variable && data().variable.value_unit_label) || "";
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
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

  function formatMetric(value) {
    const formatted = formatNumber(value, 2);
    return formatted === "n/a" ? formatted : `${formatted} ${unitLabel()}`.trim();
  }

  function formatCount(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? Math.round(number).toLocaleString() : "0";
  }

  function formatCompactCount(value) {
    const number = Number(value || 0);
    if (!Number.isFinite(number)) {
      return "0";
    }
    const abs = Math.abs(number);
    if (abs >= 1_000_000_000) {
      return `${formatNumber(number / 1_000_000_000, 1)}B`;
    }
    if (abs >= 1_000_000) {
      return `${formatNumber(number / 1_000_000, 1)}M`;
    }
    if (abs >= 1_000) {
      return `${formatNumber(number / 1_000, 1)}K`;
    }
    return formatCount(number);
  }

  function formatPixelCount(value, compact = false) {
    const count = Number(value || 0);
    const label = Math.round(count) === 1 ? "pixel" : "pixels";
    return `${compact ? formatCompactCount(count) : formatCount(count)} ${label}`;
  }

  function activeDepth() {
    return data().depth_levels[state.depthIndex];
  }

  function chartDepthLevels() {
    const depthLevels = data().depth_levels.filter((depth) => !depth.is_aggregate);
    return depthLevels.length > 0 ? depthLevels : data().depth_levels;
  }

  function depthSubtitle(depth) {
    if (depth.is_aggregate) {
      const depthCount = Number(depth.depth_count || Math.max(0, data().depth_levels.length - 1));
      return `${formatCount(depthCount)} depth levels pooled`;
    }
    return `${formatNumber(depth.actual_depth_m, 1)} m actual`;
  }

  function metricLabel(metric) {
    return METRIC_LABELS[metric] || metric;
  }

  function displayBasinName(name) {
    return BASIN_LABELS[name] || name;
  }

  function setRunLabel() {
    const active = data();
    const run = active.run || {};
    const variable = active.variable || {};
    const date = run.target_date || run.selected_date || "unknown date";
    const week = run.iso_year && run.iso_week ? `ISO ${run.iso_year}-W${String(run.iso_week).padStart(2, "0")}` : "single run";
    $("analysis-run-label").textContent = `${variable.label || state.activeVariable || "Variable"} absolute error | ${date} | ${week}`;
  }

  function setupControls() {
    const modalitySelect = $("analysis-modality-select");
    modalitySelect.innerHTML = state.variables
      .map((variable) => `<option value="${escapeHtml(variable.key)}">${escapeHtml(variable.label)}</option>`)
      .join("");
    modalitySelect.value = state.activeVariable;
    modalitySelect.addEventListener("change", function () {
      state.activeVariable = modalitySelect.value;
      state.depthIndex = 0;
      state.focus = { type: "global", id: "global", label: "Global" };
      setRunLabel();
      populateDepthSelect();
      populateMetricToggle();
      render();
    });

    const depthSelect = $("analysis-depth-select");
    depthSelect.addEventListener("change", function () {
      state.depthIndex = Number(depthSelect.value);
      render();
    });

    const resetFocus = $("analysis-reset-focus");
    resetFocus.addEventListener("click", function () {
      state.focus = { type: "global", id: "global", label: "Global" };
      render();
    });

    const metricToggle = $("analysis-metric-toggle");
    metricToggle.addEventListener("click", function (event) {
      const button = event.target.closest("button[data-metric]");
      if (!button) {
        return;
      }
      state.metric = button.dataset.metric;
      metricToggle.querySelectorAll("button").forEach((item) => {
        item.setAttribute("aria-pressed", String(item === button));
      });
      render();
    });

    populateDepthSelect();
    populateMetricToggle();
  }

  function populateDepthSelect() {
    const depthSelect = $("analysis-depth-select");
    depthSelect.innerHTML = data().depth_levels
      .map((depth, index) => `<option value="${index}">${escapeHtml(depth.label)}</option>`)
      .join("");
    state.depthIndex = Math.min(state.depthIndex, data().depth_levels.length - 1);
    depthSelect.value = String(state.depthIndex);
  }

  function populateMetricToggle() {
    const metrics = data().metrics || ["median", "mean", "p95"];
    if (!metrics.includes(state.metric)) {
      state.metric = metrics[0];
    }
    const metricToggle = $("analysis-metric-toggle");
    metricToggle.innerHTML = metrics
      .map(
        (metric) =>
          `<button type="button" data-metric="${escapeHtml(metric)}" aria-pressed="${metric === state.metric}">${escapeHtml(metricLabel(metric))}</button>`
      )
      .join("");
  }

  function focusLabel() {
    return state.focus.type === "basin" ? displayBasinName(state.focus.id) : state.focus.label;
  }

  function activeSelectionStats() {
    const depth = activeDepth();
    if (state.focus.type === "basin") {
      return depth.basins.find((basin) => basin.name === state.focus.id) || null;
    }
    if (state.focus.type === "cell") {
      return depth.grid_cells.find((cell) => cell.id === state.focus.id) || null;
    }
    return depth.global;
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

  function renderKpis() {
    const depth = activeDepth();
    const stats = activeSelectionStats() || {};
    const count = Number(stats.count || 0);
    const countLabel = count > 0 ? `${formatPixelCount(count, state.focus.type === "basin")} valid` : "No valid pixels at this depth";
    const values = [
      ["Active Depth", depth.label, depthSubtitle(depth)],
      ["Statistics", focusLabel(), countLabel],
      ["Median Error", formatMetric(stats.median), `${metricLabel(state.metric)} metric active`],
      ["Mean Error", formatMetric(stats.mean), "Average absolute error"],
      ["P95 Error", formatMetric(stats.p95), "Tail absolute error"],
    ];
    $("analysis-kpis").innerHTML = values
      .map(
        ([label, value, sub]) =>
          `<article class="analysis-kpi"><div class="analysis-kpi__label">${escapeHtml(label)}</div><div class="analysis-kpi__value">${escapeHtml(value)}</div><div class="analysis-kpi__sub">${escapeHtml(sub)}</div></article>`
      )
      .join("");
  }

  function rankingButton(item, type, label) {
    const id = item.id || item.name;
    const active = state.focus.type === type && state.focus.id === id;
    return `<button type="button" class="analysis-ranking-button ${active ? "is-active" : ""}" data-type="${escapeHtml(type)}" data-id="${escapeHtml(id)}" data-label="${escapeHtml(label)}">
      <span><span class="analysis-ranking-name">${escapeHtml(label)}</span><span class="analysis-ranking-meta">${formatPixelCount(item.count, type === "basin")}</span></span>
      <span class="analysis-ranking-value">${formatNumber(item[state.metric], 2)}</span>
    </button>`;
  }

  function renderRankings() {
    const resetFocus = $("analysis-reset-focus");
    if (resetFocus) {
      resetFocus.disabled = state.focus.type === "global";
    }
    const depth = activeDepth();
    const basins = depth.basins
      .filter((basin) => basin[state.metric] !== null)
      .sort((a, b) => Number(b[state.metric]) - Number(a[state.metric]));
    $("analysis-basin-ranking").innerHTML = basins
      .map((basin) => rankingButton(basin, "basin", displayBasinName(basin.name)))
      .join("");
    document.querySelectorAll(".analysis-ranking-button").forEach((button) => {
      button.addEventListener("click", function () {
        state.focus = { type: button.dataset.type, id: button.dataset.id, label: button.dataset.label };
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
    const legendMin = $("analysis-map-legend-min");
    const legendMax = $("analysis-map-legend-max");
    if (legendMin) {
      legendMin.textContent = `P5 or lower (${formatMetric(domain.lower)})`;
    }
    if (legendMax) {
      legendMax.textContent = `${domain.clipped ? "P95+" : "Higher error"} (${formatMetric(domain.upper)})`;
    }
  }

  function createMap() {
    const map = window.L.map("analysis-map", {
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
      `${escapeHtml(metricLabel(state.metric))}: ${escapeHtml(formatMetric(cell[state.metric]))}`,
      `P95: ${escapeHtml(formatMetric(cell.p95))}`,
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

  function renderMap() {
    const map = state.map || createMap();
    const cells = activeDepth().grid_cells.filter((cell) => cell[state.metric] !== null);
    const colorDomain = mapColorDomain(cells);
    renderMapLegend(colorDomain);
    state.mapCellLayer.clearLayers();
    for (const cell of cells) {
      const layer = window.L.rectangle(cellBounds(cell), cellMapStyle(cell, colorDomain));
      layer.bindTooltip(cellTooltipHtml(cell), {
        className: "analysis-leaflet-tooltip",
        direction: "auto",
        offset: [8, 8],
        opacity: 0.96,
        sticky: true,
      });
      layer.on("click", function () {
        state.focus = { type: "cell", id: cell.id, label: cell.label };
        render();
      });
      state.mapCellLayer.addLayer(layer);
    }
    requestAnimationFrame(() => map.invalidateSize(false));
  }

  function selectedSeries(depthLevels = chartDepthLevels()) {
    return depthLevels.map((depth) => {
      if (state.focus.type === "basin") {
        return (depth.basins.find((basin) => basin.name === state.focus.id) || {})[state.metric] ?? null;
      }
      if (state.focus.type === "cell") {
        return (depth.grid_cells.find((cell) => cell.id === state.focus.id) || {})[state.metric] ?? null;
      }
      return depth.global[state.metric] ?? null;
    });
  }

  function globalSeries(depthLevels = chartDepthLevels()) {
    return depthLevels.map((depth) => depth.global[state.metric] ?? null);
  }

  function depthXValues(depthLevels = chartDepthLevels()) {
    return depthLevels.map((depth, index) => {
      const actualDepth = Number(depth.actual_depth_m);
      return Number.isFinite(actualDepth) ? actualDepth : index;
    });
  }

  function plotlyLayout(yAxisTitle) {
    return {
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: cssVar("--analysis-text", "#d7e9f7"), family: "Roboto, sans-serif", size: 11 },
      margin: { l: 58, r: 16, t: 12, b: 44 },
      hoverlabel: {
        bgcolor: "#071d2d",
        bordercolor: cssVar("--analysis-border", "rgba(124,200,255,0.32)"),
        font: { color: cssVar("--analysis-text", "#d7e9f7") },
      },
      legend: { orientation: "h", x: 0, y: 1.12, xanchor: "left", yanchor: "bottom" },
      xaxis: {
        automargin: true,
        color: cssVar("--analysis-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        linecolor: "rgba(223,244,239,0.24)",
        zerolinecolor: "rgba(223,244,239,0.18)",
      },
      yaxis: {
        automargin: true,
        color: cssVar("--analysis-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        linecolor: "rgba(223,244,239,0.24)",
        title: { text: yAxisTitle, standoff: 8 },
        zerolinecolor: "rgba(223,244,239,0.18)",
      },
    };
  }

  function chartAxisTitle() {
    const unit = unitLabel();
    return `${metricLabel(state.metric)} absolute error${unit ? ` (${unit})` : ""}`;
  }

  function renderDepthProfile() {
    const chart = $("analysis-depth-profile");
    const depthLevels = chartDepthLevels();
    const labels = depthLevels.map((depth) => depth.label);
    const selected = selectedSeries(depthLevels);
    const global = globalSeries(depthLevels);
    const xValues = depthXValues(depthLevels);
    const traces = [
      {
        customdata: labels,
        hovertemplate: "Global<br>%{customdata}<br>%{y:.2f} " + unitLabel() + "<extra></extra>",
        line: { color: "rgba(124,200,255,0.72)", width: 2 },
        marker: { color: "#7cc8ff", size: 7 },
        mode: "lines+markers",
        name: "Global",
        x: xValues,
        y: global,
      },
    ];
    if (state.focus.type !== "global") {
      traces.push({
        customdata: labels,
        hovertemplate: `${escapeHtml(focusLabel())}<br>%{customdata}<br>%{y:.2f} ${unitLabel()}<extra></extra>`,
        line: { color: "#ffd166", width: 3 },
        marker: { color: "#ffd166", size: 8 },
        mode: "lines+markers",
        name: focusLabel(),
        x: xValues,
        y: selected,
      });
    }
    const layout = plotlyLayout(chartAxisTitle());
    layout.hovermode = "closest";
    layout.xaxis.title = { text: "Depth (m)", standoff: 8 };
    $("analysis-profile-caption").textContent = `${focusLabel()} ${metricLabel(state.metric).toLowerCase()} absolute error across depth`;
    window.Plotly.react(chart, traces, layout, PLOTLY_CONFIG);
  }

  function renderBasinChart() {
    const chart = $("analysis-basin-chart");
    const basins = activeDepth().basins.filter((basin) => basin[state.metric] !== null);
    const labels = basins.map((basin) => displayBasinName(basin.name));
    const basinIds = basins.map((basin) => basin.name);
    const colors = basins.map((basin) => (state.focus.type === "basin" && state.focus.id === basin.name ? "#ffd166" : "#7cc8ff"));
    const trace = {
      customdata: basins.map((basin) => [basin.name, formatPixelCount(basin.count, true)]),
      hovertemplate: "%{x}<br>" + metricLabel(state.metric) + ": %{y:.2f} " + unitLabel() + "<br>%{customdata[1]}<extra></extra>",
      marker: { color: colors, line: { color: "rgba(223,244,239,0.36)", width: 1 } },
      type: "bar",
      x: labels,
      y: basins.map((basin) => basin[state.metric]),
    };
    const layout = plotlyLayout(chartAxisTitle());
    layout.showlegend = false;
    layout.xaxis.automargin = true;
    layout.xaxis.tickangle = -24;
    $("analysis-basin-caption").textContent = `${metricLabel(state.metric)} absolute error by basin at ${activeDepth().label}`;
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
          state.focus = { type: "basin", id: basinId, label: displayBasinName(basinId) };
          render();
        });
      }
    });
  }

  function resizeVisuals() {
    if (state.map) {
      state.map.invalidateSize(false);
    }
    ["analysis-depth-profile", "analysis-basin-chart"].forEach((id) => {
      const chart = $(id);
      if (chart && window.Plotly) {
        window.Plotly.Plots.resize(chart);
      }
    });
  }

  function render() {
    renderKpis();
    renderRankings();
    renderMap();
    renderDepthProfile();
    renderBasinChart();
    $("analysis-selection-pill").textContent = focusLabel();
    $("analysis-map-caption").textContent = `${activeDepth().label} ${metricLabel(state.metric).toLowerCase()} absolute error by ${data().grouping.grid_size_degrees} degree cell`;
  }

  function analysisSourcesFromConfig(config) {
    const variables = config.variables && typeof config.variables === "object" ? config.variables : null;
    if (variables) {
      return Object.entries(variables)
        .map(([key, variableConfig]) => ({
          key,
          label: variableConfig.variable_label || variableConfig.variable || key,
          url: variableConfig.error_analysis_data_url,
        }))
        .filter((item) => item.url);
    }
    if (config.error_analysis_data_url) {
      return [
        {
          key: config.variable || "default",
          label: config.variable_label || config.variable || "Default",
          url: config.error_analysis_data_url,
        },
      ];
    }
    return [];
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`${url} returned HTTP ${response.status}`);
    }
    return response.json();
  }

  async function loadAllAnalysisData() {
    const config = await fetchJson(DEFAULT_GLOBE_CONFIG_URL);
    const sources = analysisSourcesFromConfig(config);
    if (sources.length === 0) {
      throw new Error("globe-config.json does not list any error analysis datasets");
    }
    const loaded = await Promise.all(
      sources.map(async (source) => {
        const payload = await fetchJson(new URL(source.url, DEFAULT_GLOBE_CONFIG_URL).toString());
        validateAnalysisPayload(payload);
        return {
          key: source.key,
          label: (payload.variable && payload.variable.label) || source.label,
          payload,
        };
      })
    );
    state.variables = loaded.map((item) => ({ key: item.key, label: item.label }));
    state.datasets = Object.fromEntries(loaded.map((item) => [item.key, item.payload]));
    state.activeVariable =
      (config.default_variable && state.datasets[config.default_variable] && config.default_variable) || loaded[0].key;
  }

  async function init() {
    try {
      requireDashboardLibraries();
      await loadAllAnalysisData();
      document.body.classList.remove("analysis-load-failed");
      setRunLabel();
      setupControls();
      setControlsDisabled(false);
      window.addEventListener("resize", resizeVisuals);
      render();
    } catch (error) {
      console.error(error);
      renderLoadFailure(error, DEFAULT_GLOBE_CONFIG_URL);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
