(function () {
  const DEFAULT_GLOBE_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/globe/globe-config.json";
  const METRIC_LABELS = { median: "Median", mean: "Mean", p90: "P90", p95: "P95" };
  const BASIN_FAN_COLORS = ["#7cc8ff", "#f6c85f", "#6cc4a1", "#e17c78", "#b39ddb", "#9ad6cb", "#f2a65a", "#88a7ff", "#c7e75f"];
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
  const COMMON_SELECTOR_DEPTHS_M = [0, 10, 50, 100, 250, 500, 1000, 2000, 2500, 5000];
  const state = {
    datasets: {},
    gridGeometries: {},
    variables: [],
    activeVariable: null,
    depthIndex: 0,
    metric: "median",
    depthProfileLogX: false,
    showBasinFan: true,
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

  function setControlsDisabled(disabled) {
    document
      .querySelectorAll(
        "#analysis-modality-select, #analysis-depth-select, #analysis-metric-toggle button, #analysis-reset-focus, #analysis-depth-scale-toggle, #analysis-basin-fan-toggle"
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
    ["analysis-basin-ranking", "analysis-detail-summary", "analysis-uncertainty-highlights"].forEach((id) => {
      const element = $(id);
      if (element) {
        element.innerHTML = "";
      }
    });
    ["analysis-depth-profile", "analysis-uncertainty-chart"].forEach((id) => {
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
    $("analysis-detail-caption").textContent = "No selected depth loaded";
    $("analysis-uncertainty-caption").textContent = "No uncertainty reliability data loaded";
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

  function formatDateLabel(value) {
    const raw = String(value || "").trim();
    return raw.replace(/\b(\d{4})(\d{2})(\d{2})\b/g, "$1-$2-$3");
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

  function commonSelectorDepthLabel(depthM) {
    const targetDepthM = Number(depthM);
    if (!Number.isFinite(targetDepthM)) {
      return "";
    }
    if (Math.abs(targetDepthM) < 0.05) {
      return "Surface";
    }
    return formatDepthMeters({ requested_depth_m: targetDepthM, actual_depth_m: targetDepthM });
  }

  function commonDepthTargetsByIndex() {
    const levels = data().depth_levels;
    const targets = {};
    const nativeDepths = levels
      .map((depth, index) => ({ depth, index }))
      .filter((item) => depthHasMetricSupport(item.depth));
    for (const targetDepth of COMMON_SELECTOR_DEPTHS_M) {
      let best = null;
      for (const item of nativeDepths) {
        const actualDepth = Number(item.depth.actual_depth_m);
        if (!Number.isFinite(actualDepth)) {
          continue;
        }
        const distance = Math.abs(actualDepth - targetDepth);
        if (best === null || distance < best.distance) {
          best = { index: item.index, distance };
        }
      }
      if (best !== null && targets[best.index] === undefined) {
        targets[best.index] = targetDepth;
      }
    }
    return targets;
  }

  function depthDisplayLabel(depth, depthIndex = null) {
    if (!depth || depth.is_aggregate) {
      return depth && depth.label ? String(depth.label) : "";
    }
    if (Number.isInteger(depthIndex)) {
      const targetDepth = commonDepthTargetsByIndex()[depthIndex];
      if (targetDepth !== undefined) {
        return commonSelectorDepthLabel(targetDepth);
      }
    }
    return formatDepthMeters(depth);
  }

  function intValue(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? Math.round(number) : 0;
  }

  function hasMetricSupport(stats) {
    if (!stats || intValue(stats.count) <= 0) {
      return false;
    }
    return Number.isFinite(Number(stats[state.metric]));
  }

  function depthHasMetricSupport(depth) {
    return Boolean(depth && !depth.is_aggregate && hasMetricSupport(depth.global));
  }

  function activeDepth() {
    return data().depth_levels[state.depthIndex];
  }

  function chartDepthLevels() {
    return data().depth_levels.filter((depth) => depthHasMetricSupport(depth));
  }

  function selectableDepthOptions() {
    const levels = data().depth_levels;
    const selected = new Map();
    const aggregateIndex = levels.findIndex((depth) => depth.is_aggregate);
    if (aggregateIndex >= 0) {
      selected.set(aggregateIndex, depthDisplayLabel(levels[aggregateIndex], aggregateIndex));
    }
    const targetDepthsByIndex = commonDepthTargetsByIndex();
    Object.keys(targetDepthsByIndex).forEach((index) => {
      selected.set(Number(index), commonSelectorDepthLabel(targetDepthsByIndex[index]));
    });
    if (
      state.depthIndex >= 0 &&
      state.depthIndex < levels.length &&
      (levels[state.depthIndex].is_aggregate || depthHasMetricSupport(levels[state.depthIndex])) &&
      !selected.has(state.depthIndex)
    ) {
      selected.set(state.depthIndex, depthDisplayLabel(levels[state.depthIndex], state.depthIndex));
    }
    return Array.from(selected, ([index, label]) => ({ index, label })).sort(
      (left, right) => left.index - right.index
    );
  }

  function selectableDepthIndices() {
    return selectableDepthOptions().map((option) => option.index);
  }

  function metricLabel(metric) {
    return METRIC_LABELS[metric] || metric;
  }

  function isDisplayBasin(name) {
    return name && name !== "Other";
  }

  function displayBasinName(name) {
    return name || "Other";
  }

  function displayBasinNames() {
    const grouping = (data() && data().grouping) || {};
    const configured = Array.isArray(grouping.basins) ? grouping.basins.filter(isDisplayBasin) : [];
    if (configured.length > 0) {
      return configured;
    }
    const names = new Set();
    data().depth_levels.forEach((depth) => {
      (depth.basins || []).forEach((basin) => {
        if (isDisplayBasin(basin.name)) {
          names.add(basin.name);
        }
      });
    });
    return Array.from(names);
  }

  function basinFanColor(index) {
    return BASIN_FAN_COLORS[index % BASIN_FAN_COLORS.length];
  }

  function setRunLabel() {
    const active = data();
    const run = active.run || {};
    const variable = active.variable || {};
    const date = formatDateLabel(run.target_date || run.selected_date || "unknown date");
    const week = run.iso_year && run.iso_week ? `ISO ${run.iso_year}-W${String(run.iso_week).padStart(2, "0")}` : "single run";
    $("analysis-run-label").textContent = `${variable.label || state.activeVariable || "Variable"} absolute error | ${date} | ${week}`;
  }

  function setupDashboardSelect() {
    const dashboardSelect = $("analysis-dashboard-select");
    if (!dashboardSelect) {
      return;
    }
    dashboardSelect.value = "analysis";
    dashboardSelect.addEventListener("change", function () {
      if (dashboardSelect.value === "temporal") {
        window.location.href = "../temporal-dashboard/";
        return;
      }
      dashboardSelect.value = "analysis";
    });
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

    const depthScaleToggle = $("analysis-depth-scale-toggle");
    depthScaleToggle.addEventListener("click", function () {
      state.depthProfileLogX = !state.depthProfileLogX;
      updateDepthScaleToggle();
      renderDepthProfile();
    });

    const basinFanToggle = $("analysis-basin-fan-toggle");
    basinFanToggle.addEventListener("click", function () {
      state.showBasinFan = !state.showBasinFan;
      updateBasinFanToggle();
      renderDepthProfile();
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
    updateDepthScaleToggle();
    updateBasinFanToggle();
  }

  function populateDepthSelect() {
    const depthSelect = $("analysis-depth-select");
    const depthLevels = data().depth_levels;
    state.depthIndex = Math.max(0, Math.min(state.depthIndex, depthLevels.length - 1));
    const options = selectableDepthOptions();
    const indices = options.map((option) => option.index);
    if (!indices.includes(state.depthIndex)) {
      state.depthIndex = indices[0] ?? 0;
    }
    depthSelect.innerHTML = options
      .map((option) => `<option value="${option.index}">${escapeHtml(option.label)}</option>`)
      .join("");
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

  function updateDepthScaleToggle() {
    const button = $("analysis-depth-scale-toggle");
    if (!button) {
      return;
    }
    const logScaleEnabled = Boolean(state.depthProfileLogX);
    button.setAttribute("aria-pressed", String(logScaleEnabled));
    button.textContent = logScaleEnabled ? "Linear X" : "Log X";
    button.setAttribute("aria-label", logScaleEnabled ? "Use linear depth axis" : "Use logarithmic depth axis");
    button.title = logScaleEnabled ? "Use linear depth axis" : "Use logarithmic depth axis";
  }

  function updateBasinFanToggle() {
    const button = $("analysis-basin-fan-toggle");
    if (!button) {
      return;
    }
    const basinFanAvailable = state.focus.type === "global";
    const basinFanEnabled = basinFanAvailable && state.showBasinFan;
    button.disabled = !basinFanAvailable;
    button.setAttribute("aria-pressed", String(basinFanEnabled));
    button.setAttribute("aria-label", state.showBasinFan ? "Hide basin fan curves" : "Show basin fan curves");
    button.title = basinFanAvailable ? button.getAttribute("aria-label") : "Basin fan is available only in global focus";
  }

  function focusLabel() {
    return state.focus.type === "basin" ? displayBasinName(state.focus.id) : state.focus.label;
  }

  function basinForCell(cell) {
    return cell.basin || "Other";
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
    const basins = depth.basins.filter((basin) => isDisplayBasin(basin.name) && hasMetricSupport(basin));
    $("analysis-basin-ranking").innerHTML = basins
      .map((basin) => rankingButton(basin, "basin", displayBasinName(basin.name)))
      .join("");
    document.querySelectorAll(".analysis-ranking-button").forEach((button) => {
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
    const basin = basinForCell(cell);
    const lines = [`<strong>${escapeHtml(cell.label)}</strong>`];
    if (isDisplayBasin(basin)) {
      lines.push(`Basin: ${escapeHtml(displayBasinName(basin))}`);
    }
    lines.push(
      `${escapeHtml(metricLabel(state.metric))}: ${escapeHtml(formatMetric(cell[state.metric]))}`,
      `P95: ${escapeHtml(formatMetric(cell.p95))}`,
      `Count: ${escapeHtml(formatPixelCount(cell.count))}`
    );
    return lines.join("<br>");
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
    const cells = activeDepth().grid_cells.filter((cell) => hasMetricSupport(cell));
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
        className: "analysis-leaflet-tooltip",
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

  function basinStatsForDepth(depth, basinName) {
    return depth.basins.find((basin) => basin.name === basinName) || null;
  }

  function cellStatsForDepth(depth, cellId) {
    return depth.grid_cells.find((cell) => cell.id === cellId) || null;
  }

  function focusedStats(depth) {
    if (state.focus.type === "basin") {
      return basinStatsForDepth(depth, state.focus.id);
    }
    if (state.focus.type === "cell") {
      return cellStatsForDepth(depth, state.focus.id);
    }
    return depth.global;
  }

  function metricValue(stats) {
    if (!hasMetricSupport(stats)) {
      return null;
    }
    const value = Number(stats && stats[state.metric]);
    return Number.isFinite(value) ? value : null;
  }

  function selectedSeries(depthLevels = chartDepthLevels()) {
    return depthLevels.map((depth) => metricValue(focusedStats(depth)));
  }


  function basinSeries(depthLevels, basinName) {
    return depthLevels.map((depth) => metricValue(basinStatsForDepth(depth, basinName)));
  }

  function depthXValues(depthLevels = chartDepthLevels()) {
    return depthLevels.map((depth, index) => {
      const actualDepth = Number(depth.actual_depth_m);
      return Number.isFinite(actualDepth) ? actualDepth : index;
    });
  }

  function logDepthXValues(xValues) {
    const positiveDepths = xValues.filter((value) => Number.isFinite(Number(value)) && Number(value) > 0);
    const firstPositiveDepth = positiveDepths.length > 0 ? Math.min(...positiveDepths) : 1;
    const surfaceDepth = Math.max(firstPositiveDepth / 2, 1e-6);
    return xValues.map((value) => {
      const depth = Number(value);
      // Plotly log axes cannot draw x <= 0; keep surface points just left of the first measured depth.
      return Number.isFinite(depth) && depth > 0 ? depth : surfaceDepth;
    });
  }

  function finitePlotValue(value) {
    if (value === null || value === undefined || value === "") {
      return false;
    }
    return Number.isFinite(Number(value));
  }

  function visibleDepthAxisRange(traces) {
    const visibleDepths = [];
    traces.forEach((trace) => {
      (trace.x || []).forEach((xValue, index) => {
        const yValue = (trace.y || [])[index];
        const depth = Number(xValue);
        if (finitePlotValue(yValue) && Number.isFinite(depth)) {
          visibleDepths.push(depth);
        }
      });
    });
    if (visibleDepths.length === 0) {
      return null;
    }
    if (!state.depthProfileLogX) {
      const maxDepth = Math.max(...visibleDepths);
      return [0, maxDepth > 0 ? maxDepth : 1];
    }
    const positiveDepths = visibleDepths.filter((value) => value > 0);
    if (positiveDepths.length === 0) {
      return null;
    }
    // Plotly log ranges are expressed as log10 values, while trace values stay in depth meters.
    return [Math.log10(Math.min(...positiveDepths)), Math.log10(Math.max(...positiveDepths))];
  }

  function commonRasterDepthPointIndices(depthLevels) {
    const selected = new Set();
    for (const targetDepth of COMMON_SELECTOR_DEPTHS_M) {
      let best = null;
      depthLevels.forEach((depth, index) => {
        const actualDepth = Number(depth.actual_depth_m);
        if (!Number.isFinite(actualDepth)) {
          return;
        }
        const distance = Math.abs(actualDepth - targetDepth);
        if (best === null || distance < best.distance) {
          best = { index, distance };
        }
      });
      if (best !== null) {
        selected.add(best.index);
      }
    }
    return selected;
  }

  function selectedDepthPointIndex(depthLevels = chartDepthLevels()) {
    const depth = activeDepth();
    if (depth.is_aggregate) {
      return -1;
    }
    return depthLevels.indexOf(depth);
  }

  function primaryMarkerValues(depthLevels, normalValue, rasterValue, activeValue) {
    const selectedPoint = selectedDepthPointIndex(depthLevels);
    const rasterPoints = commonRasterDepthPointIndices(depthLevels);
    return depthLevels.map((_depth, index) => {
      if (index === selectedPoint) {
        return activeValue;
      }
      return rasterPoints.has(index) ? rasterValue : normalValue;
    });
  }

  function primaryMarkerColors(depthLevels, normalColor) {
    const selectedPoint = selectedDepthPointIndex(depthLevels);
    return depthLevels.map((_depth, index) => (index === selectedPoint ? "#f8f2d8" : normalColor));
  }

  function primaryMarkerLineColors(depthLevels) {
    const selectedPoint = selectedDepthPointIndex(depthLevels);
    const rasterPoints = commonRasterDepthPointIndices(depthLevels);
    return depthLevels.map((_depth, index) => {
      if (index === selectedPoint) {
        return "#ffd166";
      }
      return rasterPoints.has(index) ? "#f8f2d8" : "rgba(248,242,216,0)";
    });
  }

  function primaryMarkerSymbols(depthLevels) {
    const selectedPoint = selectedDepthPointIndex(depthLevels);
    const rasterPoints = commonRasterDepthPointIndices(depthLevels);
    return depthLevels.map((_depth, index) => {
      if (index === selectedPoint) {
        return "diamond";
      }
      return rasterPoints.has(index) ? "circle-open" : "circle";
    });
  }

  function rasterDepthHoverTexts(depthLevels) {
    const rasterPoints = commonRasterDepthPointIndices(depthLevels);
    return depthLevels.map((_depth, index) => (rasterPoints.has(index) ? "<br>Raster export depth" : ""));
  }


  function globalHoverData(depthLevels) {
    const rasterTexts = rasterDepthHoverTexts(depthLevels);
    return depthLevels.map((depth, index) => [
      depthDisplayLabel(depth, data().depth_levels.indexOf(depth)),
      formatMetric(metricValue(depth.global)),
      formatPixelCount(depth.global.count),
      rasterTexts[index],
    ]);
  }

  function basinHoverData(depthLevels, basinName) {
    const rasterTexts = rasterDepthHoverTexts(depthLevels);
    return depthLevels.map((depth, index) => {
      const stats = basinStatsForDepth(depth, basinName) || {};
      return [
        depthDisplayLabel(depth, data().depth_levels.indexOf(depth)),
        formatMetric(metricValue(stats)),
        formatPixelCount(stats.count),
        rasterTexts[index],
      ];
    });
  }

  function selectedHoverData(depthLevels, selectedValues) {
    const rasterTexts = rasterDepthHoverTexts(depthLevels);
    return depthLevels.map((depth, index) => {
      const stats = focusedStats(depth) || {};
      return [
        depthDisplayLabel(depth, data().depth_levels.indexOf(depth)),
        formatMetric(selectedValues[index]),
        formatPixelCount(stats.count),
        rasterTexts[index],
      ];
    });
  }

  function selectDepthFromProfileClick(event, depthLevels) {
    const point = event.points && event.points[0];
    if (!point) {
      return;
    }
    const pointIndex = Number.isInteger(point.pointIndex) ? point.pointIndex : point.pointNumber;
    const clickedDepth = depthLevels[pointIndex];
    const clickedDepthIndex = data().depth_levels.indexOf(clickedDepth);
    if (clickedDepthIndex < 0) {
      return;
    }
    const aggregateIndex = data().depth_levels.findIndex((depth) => depth.is_aggregate);
    state.depthIndex = state.depthIndex === clickedDepthIndex ? Math.max(0, aggregateIndex) : clickedDepthIndex;
    populateDepthSelect();
    render();
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

  function primaryMarker(depthLevels, color) {
    return {
      color: primaryMarkerColors(depthLevels, color),
      line: {
        color: primaryMarkerLineColors(depthLevels),
        width: primaryMarkerValues(depthLevels, 0, 1.5, 2.2),
      },
      size: primaryMarkerValues(depthLevels, 6, 10, 13),
      symbol: primaryMarkerSymbols(depthLevels),
    };
  }

  function basinFanTraces(depthLevels, xValues) {
    if (state.focus.type !== "global" || !state.showBasinFan) {
      return [];
    }
    return displayBasinNames().map((basinName, index) => ({
      customdata: basinHoverData(depthLevels, basinName),
      hovertemplate:
        `${escapeHtml(displayBasinName(basinName))}<br>%{customdata[0]}<br>` +
        `${escapeHtml(metricLabel(state.metric))}: %{customdata[1]}<br>` +
        "Count: %{customdata[2]}%{customdata[3]}<extra></extra>",
      line: { color: basinFanColor(index), width: 1 },
      mode: "lines",
      name: displayBasinName(basinName),
      opacity: 0.36,
      x: xValues,
      y: basinSeries(depthLevels, basinName),
    }));
  }

  function renderDepthProfile() {
    const chart = $("analysis-depth-profile");
    const depthLevels = chartDepthLevels();
    const selected = selectedSeries(depthLevels);
    const linearXValues = depthXValues(depthLevels);
    const xValues = state.depthProfileLogX ? logDepthXValues(linearXValues) : linearXValues;
    const globalFocus = state.focus.type === "global";
    const primaryColor = globalFocus ? "#7cc8ff" : "#ffd166";
    const primaryName = globalFocus ? "Global" : focusLabel();
    updateBasinFanToggle();
    const traces = [
      ...basinFanTraces(depthLevels, xValues),
      {
        customdata: globalFocus ? globalHoverData(depthLevels) : selectedHoverData(depthLevels, selected),
        hovertemplate:
          `${escapeHtml(primaryName)}<br>%{customdata[0]}<br>` +
          `${escapeHtml(metricLabel(state.metric))}: %{customdata[1]}<br>` +
          "Count: %{customdata[2]}%{customdata[3]}<extra></extra>",
        line: { color: globalFocus ? "rgba(124,200,255,0.92)" : "#ffd166", width: 3 },
        marker: primaryMarker(depthLevels, primaryColor),
        mode: "lines+markers",
        name: primaryName,
        x: xValues,
        y: selected,
      },
    ];
    const layout = plotlyLayout(chartAxisTitle());
    layout.hovermode = "closest";
    layout.xaxis.type = state.depthProfileLogX ? "log" : "linear";
    layout.xaxis.title = { text: state.depthProfileLogX ? "Depth (m, log scale)" : "Depth (m)", standoff: 8 };
    const axisRange = visibleDepthAxisRange(traces);
    if (axisRange) {
      layout.xaxis.range = axisRange;
    }
    $("analysis-profile-caption").textContent = `${focusLabel()} ${metricLabel(state.metric).toLowerCase()} absolute error across depth${globalFocus && state.showBasinFan ? " with basin fan" : ""}`;
    Promise.resolve(window.Plotly.react(chart, traces, layout, PLOTLY_CONFIG)).then(() => {
      if (typeof chart.removeAllListeners === "function") {
        chart.removeAllListeners("plotly_click");
      }
      if (typeof chart.on === "function") {
        chart.on("plotly_click", function (event) {
          selectDepthFromProfileClick(event, depthLevels);
        });
      }
    });
  }

  function detailCard(label, value) {
    return `<div class="analysis-detail-card"><span class="analysis-detail-label">${escapeHtml(label)}</span><span class="analysis-detail-value">${escapeHtml(value)}</span></div>`;
  }

  function renderDetailSummary() {
    const depth = activeDepth();
    const stats = focusedStats(depth) || {};
    const details = [
      ["Focus", focusLabel()],
      ["Depth", depthDisplayLabel(depth, state.depthIndex)],
      [metricLabel(state.metric), formatMetric(metricValue(stats))],
      ["Mean", formatMetric(stats.mean)],
      ["P95", formatMetric(stats.p95)],
      ["Count", formatPixelCount(stats.count, true)],
    ];
    $("analysis-detail-caption").textContent = `${focusLabel()} summary at ${depthDisplayLabel(depth, state.depthIndex)}`;
    $("analysis-detail-summary").innerHTML = details.map((item) => detailCard(item[0], item[1])).join("");
  }

  function uncertaintyReliability() {
    return data().uncertainty_reliability || { available: false, reason: "No uncertainty reliability payload was packaged." };
  }

  function uncertaintyValue(value) {
    const formatted = formatNumber(value, 2);
    return formatted === "n/a" ? formatted : `${formatted} ${unitLabel()}`.trim();
  }

  function highlightButton(title, cell) {
    if (!cell) {
      return `<div class="analysis-uncertainty-empty">${escapeHtml(title)}: n/a</div>`;
    }
    const bias = uncertaintyValue(cell.calibration_bias);
    return `<button type="button" class="analysis-highlight-button" data-cell-id="${escapeHtml(cell.id)}" data-label="${escapeHtml(cell.label)}">
      <span><span class="analysis-highlight-title">${escapeHtml(title)}</span><span class="analysis-highlight-name">${escapeHtml(cell.label)}</span></span>
      <span class="analysis-highlight-value">${escapeHtml(bias)}</span>
    </button>`;
  }

  function renderUncertaintyHighlights(reliability) {
    const highlights = reliability.highlights || {};
    const lowUncertaintyHighError = (highlights.low_uncertainty_high_error || [])[0] || null;
    const highUncertaintyLowError = (highlights.high_uncertainty_low_error || [])[0] || null;
    $("analysis-uncertainty-highlights").innerHTML = [
      highlightButton("Low uncertainty / high error", lowUncertaintyHighError),
      highlightButton("High uncertainty / low error", highUncertaintyLowError),
    ].join("");
    document.querySelectorAll(".analysis-highlight-button[data-cell-id]").forEach((button) => {
      button.addEventListener("click", function () {
        state.focus = { type: "cell", id: button.dataset.cellId, label: button.dataset.label };
        render();
      });
    });
  }

  function finiteReliabilityValues(reliability) {
    const values = [];
    for (const cell of reliability.grid_cells || []) {
      for (const key of ["uncertainty_median", "error_median"]) {
        const value = Number(cell[key]);
        if (Number.isFinite(value)) {
          values.push(value);
        }
      }
    }
    return values;
  }

  function renderUncertaintyReliability() {
    const reliability = uncertaintyReliability();
    const chart = $("analysis-uncertainty-chart");
    if (!reliability.available) {
      if (chart && window.Plotly) {
        window.Plotly.purge(chart);
      }
      $("analysis-uncertainty-caption").textContent = reliability.reason || "No uncertainty reliability data available";
      $("analysis-uncertainty-highlights").innerHTML = '<div class="analysis-uncertainty-empty">Run with <code>--export-uncertainty</code> to enable calibration diagnostics.</div>';
      return;
    }

    const cells = (reliability.grid_cells || []).filter(
      (cell) => Number.isFinite(Number(cell.uncertainty_median)) && Number.isFinite(Number(cell.error_median))
    );
    const bins = (reliability.bins || []).filter(
      (bin) => Number.isFinite(Number(bin.uncertainty_mean)) && Number.isFinite(Number(bin.error_mean))
    );
    const maxValue = Math.max(1.0e-6, ...finiteReliabilityValues(reliability));
    const unit = unitLabel();
    const traces = [
      {
        customdata: cells.map((cell) => [cell.id, cell.label, formatPixelCount(cell.count, true), uncertaintyValue(cell.calibration_bias)]),
        hovertemplate:
          "%{customdata[1]}<br>" +
          `Uncertainty: %{x:.2f} ${escapeHtml(unit)}<br>` +
          `Actual error: %{y:.2f} ${escapeHtml(unit)}<br>` +
          "Gap: %{customdata[3]}<br>%{customdata[2]}<extra></extra>",
        marker: {
          color: cells.map((cell) => (Number(cell.calibration_bias) > 0 ? "#ef5b5b" : "#7cc8ff")),
          line: { color: "rgba(223,244,239,0.34)", width: 1 },
          opacity: 0.72,
          size: 8,
        },
        mode: "markers",
        name: "Grid cells",
        type: "scatter",
        x: cells.map((cell) => cell.uncertainty_median),
        y: cells.map((cell) => cell.error_median),
      },
      {
        hovertemplate:
          `Mean uncertainty: %{x:.2f} ${escapeHtml(unit)}<br>` +
          `Mean error: %{y:.2f} ${escapeHtml(unit)}<extra>Binned reliability</extra>`,
        line: { color: "#ffd166", width: 3 },
        marker: { color: "#ffd166", size: 8 },
        mode: "lines+markers",
        name: "Binned reliability",
        type: "scatter",
        x: bins.map((bin) => bin.uncertainty_mean),
        y: bins.map((bin) => bin.error_mean),
      },
      {
        hoverinfo: "skip",
        line: { color: "rgba(215,233,247,0.36)", dash: "dot", width: 1.5 },
        mode: "lines",
        name: "Ideal",
        type: "scatter",
        x: [0, maxValue],
        y: [0, maxValue],
      },
    ];
    const layout = plotlyLayout(`Actual error${unit ? ` (${unit})` : ""}`);
    layout.xaxis.title = { text: `Uncertainty std${unit ? ` (${unit})` : ""}`, standoff: 8 };
    layout.xaxis.range = [0, maxValue];
    layout.yaxis.range = [0, maxValue];
    layout.showlegend = true;
    layout.legend = { orientation: "h", x: 0, y: 1.14, xanchor: "left", yanchor: "bottom" };
    $("analysis-uncertainty-caption").textContent = `${reliability.depth_label || "Exported depth"} uncertainty vs realized error | ${formatPixelCount((reliability.global || {}).count, true)}`;
    renderUncertaintyHighlights(reliability);
    Promise.resolve(window.Plotly.react(chart, traces, layout, PLOTLY_CONFIG)).then(() => {
      if (typeof chart.removeAllListeners === "function") {
        chart.removeAllListeners("plotly_click");
      }
      if (typeof chart.on === "function") {
        chart.on("plotly_click", function (event) {
          const point = event.points && event.points[0];
          if (!point || !point.customdata || !point.customdata[0]) {
            return;
          }
          state.focus = { type: "cell", id: point.customdata[0], label: point.customdata[1] };
          render();
        });
      }
    });
  }

  function resizeVisuals() {
    if (state.map) {
      state.map.invalidateSize(false);
    }
    ["analysis-depth-profile", "analysis-uncertainty-chart"].forEach((id) => {
      const chart = $(id);
      if (chart && window.Plotly) {
        window.Plotly.Plots.resize(chart);
      }
    });
  }

  function render() {
    renderRankings();
    renderMap();
    renderDepthProfile();
    renderUncertaintyReliability();
    renderDetailSummary();
    $("analysis-selection-pill").textContent = focusLabel();
    const mapGeometryLabel = activeGridGeometryIndex() ? "coast-clipped " : "";
    $("analysis-map-caption").textContent = `${depthDisplayLabel(activeDepth(), state.depthIndex)} ${metricLabel(state.metric).toLowerCase()} absolute error by ${mapGeometryLabel}${data().grouping.grid_size_degrees} degree cell`;
  }

  function analysisSourcesFromConfig(config) {
    const variables = config.variables && typeof config.variables === "object" ? config.variables : null;
    if (variables) {
      return Object.entries(variables)
        .map(([key, variableConfig]) => ({
          key,
          label: variableConfig.variable_label || variableConfig.variable || key,
          url: variableConfig.error_analysis_data_url,
          gridUrl: variableConfig.analysis_grid_geojson_url || config.analysis_grid_geojson_url,
        }))
        .filter((item) => item.url);
    }
    if (config.error_analysis_data_url) {
      return [
        {
          key: config.variable || "default",
          label: config.variable_label || config.variable || "Default",
          url: config.error_analysis_data_url,
          gridUrl: config.analysis_grid_geojson_url,
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

  async function fetchOptionalJson(url) {
    if (!url) {
      return null;
    }
    try {
      return await fetchJson(url);
    } catch (error) {
      console.warn(`Optional dashboard asset failed to load: ${url}`, error);
      return null;
    }
  }

  async function loadAllAnalysisData() {
    const config = await fetchJson(DEFAULT_GLOBE_CONFIG_URL);
    const sources = analysisSourcesFromConfig(config);
    if (sources.length === 0) {
      throw new Error("globe-config.json does not list any error analysis datasets");
    }
    const loaded = await Promise.all(
      sources.map(async (source) => {
        const payloadUrl = new URL(source.url, DEFAULT_GLOBE_CONFIG_URL).toString();
        const gridUrl = source.gridUrl ? new URL(source.gridUrl, DEFAULT_GLOBE_CONFIG_URL).toString() : null;
        const [payload, gridPayload] = await Promise.all([fetchJson(payloadUrl), fetchOptionalJson(gridUrl)]);
        validateAnalysisPayload(payload);
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
    setupDashboardSelect();
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
