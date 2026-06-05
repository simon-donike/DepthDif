(function () {
  const DEFAULT_SPECTRAL_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/globe/wavenumber_spectra/spectral-config.json";
  const ALL_OCEANS = "All Oceans";
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
  const DEPTH_PALETTE = ["#7cc8ff", "#ffd166", "#6cc4a1", "#e17c78", "#b39ddb", "#f2a65a", "#88a7ff", "#c7e75f"];
  const SURFACE_REFERENCE_COLOR = "#9aa8b5";
  const DASH_MAP = {
    analysis: "../spatial-dashboard/",
    temporal: "../temporal-dashboard/",
    spectral: "../spectral-dashboard/",
  };
  const state = {
    config: null,
    basinMap: null,
    basinDataCache: {},
    activeVariable: null,
    activeBasin: ALL_OCEANS,
    activePeriodType: "season",
    activePeriodLabel: null,
    activeDepthKey: "__all__",
    activeMetric: "ratio",
    activeXAxisUnit: "cpkm",
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
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
  }

  function configUrl() {
    const params = new URLSearchParams(window.location.search);
    if (params.get("config")) {
      return params.get("config");
    }
    if (window.location.pathname.includes("/spectral-dashboard/")) {
      return DEFAULT_SPECTRAL_CONFIG_URL;
    }
    return new URL("spectral-config.json", window.location.href).toString();
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

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`${url} returned HTTP ${response.status}`);
    }
    return response.json();
  }

  function validateConfig(config) {
    if (!config || Number(config.schema_version) !== 1 || config.kind !== "wavenumber_spectral_dashboard") {
      throw new Error("spectral-config.json is not a spectral dashboard schema v1 payload");
    }
    if (!config.basin_data_urls || !config.basin_map_geojson_url) {
      throw new Error("spectral-config.json is missing basin URLs");
    }
  }

  function setControlsDisabled(disabled) {
    document
      .querySelectorAll(
        "#spectral-dashboard-select, #spectral-variable-select, #spectral-basin-select, #spectral-period-type-select, #spectral-period-label-select, #spectral-depth-select, #spectral-metric-select, #spectral-x-axis-select"
      )
      .forEach((element) => {
        element.disabled = Boolean(disabled);
      });
  }

  function renderLoadFailure(error, sourceUrl) {
    document.body.classList.add("spectral-load-failed");
    setControlsDisabled(true);
    $("spectral-run-label").textContent = "Spectral analysis data is unavailable";
    $("spectral-map-caption").textContent = "No basin data loaded";
    $("spectral-spectrum-caption").textContent = "No spectra loaded";
    $("spectral-bias-caption").textContent = "No bias data loaded";
    $("spectral-selection-pill").textContent = "Unavailable";
    const existing = document.querySelector(".spectral-error-state");
    if (existing) {
      existing.remove();
    }
    const panel = document.createElement("section");
    panel.className = "spectral-error-state";
    panel.innerHTML = [
      "<h2>Could not load the spectral dashboard data</h2>",
      `<p><strong>Manifest URL:</strong> <code>${escapeHtml(sourceUrl)}</code></p>`,
      `<p><strong>Error:</strong> ${escapeHtml(error && error.message ? error.message : error)}</p>`,
    ].join("");
    document.querySelector(".spectral-shell").appendChild(panel);
  }

  function formatNumber(value, digits = 2) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "n/a";
    }
    return number.toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
  }

  function formatCompact(value) {
    const number = Number(value || 0);
    if (!Number.isFinite(number)) {
      return "0";
    }
    if (Math.abs(number) >= 1000000) {
      return `${formatNumber(number / 1000000, 1)}M`;
    }
    if (Math.abs(number) >= 1000) {
      return `${formatNumber(number / 1000, 1)}K`;
    }
    return String(Math.round(number));
  }

  function basinConfig(name) {
    return (state.config.basins || []).find((basin) => basin.name === name) || null;
  }

  function basinLabel(name) {
    const basin = basinConfig(name);
    return (basin && basin.label) || name || ALL_OCEANS;
  }

  function layerLabel(layer, variable) {
    if (layer === "surface_observation") {
      return variable === "salinity" ? "SSS" : "OSTIA";
    }
    return (state.config.layers && state.config.layers[layer]) || layer;
  }

  function lineDash(layer) {
    if (layer === "glorys") {
      return "dot";
    }
    if (layer === "surface_observation") {
      return "dash";
    }
    return "solid";
  }

  function depthKey(row) {
    return `${row.depth_suffix || "depth"}|${row.depth_label || "Depth"}|${Number(row.actual_depth_m)}`;
  }

  function depthLabelFromKey(key) {
    if (key === "__all__") {
      return "All depths";
    }
    const parts = String(key).split("|");
    return parts[1] || String(key);
  }

  function spectrumTraceLabel(row) {
    const layer = layerLabel(row.layer, row.variable);
    const depth = row.depth_label || "";
    if (row.layer === "surface_observation" || String(depth).toLowerCase() === String(layer).toLowerCase()) {
      return layer;
    }
    return `${layer} ${depth}`.trim();
  }

  function spectrumTraceDetail(row) {
    if (row.layer === "surface_observation") {
      return layerLabel(row.layer, row.variable);
    }
    return row.depth_label || "";
  }

  function rowsForActiveBasin() {
    const payload = state.basinDataCache[state.activeBasin];
    return payload && Array.isArray(payload.rows) ? payload.rows : [];
  }

  function filteredRows(options = {}) {
    const includeDepth = options.includeDepth !== false;
    return rowsForActiveBasin().filter((row) => {
      if (row.variable !== state.activeVariable) {
        return false;
      }
      if (row.period_type !== state.activePeriodType) {
        return false;
      }
      if (state.activePeriodLabel && row.period_label !== state.activePeriodLabel) {
        return false;
      }
      if (includeDepth && state.activeDepthKey !== "__all__" && depthKey(row) !== state.activeDepthKey) {
        return false;
      }
      return true;
    });
  }

  function uniqueSorted(values) {
    return Array.from(new Set(values.filter((value) => value !== null && value !== undefined && value !== ""))).sort((a, b) =>
      String(a).localeCompare(String(b), undefined, { numeric: true })
    );
  }

  function availablePeriodTypes() {
    const rows = rowsForActiveBasin().filter((row) => row.variable === state.activeVariable);
    const values = uniqueSorted(rows.map((row) => row.period_type));
    const preferred = ["season", "year", "month"].filter((value) => values.includes(value));
    return preferred.concat(values.filter((value) => !preferred.includes(value)));
  }

  function availablePeriodLabels() {
    return uniqueSorted(
      rowsForActiveBasin()
        .filter((row) => row.variable === state.activeVariable && row.period_type === state.activePeriodType)
        .map((row) => row.period_label)
    );
  }

  function availableDepths() {
    const rows = filteredRows({ includeDepth: false });
    const depthMap = new Map();
    rows.forEach((row) => {
      if (row.layer === "surface_observation") {
        return;
      }
      depthMap.set(depthKey(row), row.depth_label || depthKey(row));
    });
    return Array.from(depthMap.entries()).sort((left, right) => {
      const leftDepth = Number(left[0].split("|")[2]);
      const rightDepth = Number(right[0].split("|")[2]);
      return (Number.isFinite(leftDepth) ? leftDepth : 0) - (Number.isFinite(rightDepth) ? rightDepth : 0);
    });
  }

  function setRunLabel() {
    const summary = state.config.source_summary || {};
    const runCount = Number(summary.run_count || 0);
    const spectra = Number(summary.spectrum_count || 0);
    const variables = (state.config.available_variables || []).join(" / ") || "variables";
    $("spectral-run-label").textContent = `${variables} | ${formatCompact(spectra)} spectra across ${formatCompact(runCount)} runs`;
  }

  function setupDashboardSelect() {
    const dashboardSelect = $("spectral-dashboard-select");
    dashboardSelect.value = "spectral";
    dashboardSelect.addEventListener("change", function () {
      if (dashboardSelect.value !== "spectral") {
        window.location.href = DASH_MAP[dashboardSelect.value] || "../analysis/";
        return;
      }
      dashboardSelect.value = "spectral";
    });
  }

  function refreshSelects() {
    const variableSelect = $("spectral-variable-select");
    variableSelect.innerHTML = (state.config.available_variables || [])
      .map((variable) => `<option value="${escapeHtml(variable)}">${escapeHtml(variable.charAt(0).toUpperCase() + variable.slice(1))}</option>`)
      .join("");
    variableSelect.value = state.activeVariable;

    const basinSelect = $("spectral-basin-select");
    basinSelect.innerHTML = (state.config.basins || [])
      .map((basin) => `<option value="${escapeHtml(basin.name)}">${escapeHtml(basin.label || basin.name)}</option>`)
      .join("");
    basinSelect.value = state.activeBasin;

    const periodTypes = availablePeriodTypes();
    if (!periodTypes.includes(state.activePeriodType)) {
      state.activePeriodType = periodTypes[0] || "year";
    }
    const periodTypeSelect = $("spectral-period-type-select");
    periodTypeSelect.innerHTML = periodTypes.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
    periodTypeSelect.value = state.activePeriodType;

    const periodLabels = availablePeriodLabels();
    if (!periodLabels.includes(state.activePeriodLabel)) {
      state.activePeriodLabel = periodLabels[0] || null;
    }
    const periodLabelSelect = $("spectral-period-label-select");
    periodLabelSelect.innerHTML = periodLabels.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
    periodLabelSelect.value = state.activePeriodLabel || "";

    const depths = availableDepths();
    if (state.activeDepthKey !== "__all__" && !depths.find((item) => item[0] === state.activeDepthKey)) {
      state.activeDepthKey = "__all__";
    }
    const depthSelect = $("spectral-depth-select");
    depthSelect.innerHTML = [`<option value="__all__">All depths</option>`]
      .concat(depths.map(([key, label]) => `<option value="${escapeHtml(key)}">${escapeHtml(label)}</option>`))
      .join("");
    depthSelect.value = state.activeDepthKey;
    $("spectral-metric-select").value = state.activeMetric;
    $("spectral-x-axis-select").value = state.activeXAxisUnit;
  }

  function setupControls() {
    setupDashboardSelect();
    $("spectral-variable-select").addEventListener("change", async function () {
      state.activeVariable = this.value;
      state.activePeriodLabel = null;
      state.activeDepthKey = "__all__";
      await render();
    });
    $("spectral-basin-select").addEventListener("change", async function () {
      await setActiveBasin(this.value);
    });
    $("spectral-period-type-select").addEventListener("change", async function () {
      state.activePeriodType = this.value;
      state.activePeriodLabel = null;
      state.activeDepthKey = "__all__";
      await render();
    });
    $("spectral-period-label-select").addEventListener("change", async function () {
      state.activePeriodLabel = this.value;
      state.activeDepthKey = "__all__";
      await render();
    });
    $("spectral-depth-select").addEventListener("change", async function () {
      state.activeDepthKey = this.value;
      await render();
    });
    $("spectral-metric-select").addEventListener("change", async function () {
      state.activeMetric = this.value;
      await render();
    });
    $("spectral-x-axis-select").addEventListener("change", async function () {
      state.activeXAxisUnit = this.value === "km" ? "km" : "cpkm";
      await render();
    });
  }

  async function loadBasinData(basinName) {
    if (!basinName || state.basinDataCache[basinName]) {
      return;
    }
    const basinUrl = state.config.basin_data_urls[basinName];
    if (!basinUrl) {
      throw new Error(`No basin data URL configured for ${basinName}`);
    }
    state.basinDataCache[basinName] = await fetchJson(new URL(basinUrl, configUrl()).toString());
  }

  async function setActiveBasin(basinName) {
    state.activeBasin = basinName || ALL_OCEANS;
    state.activePeriodLabel = null;
    state.activeDepthKey = "__all__";
    await loadBasinData(state.activeBasin);
    await render();
  }

  function createMap() {
    const map = window.L.map("spectral-map", {
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
      color: active ? cssVar("--spectral-amber", "#ffd166") : "rgba(124, 200, 255, 0.38)",
      fillColor: active ? cssVar("--spectral-teal", "#7cc8ff") : "rgba(124, 200, 255, 0.12)",
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
          className: "spectral-leaflet-tooltip",
          direction: "auto",
          sticky: true,
        });
        layer.on("click", async function (event) {
          if (event.originalEvent) {
            window.L.DomEvent.stop(event.originalEvent);
          }
          await setActiveBasin(state.activeBasin === name ? ALL_OCEANS : name);
        });
      },
    }).addTo(map);
    requestAnimationFrame(() => map.invalidateSize(false));
  }

  function plotlyBaseLayout() {
    return {
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: cssVar("--spectral-text", "#d7e9f7"), family: "Roboto, sans-serif", size: 12 },
      margin: { l: 72, r: 20, t: 18, b: 56 },
      hoverlabel: {
        bgcolor: "#071d2d",
        bordercolor: cssVar("--spectral-border", "rgba(124,200,255,0.32)"),
        font: { color: cssVar("--spectral-text", "#d7e9f7") },
      },
      xaxis: {
        automargin: true,
        color: cssVar("--spectral-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        title: { text: xAxisTitle(), standoff: 10 },
        type: "log",
      },
      yaxis: {
        automargin: true,
        color: cssVar("--spectral-muted", "rgba(215,233,247,0.72)"),
        gridcolor: "rgba(223,244,239,0.1)",
        zerolinecolor: "rgba(223,244,239,0.18)",
      },
      legend: { orientation: "h", x: 0, y: 1.08, font: { size: 10 } },
    };
  }

  function renderEmptyChart(chartId, layout, message) {
    layout.annotations = [
      {
        font: { color: cssVar("--spectral-muted", "#5f6f7c"), size: 13 },
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

  function groupRows(rows, keyFn) {
    const groups = new Map();
    rows.forEach((row) => {
      const key = keyFn(row);
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key).push(row);
    });
    return groups;
  }

  function depthColor(depthKeyValue) {
    const depths = availableDepths().map((item) => item[0]);
    const index = Math.max(0, depths.indexOf(depthKeyValue));
    return DEPTH_PALETTE[index % DEPTH_PALETTE.length];
  }

  function spectrumTraceColor(row) {
    if (row.layer === "surface_observation") {
      return SURFACE_REFERENCE_COLOR;
    }
    return depthColor(depthKey(row));
  }

  function spectralValue(row) {
    const psd = Number(row.psd_mean);
    if (Number.isFinite(psd)) {
      return psd;
    }
    return Number(row.power_mean);
  }

  function sortedFiniteRows(rows) {
    return rows
      .filter((row) => Number.isFinite(Number(row.wavelength_km)) && Number.isFinite(spectralValue(row)) && spectralValue(row) > 0)
      .sort((left, right) => Number(xAxisValue(left.wavelength_km)) - Number(xAxisValue(right.wavelength_km)));
  }

  function horizontalWavenumber(wavelengthKm) {
    const wavelength = Number(wavelengthKm);
    return Number.isFinite(wavelength) && wavelength > 0 ? 1 / wavelength : null;
  }

  function xAxisValue(wavelengthKm) {
    const wavelength = Number(wavelengthKm);
    if (!Number.isFinite(wavelength) || wavelength <= 0) {
      return null;
    }
    return state.activeXAxisUnit === "km" ? wavelength : 1 / wavelength;
  }

  function xAxisTitle() {
    return state.activeXAxisUnit === "km" ? "Wavelength [km]" : "Horizontal wavenumber [cpkm]";
  }

  function xAxisHoverLabel() {
    return state.activeXAxisUnit === "km" ? "Wavelength" : "Horizontal wavenumber";
  }

  function xAxisHoverUnit() {
    return state.activeXAxisUnit === "km" ? "km" : "cpkm";
  }

  function spectralPowerAxisTitle() {
    return `PSD [${spectralPowerUnitLabel()}]`;
  }

  function renderSpectrumChart() {
    const rows = filteredRows().filter((row) => ["prediction", "glorys", "surface_observation"].includes(row.layer));
    const layout = plotlyBaseLayout();
    layout.yaxis.title = { text: spectralPowerAxisTitle(), standoff: 10 };
    layout.yaxis.type = "log";
    if (rows.length === 0) {
      $("spectral-spectrum-caption").textContent = `${basinLabel(state.activeBasin)} has no spectral rows`;
      renderEmptyChart("spectral-spectrum-chart", layout, "No spectral data for the selected filters");
      return;
    }
    const traces = [];
    const groups = groupRows(rows, (row) => `${row.layer}|${depthKey(row)}`);
    groups.forEach((groupRowsValue, key) => {
      const sorted = sortedFiniteRows(groupRowsValue);
      if (sorted.length === 0) {
        return;
      }
      const first = sorted[0];
      const color = spectrumTraceColor(first);
      traces.push({
        customdata: sorted.map((row) => [spectrumTraceLabel(row), spectrumTraceDetail(row), row.spectrum_count || 0, row.wavelength_km]),
        hovertemplate:
          `%{customdata[0]}<br>%{customdata[1]}<br>${xAxisHoverLabel()}: %{x:.4g} ${xAxisHoverUnit()}<br>` +
          "Wavelength: %{customdata[3]:.2f} km<br>PSD: %{y:.4g}<br>Spectra: %{customdata[2]}<extra></extra>",
        line: { color, dash: lineDash(first.layer), width: first.layer === "prediction" ? 2.6 : 2 },
        mode: "lines+markers",
        marker: { size: 4, color },
        name: spectrumTraceLabel(first),
        visible: true,
        x: sorted.map((row) => xAxisValue(row.wavelength_km)),
        y: sorted.map((row) => spectralValue(row)),
      });
    });
    if (traces.length === 0) {
      renderEmptyChart("spectral-spectrum-chart", layout, "No finite spectral power for the selected filters");
      return;
    }
    window.Plotly.react($("spectral-spectrum-chart"), traces, layout, PLOTLY_CONFIG);
    $("spectral-spectrum-caption").textContent = `${basinLabel(state.activeBasin)} ${state.activeVariable} ${state.activePeriodLabel || ""} ${depthLabelFromKey(state.activeDepthKey)}`;
  }

  function meanByWavelength(rows) {
    const values = new Map();
    rows.forEach((row) => {
      const wavelength = Number(row.wavelength_km);
      const power = spectralValue(row);
      if (!Number.isFinite(wavelength) || !Number.isFinite(power)) {
        return;
      }
      const current = values.get(wavelength) || { sum: 0, count: 0 };
      current.sum += power;
      current.count += 1;
      values.set(wavelength, current);
    });
    return values;
  }

  function pairedBiasRows() {
    const rows = filteredRows().filter((row) => row.layer === "prediction" || row.layer === "glorys");
    const depthGroups = groupRows(rows, (row) => depthKey(row));
    const result = [];
    depthGroups.forEach((groupRowsValue, depthKeyValue) => {
      const prediction = meanByWavelength(groupRowsValue.filter((row) => row.layer === "prediction"));
      const glorys = meanByWavelength(groupRowsValue.filter((row) => row.layer === "glorys"));
      const wavelengths = Array.from(prediction.keys()).filter((wavelength) => glorys.has(wavelength)).sort((a, b) => a - b);
      const values = wavelengths
        .map((wavelength) => {
          const pred = prediction.get(wavelength).sum / prediction.get(wavelength).count;
          const truth = glorys.get(wavelength).sum / glorys.get(wavelength).count;
          if (!Number.isFinite(pred) || !Number.isFinite(truth) || truth === 0) {
            return null;
          }
          let value = pred / truth;
          if (state.activeMetric === "relative_bias") {
            value = (pred - truth) / truth;
          } else if (state.activeMetric === "difference") {
            value = pred - truth;
          }
          return { wavelength, xValue: xAxisValue(wavelength), value, pred, truth };
        })
        .filter(Boolean);
      if (values.length > 0) {
        result.push({ depthKeyValue, label: depthLabelFromKey(depthKeyValue), values });
      }
    });
    return result;
  }

  function biasMetricLabel() {
    if (state.activeMetric === "relative_bias") {
      return "Relative bias";
    }
    if (state.activeMetric === "difference") {
      return "PSD difference";
    }
    return "Prediction / GLORYS";
  }

  function spectralPowerUnitLabel() {
    if (state.activeVariable === "temperature") {
      return "degC^2/cpkm";
    }
    if (state.activeVariable === "salinity") {
      return "salinity^2/cpkm";
    }
    return "field^2/cpkm";
  }

  function biasMetricAxisTitle() {
    if (state.activeMetric === "relative_bias") {
      return "Relative bias (fraction)";
    }
    if (state.activeMetric === "difference") {
      return `PSD difference (${spectralPowerUnitLabel()})`;
    }
    return "Prediction / GLORYS (unitless)";
  }

  function renderBiasChart() {
    const layout = plotlyBaseLayout();
    layout.yaxis.title = { text: biasMetricAxisTitle(), standoff: 10 };
    layout.yaxis.type = state.activeMetric === "ratio" ? "log" : "linear";
    if (state.activeMetric === "ratio") {
      layout.yaxis.zeroline = false;
    }
    const groups = pairedBiasRows();
    if (groups.length === 0) {
      $("spectral-bias-caption").textContent = `${basinLabel(state.activeBasin)} has no prediction/GLORYS pair`;
      renderEmptyChart("spectral-bias-chart", layout, "No paired prediction and GLORYS spectra");
      return;
    }
    const traces = groups.map((group) => ({
      customdata: group.values.map((row) => [group.label, row.pred, row.truth, row.wavelength]),
      hovertemplate:
        `%{customdata[0]}<br>${xAxisHoverLabel()}: %{x:.4g} ${xAxisHoverUnit()}<br>Wavelength: %{customdata[3]:.2f} km<br>` +
        `${biasMetricAxisTitle()}: %{y:.4g}` +
        "<br>Prediction: %{customdata[1]:.4g}<br>GLORYS: %{customdata[2]:.4g}<extra></extra>",
      line: { color: depthColor(group.depthKeyValue), width: 2.4 },
      marker: { color: depthColor(group.depthKeyValue), size: 5 },
      mode: "lines+markers",
      name: group.label,
      x: group.values.map((row) => row.xValue),
      y: group.values.map((row) => row.value),
    }));
    window.Plotly.react($("spectral-bias-chart"), traces, layout, PLOTLY_CONFIG);
    $("spectral-bias-caption").textContent = `${basinLabel(state.activeBasin)} ${biasMetricLabel().toLowerCase()} by ${state.activeXAxisUnit === "km" ? "wavelength" : "horizontal wavenumber"}`;
  }

  function averageBiasInRange(minWavelength, maxWavelength) {
    const values = pairedBiasRows()
      .flatMap((group) => group.values)
      .filter((row) => row.wavelength >= minWavelength && row.wavelength <= maxWavelength && Number.isFinite(row.value))
      .map((row) => row.value);
    if (values.length === 0) {
      return null;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function slopeForLayer(layer) {
    const rows = sortedFiniteRows(filteredRows().filter((row) => row.layer === layer));
    if (rows.length < 2) {
      return null;
    }
    const byWavelength = meanByWavelength(rows);
    const points = Array.from(byWavelength.entries())
      .map(([wavelength, value]) => ({ wavelength, power: value.sum / value.count }))
      .filter((row) => row.wavelength > 0 && row.power > 0);
    if (points.length < 2) {
      return null;
    }
    const xs = points.map((row) => Math.log(row.wavelength));
    const ys = points.map((row) => Math.log(row.power));
    const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
    const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length;
    let numerator = 0;
    let denominator = 0;
    xs.forEach((x, index) => {
      numerator += (x - meanX) * (ys[index] - meanY);
      denominator += (x - meanX) * (x - meanX);
    });
    return denominator > 0 ? numerator / denominator : null;
  }

  function renderSummaryCards() {
    const rows = filteredRows();
    const groupCounts = new Map();
    rows.forEach((row) => {
      const key = `${row.layer}|${depthKey(row)}|${row.period_type}|${row.period_label}`;
      if (!groupCounts.has(key)) {
        groupCounts.set(key, Number(row.spectrum_count || 0));
      }
    });
    const spectrumCount = Array.from(groupCounts.values()).reduce((sum, value) => sum + value, 0);
    const edges = (state.config.wavelength_bin_edges_km || []).map(Number).filter(Number.isFinite);
    const minWavelength = edges.length > 0 ? Math.min(...edges) : null;
    const maxWavelength = edges.length > 0 ? Math.max(...edges) : null;
    const highMax = minWavelength !== null ? Math.min(100, maxWavelength || 100) : 100;
    const largeMin = maxWavelength !== null ? Math.max(300, minWavelength || 300) : 300;
    const highBias = minWavelength !== null ? averageBiasInRange(minWavelength, highMax) : null;
    const largeBias = maxWavelength !== null ? averageBiasInRange(largeMin, maxWavelength) : null;
    const predictionSlope = slopeForLayer("prediction");
    const glorysSlope = slopeForLayer("glorys");
    const slopeDifference = predictionSlope !== null && glorysSlope !== null ? predictionSlope - glorysSlope : null;
    const cards = [
      ["Spectra", formatCompact(spectrumCount)],
      ["Wavelengths", minWavelength !== null && maxWavelength !== null ? `${formatNumber(minWavelength, 0)}-${formatNumber(maxWavelength, 0)} km` : "n/a"],
      ["High freq", highBias === null ? "n/a" : formatNumber(highBias, 2)],
      ["Large scale", largeBias === null ? "n/a" : formatNumber(largeBias, 2)],
      ["Slope diff", slopeDifference === null ? "n/a" : formatNumber(slopeDifference, 2)],
    ];
    $("spectral-summary-cards").innerHTML = cards
      .map(
        ([label, value]) =>
          `<div class="spectral-summary-card"><span class="spectral-summary-label">${escapeHtml(label)}</span><span class="spectral-summary-value">${escapeHtml(value)}</span></div>`
      )
      .join("");
    $("spectral-summary-caption").textContent = `${basinLabel(state.activeBasin)} | ${state.activeVariable} | ${state.activePeriodLabel || "no period"}`;
  }

  async function render() {
    setRunLabel();
    refreshSelects();
    $("spectral-selection-pill").textContent = basinLabel(state.activeBasin);
    $("spectral-map-caption").textContent = `${basinLabel(state.activeBasin)} spectral basin`;
    renderMap();
    renderSummaryCards();
    renderSpectrumChart();
    renderBiasChart();
  }

  async function loadDashboard(sourceUrl) {
    state.config = await fetchJson(sourceUrl);
    validateConfig(state.config);
    state.activeVariable =
      (state.config.default_variable && (state.config.available_variables || []).includes(state.config.default_variable) && state.config.default_variable) ||
      ((state.config.available_variables || [])[0] || null);
    const params = new URLSearchParams(window.location.search);
    const requestedBasin = params.get("basin");
    const basinNames = (state.config.basins || []).map((basin) => basin.name);
    state.activeBasin = basinNames.includes(requestedBasin) ? requestedBasin : ALL_OCEANS;
    state.basinMap = await fetchJson(new URL(state.config.basin_map_geojson_url, sourceUrl).toString());
    await loadBasinData(state.activeBasin);
  }

  function resizeVisuals() {
    if (state.map) {
      state.map.invalidateSize(false);
    }
    ["spectral-spectrum-chart", "spectral-bias-chart"].forEach((id) => {
      const chart = $(id);
      if (chart && window.Plotly) {
        window.Plotly.Plots.resize(chart);
      }
    });
  }

  async function init() {
    const sourceUrl = configUrl();
    try {
      requireDashboardLibraries();
      await loadDashboard(sourceUrl);
      setupControls();
      setControlsDisabled(false);
      window.addEventListener("resize", resizeVisuals);
      await render();
    } catch (error) {
      console.error(error);
      renderLoadFailure(error, sourceUrl);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
