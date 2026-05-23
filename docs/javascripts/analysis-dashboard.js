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
  const CONTINENT_OUTLINES = [
    {
      name: "North America",
      points: [
        [-168, 72],
        [-140, 70],
        [-125, 56],
        [-124, 49],
        [-110, 48],
        [-97, 50],
        [-82, 46],
        [-66, 45],
        [-52, 48],
        [-59, 58],
        [-76, 63],
        [-86, 72],
        [-112, 74],
        [-135, 70],
        [-168, 72],
      ],
    },
    {
      name: "Central America",
      points: [
        [-117, 32],
        [-105, 24],
        [-96, 19],
        [-88, 18],
        [-83, 10],
        [-78, 9],
        [-77, 18],
        [-90, 22],
        [-99, 26],
        [-117, 32],
      ],
    },
    {
      name: "South America",
      points: [
        [-81, 12],
        [-72, 8],
        [-67, 3],
        [-76, -12],
        [-71, -30],
        [-74, -53],
        [-63, -55],
        [-51, -36],
        [-39, -20],
        [-35, -7],
        [-48, 2],
        [-61, 8],
        [-81, 12],
      ],
    },
    {
      name: "Greenland",
      points: [
        [-52, 60],
        [-39, 64],
        [-22, 70],
        [-29, 81],
        [-48, 83],
        [-65, 76],
        [-72, 67],
        [-52, 60],
      ],
    },
    {
      name: "Africa",
      points: [
        [-17, 37],
        [1, 36],
        [15, 33],
        [32, 31],
        [43, 12],
        [51, 11],
        [43, -12],
        [33, -27],
        [20, -35],
        [12, -29],
        [3, -5],
        [-10, 6],
        [-17, 22],
        [-17, 37],
      ],
    },
    {
      name: "Eurasia",
      points: [
        [-10, 36],
        [-6, 50],
        [12, 58],
        [30, 70],
        [62, 72],
        [94, 74],
        [132, 60],
        [164, 56],
        [180, 66],
        [180, 24],
        [121, 4],
        [105, -6],
        [82, 8],
        [65, 24],
        [45, 12],
        [36, 31],
        [20, 41],
        [5, 43],
        [-10, 36],
      ],
    },
    {
      name: "Australia",
      points: [
        [113, -11],
        [126, -13],
        [144, -11],
        [154, -27],
        [146, -39],
        [130, -35],
        [115, -34],
        [112, -22],
        [113, -11],
      ],
    },
    {
      name: "Antarctica",
      points: [
        [-180, -70],
        [-130, -73],
        [-70, -71],
        [-20, -75],
        [40, -70],
        [100, -74],
        [160, -70],
        [180, -72],
      ],
      closed: false,
    },
  ];
  const state = {
    datasets: {},
    variables: [],
    activeVariable: null,
    depthIndex: 0,
    metric: "median",
    focus: { type: "global", id: "global", label: "Global" },
    hitCells: [],
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

  function data() {
    return state.datasets[state.activeVariable];
  }

  function setControlsDisabled(disabled) {
    document
      .querySelectorAll(
        "#analysis-modality-select, #analysis-depth-select, #analysis-metric-toggle button"
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
      if (element) {
        element.innerHTML = "";
      }
    });
    const canvas = $("analysis-map");
    if (canvas) {
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
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
      "<p>The page is working, but the packaged globe manifest or analysis JSON files could not be loaded.</p>",
      `<p><strong>Manifest URL:</strong> <code>${escapeHtml(sourceUrl)}</code></p>`,
      `<p><strong>Error:</strong> ${escapeHtml(error && error.message ? error.message : error)}</p>`,
      "<p>Run the normal globe packaging/export command. It writes <code>globe-config.json</code> plus <code>error-analysis.json</code> files for every packaged modality.</p>",
    ].join("");
    document.querySelector(".analysis-shell").appendChild(panel);
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
      .map((variable) => `<option value="${variable.key}">${variable.label}</option>`)
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
      .map((depth, index) => `<option value="${index}">${depth.label}</option>`)
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
          `<button type="button" data-metric="${metric}" aria-pressed="${metric === state.metric}">${metricLabel(metric)}</button>`
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
    if (lonValue >= 20 && lonValue < 147 && latValue > -60 && latValue < 32) {
      return "Indian";
    }
    if ((lonValue >= -70 && lonValue < 20 && latValue > -60 && latValue < 66) || (lonValue >= 147 && latValue >= 50)) {
      return "Atlantic";
    }
    if (lonValue >= -180 && lonValue < 180 && latValue > -60 && latValue < 66) {
      return "Pacific";
    }
    return "Other";
  }

  function basinForCell(cell) {
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
      ["Active Depth", depth.label, `${formatNumber(depth.actual_depth_m, 1)} m actual`],
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

  function colorFor(value, max) {
    const t = Math.max(0, Math.min(1, Number(value || 0) / Math.max(1e-9, max)));
    const stops = [
      [73, 209, 125],
      [242, 196, 81],
      [237, 91, 91],
    ];
    const lower = t < 0.5 ? stops[0] : stops[1];
    const upper = t < 0.5 ? stops[1] : stops[2];
    const local = t < 0.5 ? t * 2 : (t - 0.5) * 2;
    const rgb = lower.map((channel, index) => Math.round(channel + (upper[index] - channel) * local));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  function renderMap() {
    const canvas = $("analysis-map");
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#07100f";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawMapGrid(ctx, rect.width, rect.height);

    const cells = activeDepth().grid_cells.filter((cell) => cell[state.metric] !== null);
    const max = Math.max(...cells.map((cell) => Number(cell[state.metric] || 0)), 1);
    renderMapLegend(max);
    state.hitCells = [];
    for (const cell of cells) {
      const x = ((cell.west + 180) / 360) * rect.width;
      const y = ((90 - cell.north) / 180) * rect.height;
      const w = ((cell.east - cell.west) / 360) * rect.width;
      const h = ((cell.north - cell.south) / 180) * rect.height;
      const basin = basinForCell(cell);
      const basinIsActive = state.focus.type === "basin" && state.focus.id === basin;
      const cellIsActive = state.focus.type === "cell" && state.focus.id === cell.id;
      ctx.globalAlpha = 0.9;
      if (state.focus.type === "basin") {
        ctx.globalAlpha = basinIsActive ? 0.98 : 0.24;
      } else if (state.focus.type === "cell") {
        ctx.globalAlpha = cellIsActive ? 0.98 : 0.34;
      }
      ctx.fillStyle = colorFor(cell[state.metric], max);
      ctx.fillRect(x, y, Math.max(1, w), Math.max(1, h));
      if (basinIsActive || cellIsActive) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = cellIsActive ? "#edf8f4" : "rgba(255,209,102,0.92)";
        ctx.lineWidth = cellIsActive ? 2 : 1.1;
        ctx.strokeRect(x, y, Math.max(1, w), Math.max(1, h));
      }
      state.hitCells.push({ x, y, w: Math.max(1, w), h: Math.max(1, h), cell });
    }
    ctx.globalAlpha = 1;
    drawContinentOutlines(ctx, rect.width, rect.height);
  }

  function renderMapLegend(max) {
    const legendMax = $("analysis-map-legend-max");
    if (legendMax) {
      legendMax.textContent = `Higher error (${formatMetric(max)})`;
    }
  }

  function projectMapPoint(lon, lat, width, height) {
    return [((lon + 180) / 360) * width, ((90 - lat) / 180) * height];
  }

  function drawContinentOutlines(ctx, width, height) {
    ctx.save();
    ctx.fillStyle = "rgba(4,19,31,0.2)";
    ctx.strokeStyle = "rgba(237,248,244,0.8)";
    ctx.lineWidth = 1.35;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (const outline of CONTINENT_OUTLINES) {
      ctx.beginPath();
      outline.points.forEach(([lon, lat], index) => {
        const [x, y] = projectMapPoint(lon, lat, width, height);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      if (outline.closed !== false) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawMapGrid(ctx, width, height) {
    ctx.strokeStyle = "rgba(223,244,239,0.12)";
    ctx.lineWidth = 1;
    for (let lon = -180; lon <= 180; lon += 30) {
      const x = ((lon + 180) / 360) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let lat = -60; lat <= 60; lat += 30) {
      const y = ((90 - lat) / 180) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(223,244,239,0.34)";
    ctx.font = "12px Roboto, sans-serif";
    ctx.fillText("180 W", 10, height - 10);
    ctx.fillText("0", width / 2 - 4, height - 10);
    ctx.fillText("180 E", width - 48, height - 10);
  }

  function positionTooltip(tooltip, event) {
    const margin = 8;
    const offset = 8;
    const containerRect = tooltip.parentElement.getBoundingClientRect();
    const tooltipWidth = tooltip.offsetWidth || 0;
    const tooltipHeight = tooltip.offsetHeight || 0;
    const left = Math.min(event.clientX - containerRect.left + offset, containerRect.width - tooltipWidth - margin);
    const top = Math.min(event.clientY - containerRect.top + offset, containerRect.height - tooltipHeight - margin);
    tooltip.style.left = `${Math.max(margin, left)}px`;
    tooltip.style.top = `${Math.max(margin, top)}px`;
  }

  function setupMapEvents() {
    const canvas = $("analysis-map");
    const tooltip = $("analysis-map-tooltip");
    canvas.addEventListener("mousemove", function (event) {
      const box = canvas.getBoundingClientRect();
      const x = event.clientX - box.left;
      const y = event.clientY - box.top;
      const hit = state.hitCells.find((item) => x >= item.x && x <= item.x + item.w && y >= item.y && y <= item.y + item.h);
      if (!hit) {
        tooltip.hidden = true;
        return;
      }
      tooltip.innerHTML = `<strong>${escapeHtml(hit.cell.label)}</strong><br>Basin: ${escapeHtml(displayBasinName(basinForCell(hit.cell)))}<br>${metricLabel(state.metric)}: ${formatMetric(hit.cell[state.metric])}<br>P95: ${formatMetric(hit.cell.p95)}<br>Count: ${formatPixelCount(hit.cell.count)}`;
      tooltip.hidden = false;
      positionTooltip(tooltip, event);
    });
    canvas.addEventListener("mouseleave", function () {
      tooltip.hidden = true;
    });
    canvas.addEventListener("click", function (event) {
      const box = canvas.getBoundingClientRect();
      const x = event.clientX - box.left;
      const y = event.clientY - box.top;
      const hit = state.hitCells.find((item) => x >= item.x && x <= item.x + item.w && y >= item.y && y <= item.y + item.h);
      if (hit) {
        state.focus = { type: "cell", id: hit.cell.id, label: hit.cell.label };
        render();
      }
    });
  }

  function selectedSeries() {
    return data().depth_levels.map((depth) => {
      if (state.focus.type === "basin") {
        return (depth.basins.find((basin) => basin.name === state.focus.id) || {})[state.metric] ?? null;
      }
      if (state.focus.type === "cell") {
        return (depth.grid_cells.find((cell) => cell.id === state.focus.id) || {})[state.metric] ?? null;
      }
      return depth.global[state.metric] ?? null;
    });
  }

  function globalSeries() {
    return data().depth_levels.map((depth) => depth.global[state.metric] ?? null);
  }

  function renderDepthProfile() {
    const labels = data().depth_levels.map((depth) => depth.label);
    const selected = selectedSeries();
    const global = globalSeries();
    const seriesList = state.focus.type === "global" ? [global] : [global, selected];
    const seriesNames = state.focus.type === "global" ? ["Global"] : ["Global", focusLabel()];
    $("analysis-profile-caption").textContent = `${focusLabel()} ${metricLabel(state.metric).toLowerCase()} absolute error across depth`;
    drawLineChart($("analysis-depth-profile"), seriesList, labels, seriesNames);
  }

  function renderBasinChart() {
    const basins = activeDepth().basins.filter((basin) => basin[state.metric] !== null);
    drawBarChart(
      $("analysis-basin-chart"),
      basins.map((basin) => basin[state.metric]),
      basins.map((basin) => displayBasinName(basin.name)),
      basins.map((basin) => basin.name)
    );
  }

  function chartAxisTitle() {
    const unit = unitLabel();
    return `${metricLabel(state.metric)} absolute error${unit ? ` (${unit})` : ""}`;
  }

  function chartTicks(max) {
    const top = Number.isFinite(Number(max)) && Number(max) > 0 ? Number(max) : 1;
    return [0, 0.25, 0.5, 0.75, 1].map((fraction) => top * fraction);
  }

  function drawLineChart(svg, seriesList, labels, seriesNames) {
    const width = svg.clientWidth || 700;
    const height = 300;
    const pad = { left: 68, right: 20, top: 26, bottom: 42 };
    const values = seriesList
      .flat()
      .filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value)))
      .map(Number);
    const max = Math.max(...values, 1);
    const xFor = (index) => pad.left + index * ((width - pad.left - pad.right) / Math.max(1, labels.length - 1));
    const yFor = (value) => height - pad.bottom - (Number(value) / max) * (height - pad.top - pad.bottom);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<line class="analysis-axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"/><line class="analysis-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"/><text class="analysis-axis-label" transform="translate(14 ${height / 2}) rotate(-90)" text-anchor="middle">${escapeHtml(chartAxisTitle())}</text>`;
    chartTicks(max).forEach((tick) => {
      const y = yFor(tick);
      svg.insertAdjacentHTML("beforeend", `<line class="analysis-gridline" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}"/><text class="analysis-axis-tick" x="${pad.left - 8}" y="${y + 3}" text-anchor="end">${formatNumber(tick, 2)}</text>`);
    });
    seriesList.forEach((series, seriesIndex) => {
      const points = series.map((value, index) => (value === null || value === undefined ? null : [xFor(index), yFor(value), value]));
      const path = points.filter(Boolean).map((point, index) => `${index ? "L" : "M"}${point[0]},${point[1]}`).join(" ");
      if (path) {
        svg.insertAdjacentHTML("beforeend", `<path class="analysis-line ${seriesIndex === 0 ? "analysis-line--global" : ""}" d="${path}"/>`);
      }
      points.forEach((point, index) => {
        if (!point) {
          return;
        }
        const title = `${seriesNames[seriesIndex]} ${labels[index]}: ${formatMetric(point[2])}`;
        const pointClass = seriesIndex === 0 ? "analysis-point analysis-point--global" : "analysis-point";
        svg.insertAdjacentHTML("beforeend", `<circle class="${pointClass}" cx="${point[0]}" cy="${point[1]}" r="4"><title>${escapeHtml(title)}</title></circle><circle class="analysis-point-hit" cx="${point[0]}" cy="${point[1]}" r="10"><title>${escapeHtml(title)}</title></circle>`);
      });
    });
    labels.forEach((label, index) => {
      if (index === 0 || index === labels.length - 1 || index % 2 === 0) {
        svg.insertAdjacentHTML("beforeend", `<text class="analysis-axis-tick" x="${xFor(index)}" y="${height - 10}" text-anchor="middle">${escapeHtml(label)}</text>`);
      }
    });
  }

  function drawBarChart(svg, values, labels, basinIds) {
    const width = svg.clientWidth || 520;
    const height = 300;
    const pad = { left: 70, right: 16, top: 24, bottom: 86 };
    const finiteValues = values.filter((value) => Number.isFinite(Number(value))).map(Number);
    const max = Math.max(...finiteValues, 1);
    const barWidth = (width - pad.left - pad.right) / Math.max(1, values.length);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<line class="analysis-axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"/><line class="analysis-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"/><text class="analysis-axis-label" transform="translate(14 ${height / 2}) rotate(-90)" text-anchor="middle">${escapeHtml(chartAxisTitle())}</text>`;
    const yFor = (value) => height - pad.bottom - (Number(value) / max) * (height - pad.top - pad.bottom);
    chartTicks(max).forEach((tick) => {
      const y = yFor(tick);
      svg.insertAdjacentHTML("beforeend", `<line class="analysis-gridline" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}"/><text class="analysis-axis-tick" x="${pad.left - 8}" y="${y + 3}" text-anchor="end">${formatNumber(tick, 2)}</text>`);
    });
    values.forEach((value, index) => {
      const barHeight = (Number(value) / max) * (height - pad.top - pad.bottom);
      const x = pad.left + index * barWidth + 6;
      const y = height - pad.bottom - barHeight;
      const innerWidth = Math.max(6, barWidth - 12);
      const active = state.focus.type === "basin" && state.focus.id === basinIds[index];
      svg.insertAdjacentHTML("beforeend", `<rect class="analysis-bar ${active ? "is-active" : ""}" data-basin-id="${escapeHtml(basinIds[index])}" data-basin-label="${escapeHtml(labels[index])}" x="${x}" y="${y}" width="${innerWidth}" height="${barHeight}"><title>${escapeHtml(labels[index])}: ${formatMetric(value)}</title></rect><text class="analysis-axis-tick" transform="translate(${x + innerWidth / 2} ${height - 18}) rotate(-35)" text-anchor="end">${escapeHtml(labels[index])}</text>`);
    });
    svg.querySelectorAll("rect[data-basin-id]").forEach((bar) => {
      bar.addEventListener("click", function () {
        state.focus = { type: "basin", id: bar.dataset.basinId, label: bar.dataset.basinLabel };
        render();
      });
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
    $("analysis-basin-caption").textContent = `${metricLabel(state.metric)} absolute error by basin at ${activeDepth().label}`;
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
      await loadAllAnalysisData();
      document.body.classList.remove("analysis-load-failed");
      setRunLabel();
      setupControls();
      setupMapEvents();
      setControlsDisabled(false);
      window.addEventListener("resize", render);
      render();
    } catch (error) {
      console.error(error);
      renderLoadFailure(error, DEFAULT_GLOBE_CONFIG_URL);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
