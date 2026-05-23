(function () {
  const DEFAULT_GLOBE_CONFIG_URL =
    "https://globe-assets.hyperalislabs.com/inference_production/globe/globe-config.json";
  const METRIC_LABELS = { median: "Median", mean: "Mean", p90: "P90", p95: "P95" };
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
        "#analysis-modality-select, #analysis-depth-select, #analysis-focus-select, #analysis-metric-toggle button"
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
    ["analysis-kpis", "analysis-basin-ranking", "analysis-cell-ranking"].forEach((id) => {
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
    return `${formatNumber(value, 2)} ${unitLabel()}`;
  }

  function formatCount(value) {
    return Number(value || 0).toLocaleString();
  }

  function activeDepth() {
    return data().depth_levels[state.depthIndex];
  }

  function metricLabel(metric) {
    return METRIC_LABELS[metric] || metric;
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

    $("analysis-focus-select").addEventListener("change", function (event) {
      const [type, id] = String(event.target.value).split(":");
      state.focus = focusFromId(type, id);
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

  function focusFromId(type, id) {
    if (type === "basin") {
      return { type, id, label: id };
    }
    if (type === "cell") {
      const cell = activeDepth().grid_cells.find((item) => item.id === id);
      return { type, id, label: cell ? cell.label : id };
    }
    return { type: "global", id: "global", label: "Global" };
  }

  function syncFocusSelect() {
    const select = $("analysis-focus-select");
    const basins = activeDepth().basins.filter((basin) => basin.count > 0);
    const cells = activeDepth().top_cells[state.metric] || [];
    const value = `${state.focus.type}:${state.focus.id}`;
    select.innerHTML = [
      '<option value="global:global">Global</option>',
      ...basins.map((basin) => `<option value="basin:${basin.name}">${basin.name}</option>`),
      ...cells.map((cell) => `<option value="cell:${cell.id}">${cell.label}</option>`),
    ].join("");
    select.value = Array.from(select.options).some((option) => option.value === value)
      ? value
      : "global:global";
    if (select.value === "global:global" && state.focus.type !== "global") {
      state.focus = { type: "global", id: "global", label: "Global" };
    }
  }

  function renderKpis() {
    const depth = activeDepth();
    const global = depth.global;
    const values = [
      ["Active Depth", depth.label, `${formatNumber(depth.actual_depth_m, 1)} m actual`],
      ["Median Error", formatMetric(global.median), `${formatCount(global.count)} valid pixels`],
      ["Mean Error", formatMetric(global.mean), "Accumulated ocean support"],
      ["P95 Error", formatMetric(global.p95), "Tail error pressure"],
      ["Worst Hotspot", formatMetric((depth.top_cells[state.metric] || [])[0]?.[state.metric]), (depth.top_cells[state.metric] || [])[0]?.label || "n/a"],
    ];
    $("analysis-kpis").innerHTML = values
      .map(
        ([label, value, sub]) =>
          `<article class="analysis-kpi"><div class="analysis-kpi__label">${label}</div><div class="analysis-kpi__value">${value}</div><div class="analysis-kpi__sub">${sub}</div></article>`
      )
      .join("");
  }

  function rankingButton(item, type, label) {
    const id = item.id || item.name;
    const active = state.focus.type === type && state.focus.id === id;
    return `<button type="button" class="analysis-ranking-button ${active ? "is-active" : ""}" data-type="${type}" data-id="${id}" data-label="${label}">
      <span><span class="analysis-ranking-name">${label}</span><span class="analysis-ranking-meta">${formatCount(item.count)} pixels</span></span>
      <span class="analysis-ranking-value">${formatNumber(item[state.metric], 2)}</span>
    </button>`;
  }

  function renderRankings() {
    const depth = activeDepth();
    const basins = depth.basins
      .filter((basin) => basin[state.metric] !== null)
      .sort((a, b) => Number(b[state.metric]) - Number(a[state.metric]));
    $("analysis-basin-ranking").innerHTML = basins
      .map((basin) => rankingButton(basin, "basin", basin.name))
      .join("");
    $("analysis-cell-ranking").innerHTML = (depth.top_cells[state.metric] || [])
      .map((cell) => rankingButton(cell, "cell", cell.label))
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
    state.hitCells = [];
    for (const cell of cells) {
      const x = ((cell.west + 180) / 360) * rect.width;
      const y = ((90 - cell.north) / 180) * rect.height;
      const w = ((cell.east - cell.west) / 360) * rect.width;
      const h = ((cell.north - cell.south) / 180) * rect.height;
      ctx.globalAlpha = state.focus.type === "cell" && state.focus.id !== cell.id ? 0.38 : 0.9;
      ctx.fillStyle = colorFor(cell[state.metric], max);
      ctx.fillRect(x, y, Math.max(1, w), Math.max(1, h));
      if (state.focus.type === "cell" && state.focus.id === cell.id) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#edf8f4";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, Math.max(1, w), Math.max(1, h));
      }
      state.hitCells.push({ x, y, w: Math.max(1, w), h: Math.max(1, h), cell });
    }
    ctx.globalAlpha = 1;
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
      tooltip.innerHTML = `<strong>${hit.cell.label}</strong><br>${metricLabel(state.metric)}: ${formatMetric(hit.cell[state.metric])}<br>P95: ${formatMetric(hit.cell.p95)}<br>Count: ${formatCount(hit.cell.count)}`;
      tooltip.hidden = false;
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
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
    $("analysis-profile-caption").textContent = `${state.focus.label} ${metricLabel(state.metric).toLowerCase()} error curve`;
    drawLineChart($("analysis-depth-profile"), [global, selected], labels);
  }

  function renderBasinChart() {
    const basins = activeDepth().basins.filter((basin) => basin[state.metric] !== null);
    drawBarChart(
      $("analysis-basin-chart"),
      basins.map((basin) => basin[state.metric]),
      basins.map((basin) => basin.name)
    );
  }

  function drawLineChart(svg, seriesList, labels) {
    const width = svg.clientWidth || 700;
    const height = 260;
    const pad = { left: 42, right: 18, top: 20, bottom: 34 };
    const values = seriesList.flat().filter((value) => value !== null).map(Number);
    const max = Math.max(...values, 1);
    const xFor = (index) => pad.left + index * ((width - pad.left - pad.right) / Math.max(1, labels.length - 1));
    const yFor = (value) => height - pad.bottom - (Number(value) / max) * (height - pad.top - pad.bottom);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<line class="analysis-axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"/><line class="analysis-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"/>`;
    seriesList.forEach((series, seriesIndex) => {
      const points = series.map((value, index) => (value === null ? null : [xFor(index), yFor(value), value]));
      const path = points.filter(Boolean).map((point, index) => `${index ? "L" : "M"}${point[0]},${point[1]}`).join(" ");
      svg.insertAdjacentHTML("beforeend", `<path class="analysis-line ${seriesIndex === 0 ? "analysis-line--global" : ""}" d="${path}"/>`);
      if (seriesIndex === 1) {
        points.forEach((point, index) => {
          if (point) {
            svg.insertAdjacentHTML("beforeend", `<circle class="analysis-point" cx="${point[0]}" cy="${point[1]}" r="4"><title>${labels[index]}: ${formatMetric(point[2])}</title></circle>`);
          }
        });
      }
    });
    labels.forEach((label, index) => {
      if (index === 0 || index === labels.length - 1 || index % 2 === 0) {
        svg.insertAdjacentHTML("beforeend", `<text x="${xFor(index)}" y="${height - 10}" fill="rgba(223,244,239,.62)" font-size="11" text-anchor="middle">${label}</text>`);
      }
    });
  }

  function drawBarChart(svg, values, labels) {
    const width = svg.clientWidth || 520;
    const height = 260;
    const pad = { left: 38, right: 14, top: 16, bottom: 36 };
    const max = Math.max(...values.map(Number), 1);
    const barWidth = (width - pad.left - pad.right) / Math.max(1, values.length);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<line class="analysis-axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"/>`;
    values.forEach((value, index) => {
      const barHeight = (Number(value) / max) * (height - pad.top - pad.bottom);
      const x = pad.left + index * barWidth + 5;
      const y = height - pad.bottom - barHeight;
      svg.insertAdjacentHTML("beforeend", `<rect class="analysis-bar" x="${x}" y="${y}" width="${Math.max(5, barWidth - 10)}" height="${barHeight}"><title>${labels[index]}: ${formatMetric(value)}</title></rect><text x="${x + Math.max(5, barWidth - 10) / 2}" y="${height - 12}" fill="rgba(223,244,239,.62)" font-size="11" text-anchor="middle">${labels[index].slice(0, 3)}</text>`);
    });
  }

  function render() {
    syncFocusSelect();
    renderKpis();
    renderRankings();
    renderMap();
    renderDepthProfile();
    renderBasinChart();
    $("analysis-selection-pill").textContent = state.focus.label;
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
