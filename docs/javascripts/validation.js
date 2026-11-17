/* Saved campaign data only: this page never runs inference or recomputes paper scores. */
"use strict";
(() => {
  const base = "../assets/validation/";
  const $ = (id) => document.getElementById(id);
  const labels = { residual: "DepthDif", glorys: "GLORYS (packaged)", glorys_packaged: "GLORYS (packaged)", climatology: "Climatology", lstm: "LSTM", cnn: "CNN", unet3d: "UNet3D", unet2d: "UNet2D", glorys_bias: "GLORYS + calibrated bias", baseline_oi_glorys_full: "OI · GLORYS", baseline_oi_climatology_full: "OI · climatology", observation: "Italian XBT observation" };
  const colors = { residual: "#7de0bc", glorys: "#ffd166", glorys_packaged: "#ffd166", observation: "#f4fbff", climatology: "#8d9aa5", lstm: "#b39bff", cnn: "#ff8c91", unet3d: "#6ca9ff", unet2d: "#e1a4de", glorys_bias: "#f59d57", baseline_oi_glorys_full: "#4bc6ca", baseline_oi_climatology_full: "#b2ce6e" };
  const name = (method) => labels[method] || method;
  const fmt = (value) => Number.isFinite(value) ? value.toFixed(4) : "—";
  const signed = (value) => Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(4)}` : "—";
  let campaign, selected, request = 0;
  const config = { responsive: true, displaylogo: false, toImageButtonOptions: { format: "svg", filename: "depthdif-validation" } };
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function layout(x, y, extra = {}) {
    return { paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#c5d7e4", family: "system-ui, sans-serif", size: 12 }, margin: { t: 20, r: 25, b: 85, l: 65 }, xaxis: { title: x, gridcolor: "#203b4c", zerolinecolor: "#516575" }, yaxis: { title: y, gridcolor: "#203b4c", zerolinecolor: "#516575" }, legend: { orientation: "h", y: -0.22, font: { size: 11 } }, hovermode: "closest", ...extra };
  }
  function line(method, x, y, extra = {}) {
    return { type: "scatter", mode: "lines", name: name(method), x, y, line: { color: colors[method], width: method === "residual" ? 3 : 1.8 }, visible: ["residual", "glorys", "glorys_packaged", "observation"].includes(method) ? true : "legendonly", ...extra };
  }
  async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
    return response.json();
  }
  function showError(error) {
    $("load-error").hidden = false;
    $("load-error").textContent = `Results could not be fully displayed: ${error.message}. Reload to retry; available download links remain usable.`;
  }
  function tableRows(id, rows) {
    const body = $(id).querySelector("tbody");
    body.replaceChildren(...rows.map(({ cells, highlight }) => {
      const row = element("tr", undefined, highlight ? "model" : undefined);
      cells.forEach((cell) => row.append(element("td", cell)));
      return row;
    }));
  }
  function depthChart() {
    const metric = $("metric").value;
    const methods = [...new Set(selected.depth.map((row) => row.method))];
    const traces = methods.map((method) => {
      const rows = selected.depth.filter((row) => row.method === method).sort((a, b) => a.depth_m - b.depth_m);
      return line(method, rows.map((row) => row[metric]), rows.map((row) => row.depth_m), { customdata: rows.map((row) => [row.n_profiles, row.n_points, row.depth_lower_m, row.depth_upper_m]), hovertemplate: "%{x:.4f} °C · %{y} m<br>%{customdata[0]} profiles · %{customdata[1]} points<br>Bounds: %{customdata[2]}–%{customdata[3]} m<extra>%{fullData.name}</extra>" });
    });
    return Plotly.react("depth-chart", traces, layout(`${metric.toUpperCase()} (°C)`, "Depth (m)", { yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#203b4c" } }), config);
  }
  function locationChart() {
    const profiles = selected.profiles;
    return Plotly.react("locations-chart", [{ type: "scatter", mode: "markers", x: profiles.map((p) => p.longitude), y: profiles.map((p) => p.latitude), customdata: profiles.map((p) => p.id), text: profiles.map((p) => p.id), hovertemplate: "%{text}<br>%{x:.2f}° E · %{y:.2f}° N<extra></extra>", marker: { size: profiles.map((p) => p.id === $("profile").value ? 12 : 6), color: profiles.map((p) => p.id === $("profile").value ? "#ffd166" : "#7de0bc"), opacity: .85 } }], layout("Longitude (° E)", "Latitude (° N)", { margin: { t: 20, r: 15, b: 55, l: 65 }, showlegend: false }), config);
  }
  async function profileChart() {
    // Ignore slow responses from a previously selected cast or cohort.
    const token = ++request;
    const cohort = selected;
    const profile = cohort.profiles.find((p) => p.id === $("profile").value);
    if (!profile) return;
    $("profile-meta").textContent = `${profile.id} · ${profile.time} · ${profile.latitude.toFixed(3)}° N, ${profile.longitude.toFixed(3)}° E · ${profile.points.toLocaleString()} depths · evaluation only`;
    await locationChart();
    const data = await fetchJSON(`${base}${cohort.id}/${profile.file}`);
    if (token !== request || selected !== cohort) return;
    const methods = Object.keys(data).filter((key) => key !== "depth_m");
    labels.observation = cohort.id.startsWith("italian") ? "Italian XBT observation" : "EN4 observation";
    await Plotly.react("profile-chart", methods.map((method) => line(method, data[method], data.depth_m)), layout("Potential temperature (°C)", "Depth (m)", { yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#203b4c" } }), config);
    await Plotly.react("error-chart", methods.filter((method) => method !== "observation").map((method) => line(method, data.depth_m, data[method].map((value, index) => value - data.observation[index]))), layout("Depth (m)", "Prediction − observation (°C)"), config);
  }
  function conditioningChart() {
    const rows = selected.conditioning;
    const order = ["0", "1", "2", "4", "8", "full"];
    const methods = [...new Set(rows.map((row) => /^\d+$/.test(String(row.method)) || row.method === "full" ? "residual" : row.method))];
    return Plotly.react("conditioning-chart", methods.map((method) => {
      const values = rows.filter((row) => (method === "residual" ? /^\d+$/.test(String(row.method)) || row.method === "full" : row.method === method));
      return line(method, order, order.map((budget) => values.find((row) => String(row.budget) === budget)?.mae ?? null), { mode: "lines+markers", connectgaps: false });
    }), layout("Retained context budget per patch", "MAE (°C)", { xaxis: { title: "Retained context budget per patch", type: "category", categoryorder: "array", categoryarray: order } }), config);
  }
  function downloadLink(label, file) {
    const link = element("a", label);
    link.href = `${base}${selected.id}/${file}`;
    link.setAttribute("download", "");
    return link;
  }
  async function render() {
    ++request;
    selected = campaign.cohorts.find((cohort) => cohort.id === $("cohort").value);
    $("load-error").hidden = true;
    $("results").hidden = !selected.available;
    $("pending").hidden = selected.available;
    $("status").textContent = selected.available ? `Accuracy available · campaign ${selected.run_status}` : `Results pending · campaign ${selected.run_status}`;
    $("provenance").textContent = JSON.stringify({ campaign: campaign.campaign, snapshot_utc: campaign.updated_utc, model: selected.model || "Pending", membership: selected.membership || "Pending" }, null, 2);
    $("stages").replaceChildren(...selected.stages.map((stage) => {
      const node = element("div", undefined, "stage");
      node.append(element("strong", stage.name), element("span", stage.status, stage.status), element("p", stage.summary));
      return node;
    }));
    if (!selected.available) {
      $("pending-counts").textContent = selected.membership ? `${selected.membership.verified_profiles.toLocaleString()} selected profiles passed archive-screened registry membership checks.` : "Waiting for the screened cohort and completed accuracy artifacts.";
      return;
    }
    const model = selected.overall.find((row) => row.method === "residual");
    const glorys = selected.overall.find((row) => row.method === "glorys_packaged");
    const counts = selected.counts;
    $("stats").replaceChildren(...[[counts.profiles.toLocaleString(), "evaluation profiles"], [counts.points.toLocaleString(), "temperature measurements"], [fmt(model.mae), "DepthDif MAE · °C"], [signed(model.mae - glorys.mae), "MAE difference vs GLORYS · °C"]].map(([value, label]) => {
      const node = element("div", undefined, "stat");
      node.append(element("strong", value), element("span", label));
      return node;
    }));
    const calibration = selected.partition.find((row) => row.partition === "calibration");
    $("interpretation").textContent = `${counts.clusters} evaluation cluster${counts.clusters === 1 ? " (one cruise)" : "s"}; ${calibration?.profiles ?? 0} profiles reserved for calibration. DepthDif MAE is ${Math.abs(model.mae - glorys.mae).toFixed(4)} °C ${model.mae > glorys.mae ? "higher" : "lower"} than packaged GLORYS. These are descriptive discovery results${counts.clusters === 1 ? "; one cruise does not support confidence intervals or significance claims" : ""}.`;
    tableRows("overall-table", selected.overall.map((row) => ({ highlight: row.method === "residual", cells: [name(row.method), fmt(row.mae), fmt(row.rmse), fmt(row.crps), signed(row.mae - glorys.mae)] })));
    $("overall-csv").href = `${base}${selected.id}/overall_scores.csv`;
    $("baseline-status").textContent = selected.baselines.filter((row) => row.status !== "available").map((row) => `${name(row.method)}: ${row.status}, ${row.invalid_points} invalid predictions across the full input cohort; excluded from the shared comparison.`).join(" ");
    $("depth-note").textContent = `${selected.id.startsWith("italian") ? "25 m bins, left-closed and right-open, plotted at bin centers." : "Exact model depth levels."} Hover for profile and measurement counts. Click a legend label to show or hide a method. Curves are descriptive; no confidence bands are implied.`;
    tableRows("depth-table", selected.depth.map((row) => ({ highlight: row.method === "residual", cells: [name(row.method), row.depth_resolution === "bin" ? `${row.depth_lower_m}–${row.depth_upper_m}` : row.depth_m, row.n_profiles, row.n_points, fmt(row.mae), fmt(row.rmse), fmt(row.bias), fmt(row.crps)] })));
    $("profile").replaceChildren(...selected.profiles.map((profile) => { const option = element("option", profile.id); option.value = profile.id; return option; }));
    $("paper-pack").href = `${base}${selected.id}/paper_artifacts.zip`;
    $("download-links").replaceChildren(downloadLink("Overall CSV", "overall_scores.csv"), downloadLink("Overall LaTeX", "overall_scores.tex"), downloadLink("Provenance JSON", "provenance.json"), ...selected.downloads.filter((item) => ["depth_scores.csv", "depth_scores.tex", "depth_accuracy_temperature.pdf", "primary_scorecard.csv", "partition_summary.csv"].includes(item.name)).map((item) => downloadLink(item.name, item.file)));
    $("artifact-list").replaceChildren(...selected.downloads.map((item) => downloadLink(item.file.replace("artifacts/", ""), item.file)));
    // Render only after revealing the panel so Plotly measures the real layout width.
    if (typeof Plotly === "undefined") throw new Error("The chart library could not load. Tables and downloads remain available");
    await Promise.all([depthChart(), conditioningChart(), profileChart()]);
    $("locations-chart").removeAllListeners("plotly_click");
    $("locations-chart").on("plotly_click", (event) => { $("profile").value = event.points[0].customdata; profileChart().catch(showError); });
  }
  async function start() {
    campaign = await fetchJSON(`${base}campaign.json`);
    $("snapshot").textContent = `Snapshot ${new Date(campaign.updated_utc).toLocaleString()} · refresh requires a new export`;
    $("cohort").replaceChildren(...campaign.cohorts.map((cohort) => { const option = element("option", `${cohort.label}${cohort.available ? "" : " · pending"}`); option.value = cohort.id; return option; }));
    $("cohort").disabled = false;
    $("cohort").addEventListener("change", () => render().catch(showError));
    $("metric").addEventListener("change", () => depthChart().catch(showError));
    $("profile").addEventListener("change", () => profileChart().catch(showError));
    await render();
  }
  start().catch(showError);
})();
