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
  let campaign, selected, request = 0, map, markers = new Map(), mapCohort;
  let spatial, spatialCohort, spatialRequest = 0, stackImages = new Map();
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
  function locationMap() {
    if (typeof L === "undefined") throw new Error("Leaflet could not load");
    if (!map) {
      map = L.map("locations-map", { scrollWheelZoom: false });
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>', maxZoom: 19
      }).addTo(map);
      L.control.scale().addTo(map);
    }
    if (mapCohort !== selected.id) {
      markers.forEach((marker) => marker.remove());
      markers.clear();
      // Leaflet needs an initial view before marker SVG elements are available.
      map.fitBounds(selected.profiles.map((profile) => [profile.latitude, profile.longitude]), { padding: [24, 24], maxZoom: 7 });
      selected.profiles.forEach((profile) => {
        const marker = L.circleMarker([profile.latitude, profile.longitude], { radius: 5, color: "#e1edf5", weight: 1, fillColor: "#7de0bc", fillOpacity: .8 }).addTo(map);
        marker.bindTooltip(element("span", profile.id));
        marker.bindPopup(element("span", `${profile.id} · ${profile.time}`));
        const choose = () => { $("profile").value = profile.id; profileChart().catch(showError); };
        marker.on("click", choose);
        const node = marker.getElement();
        node.setAttribute("tabindex", "0");
        node.setAttribute("role", "button");
        node.setAttribute("aria-label", `Compare profile ${profile.id}`);
        node.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); marker.openPopup(); choose(); } });
        markers.set(profile.id, marker);
      });
      mapCohort = selected.id;
    }
    map.invalidateSize();
    markers.forEach((marker, id) => marker.setStyle({ radius: id === $("profile").value ? 8 : 5, fillColor: id === $("profile").value ? "#ffd166" : "#7de0bc" }));
  }
  async function profileChart() {
    // Ignore slow responses from a previously selected cast or cohort.
    const token = ++request;
    const cohort = selected;
    const profile = cohort.profiles.find((p) => p.id === $("profile").value);
    if (!profile) return;
    $("profile-meta").textContent = `${profile.id} · ${profile.time} · ${profile.latitude.toFixed(3)}° N, ${profile.longitude.toFixed(3)}° E · ${profile.points.toLocaleString()} depths · evaluation only`;
    locationMap();
    if (spatial && spatialCohort === cohort.id) {
      const patch = spatial.patches.find((item) => item.profiles.some((item) => item.profile_id === profile.id));
      if (patch && String(patch.patch_index) !== $("patch").value) { $("patch").value = patch.patch_index; loadPatch().catch(showError); }
      else drawDepth();
    }
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
  function drawDepth() {
    if (!spatial || spatialCohort !== selected.id) return;
    const patch = spatial.patches.find((item) => String(item.patch_index) === $("patch").value);
    if (!patch) return;
    const level = Number($("depth-level").value);
    $("depth-value").textContent = `${spatial.depths_m[level].toFixed(2)} m · level ${level + 1}/${spatial.depths_m.length}`;
    $("depth-previous").disabled = level === 0;
    $("depth-next").disabled = level === spatial.depths_m.length - 1;
    $("spatial-images").querySelectorAll("canvas").forEach((canvas) => {
      const image = stackImages.get(canvas.dataset.method);
      if (!image) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      // Each PNG contains one original-resolution raster per depth, stacked vertically.
      ctx.drawImage(image, 0, level * patch.height, patch.width, patch.height, 0, 0, canvas.width, canvas.height);
      patch.profiles.forEach((profile) => {
        const x = (profile.col + .5) * canvas.width / patch.width, y = (profile.row + .5) * canvas.height / patch.height;
        ctx.strokeStyle = profile.profile_id === $("profile").value ? "#ff3939" : "white";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y); ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3); ctx.stroke();
      });
      canvas.setAttribute("aria-label", `${name(canvas.dataset.method)}, depth ${spatial.depths_m[level].toFixed(2)} metres, patch ${patch.patch_index}`);
    });
  }
  function stepDepth(delta) {
    if (!spatial) return;
    $("depth-level").value = Math.max(0, Math.min(spatial.depths_m.length - 1, Number($("depth-level").value) + delta));
    drawDepth();
  }
  async function loadPatch() {
    const token = ++spatialRequest;
    const patch = spatial.patches.find((item) => String(item.patch_index) === $("patch").value);
    stackImages.clear();
    $("spatial-images").replaceChildren();
    $("spatial-status").textContent = "Loading spatial fields…";
    $("patch-meta").textContent = `Patch ${patch.patch_index} · target date ${patch.date} · ${patch.width} × ${patch.height} cells · bounds W/S/E/N: ${patch.bounds.map((value) => value.toFixed(2)).join(", ")}° · north at top`;
    $("colour-min").textContent = `${patch.limits_c[0].toFixed(2)} °C`;
    $("colour-max").textContent = `${patch.limits_c[1].toFixed(2)} °C`;
    $("colour-bar").src = `${base}${selected.id}/spatial/colour_bar.png`;
    const order = ["residual", "glorys", "climatology", "idw", "lstm", "cnn", "unet3d", "unet2d"];
    const methods = order.filter((method) => patch.methods[method]);
    const loaded = await Promise.all(methods.map(async (method) => {
      const image = new Image();
      image.src = `${base}${selected.id}/spatial/${patch.methods[method]}`;
      await image.decode();
      return [method, image];
    }));
    if (token !== spatialRequest) return;
    loaded.forEach(([method, image]) => {
      stackImages.set(method, image);
      const figure = element("figure"), canvas = element("canvas");
      canvas.width = patch.width * 2; canvas.height = patch.height * 2;
      canvas.dataset.method = method;
      figure.append(element("figcaption", method === "idw" ? "IDW (incomplete support)" : name(method)), canvas);
      $("spatial-images").append(figure);
    });
    $("spatial-status").textContent = `${spatial.patches.length} evaluation patches · ${spatial.depths_m.length} model depths · ${methods.length} fields. Colour range is fixed across methods and depths within the selected patch.`;
    drawDepth();
  }
  async function spatialViewer() {
    const cohort = selected;
    const token = ++spatialRequest;
    spatial = null;
    $("spatial-controls").hidden = true;
    $("patch").disabled = true;
    $("patch").replaceChildren();
    if (!cohort.spatial) { $("spatial-status").textContent = "Spatial images have not been exported for this cohort."; return; }
    const data = await fetchJSON(`${base}${cohort.id}/${cohort.spatial}`);
    if (token !== spatialRequest || selected !== cohort) return;
    spatial = data; spatialCohort = cohort.id;
    $("patch").replaceChildren(...data.patches.map((patch) => { const option = element("option", `${patch.date} · patch ${patch.patch_index}`); option.value = patch.patch_index; return option; }));
    const matching = data.patches.find((patch) => patch.profiles.some((profile) => profile.profile_id === $("profile").value));
    if (matching) $("patch").value = matching.patch_index;
    $("patch").disabled = false;
    $("depth-level").max = data.depths_m.length - 1;
    $("depth-level").value = 0;
    $("spatial-controls").hidden = false;
    await loadPatch();
  }
  async function render() {
    ++request;
    ++spatialRequest;
    spatial = null;
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
    // Render only after revealing the panel so Plotly measures the real layout width.
    if (typeof Plotly === "undefined") throw new Error("The chart library could not load. Tables and downloads remain available");
    await Promise.all([depthChart(), conditioningChart(), profileChart(), spatialViewer()]);
  }
  async function start() {
    campaign = await fetchJSON(`${base}campaign.json`);
    $("snapshot").textContent = `Snapshot ${new Date(campaign.updated_utc).toLocaleString()} · static snapshot`;
    $("cohort").replaceChildren(...campaign.cohorts.map((cohort) => { const option = element("option", `${cohort.label}${cohort.available ? "" : " · pending"}`); option.value = cohort.id; return option; }));
    $("cohort").disabled = false;
    $("cohort").addEventListener("change", () => render().catch(showError));
    $("metric").addEventListener("change", () => depthChart().catch(showError));
    $("profile").addEventListener("change", () => profileChart().catch(showError));
    $("patch").addEventListener("change", () => loadPatch().catch(showError));
    $("depth-level").addEventListener("input", drawDepth);
    $("depth-previous").addEventListener("click", () => stepDepth(-1));
    $("depth-next").addEventListener("click", () => stepDepth(1));
    let lastWheel = 0;
    $("spatial-images").addEventListener("wheel", (event) => {
      if (!spatial || event.ctrlKey || !event.deltaY) return;
      event.preventDefault();
      if (performance.now() - lastWheel > 80) { stepDepth(Math.sign(event.deltaY)); lastWheel = performance.now(); }
    }, { passive: false });
    $("spatial-images").addEventListener("keydown", (event) => {
      if (["ArrowDown", "ArrowRight", "ArrowUp", "ArrowLeft"].includes(event.key)) { event.preventDefault(); stepDepth(["ArrowDown", "ArrowRight"].includes(event.key) ? 1 : -1); }
    });
    await render();
  }
  start().catch(showError);
})();
