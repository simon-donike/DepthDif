/* Saved campaign data only: this page never runs inference or recomputes paper scores. */
"use strict";
(() => {
  const base = "../assets/validation/";
  const $ = (id) => document.getElementById(id);
  const labels = { residual: "DepthDif", glorys: "GLORYS (packaged)", glorys_packaged: "GLORYS (packaged)", climatology: "Climatology", lstm: "LSTM", cnn: "CNN", unet3d: "UNet3D", unet2d: "UNet2D", baseline_oi_climatology_full: "OI · climatology", observation: "Italian XBT observation" };
  labels.idw = "IDW";
  const colors = { residual: "#7de0bc", glorys: "#ffd166", glorys_packaged: "#ffd166", observation: "#f4fbff", climatology: "#8d9aa5", lstm: "#b39bff", cnn: "#ff8c91", unet3d: "#6ca9ff", unet2d: "#e1a4de", baseline_oi_climatology_full: "#b2ce6e" };
  const name = (method) => labels[method] || method;
  const includeMethod = (method) => !/^(glorys_bias|baseline_glorys_bias|baseline_oi_glorys)/.test(method);
  const fmt = (value) => Number.isFinite(value) ? value.toFixed(4) : "—";
  const signed = (value) => Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(4)}` : "—";
  let campaign, selected, request = 0, map, markers = new Map(), mapCohort;
  let spatial, spatialCohort, spatialRequest = 0, stackImages = new Map();
  let missionLayers = [], layerControl, supportLayer, supportMapCohort;
  const missionColors = ["#7de0bc", "#e9a1ff", "#ffba75", "#6ca9ff"];
  let playback, activeHelp, helpTimer;
  const explanations = {
    mae: ["MAE", "Typical error size in °C, ignoring whether the prediction is too warm or too cold. Errors are averaged within each profile first, so every profile counts equally. Lower is better."],
    rmse: ["RMSE", "An error score in °C that gives more weight to large mistakes. Each profile counts equally. Lower is better."],
    bias: ["Signed bias", "Average prediction minus observation. Positive means too warm; negative means too cold. Warm and cold errors can cancel."],
    crps: ["CRPS", "A score for predictions that contain several possible outcomes. This run has only one prediction per case, so CRPS equals MAE and tells us nothing extra about uncertainty."],
    delta: ["Δ MAE vs GLORYS", "The method’s MAE minus original GLORYS MAE. Negative means a smaller error than GLORYS; positive means a larger error. Both use the same observations."],
    weighting: ["Equal profile weighting", "Every profile counts equally, even if one cast contains more depth measurements than another."],
    common: ["Common evaluation support", "Methods are compared at the same observation locations and depths. A method with missing predictions is left out of that comparison."],
    depths: ["Depth bins", "XBT measurements are grouped into 25 m depth bands, such as 0–25 m and 25–50 m. Each point on the curve represents one band. EN4 is shown at exact model depth levels. Fewer casts reach the deepest depths."],
    profiles: ["Profiles and points", "One profile is one cast measuring temperature at many depths. A mission or cruise contains many casts. Measurement counts are not counts of independent missions."],
    residual: ["DepthDif", "DepthDif predicts how temperature differs from the usual seasonal background, then adds that background back. The model weights are fixed for this comparison."],
    glorys: ["GLORYS (packaged)", "The original GLORYS ocean reanalysis, as stored in our dataset. It uses a 0.1° grid and temperature steps of about 0.15 °C. Native GLORYS keeps the original finer grid and numerical precision. No bias correction or optimal interpolation is applied to this reference."],
    climatology: ["Climatology", "The usual temperature for that month, location and depth, estimated from training data. It is a background reference rather than a prediction of the individual event."],
    idw: ["IDW", "Inverse-distance weighting estimates temperature from nearby EN4 inputs, giving closer locations more weight. It misses some evaluation points in this run, so its images are shown but it is excluded from the main score table."],
    lstm: ["LSTM", "A neural-network baseline that processes temperature information along depth. Its trained weights are fixed during validation."],
    cnn: ["CNN", "A neural-network baseline that uses local patterns to reconstruct temperature. It uses its own fixed checkpoint and input setup."],
    unet3d: ["UNet3D", "A neural-network baseline that learns patterns across depth and both horizontal directions. Its trained weights are fixed during validation."],
    unet2d: ["UNet2D", "A neural-network baseline that learns horizontal patterns, with depth levels supplied as separate channels. Its trained weights are fixed during validation."],
    baseline_oi_climatology_full: ["OI · climatology", "Optimal interpolation combines the usual seasonal temperature with nearby EN4 observations. Its settings were fixed in advance; it is not fitted to the Italian XBT measurements."],
    conditioning: ["Conditioning", "Information supplied to the prediction: retained EN4 temperature measurements and sea-surface temperature. The Italian XBT casts are used for checking predictions, not as inputs."],
    en4: ["EN4 supports", "EN4 supplies the observed temperature inputs. Markers show occupied input grid cells after quality checks and target masking. Several source profiles can share a cell."],
    xbt: ["Italian XBT", "An expendable bathythermograph measures temperature as it falls through the water. We use the corrected Italian measurements that passed the quality check."],
    masking: ["Holdout buffer", "Evaluation target cells are removed from EN4 inputs in every overlapping patch. For Italian XBT, immediate neighbouring cells are also removed. This helps prevent the model from receiving the answers it is being tested on."],
    screening: ["Archive screening", "The profiles were checked against the documented archive-screened list. That does not prove they were never used by GLORYS; we do not have its assimilation logs."],
    partition: ["Evaluation cohorts", "All 102 Italian casts from two missions are evaluated. EN4 is evaluated separately on a frozen 64-patch preliminary subset. Every selected profile that passes the support checks contributes to its cohort’s results."],
    discovery: ["Discovery campaign", "The 2016 data helped select the model checkpoint. These results describe development performance; an untouched dataset is still needed for independent confirmation."],
    ema: ["EMA weights", "The model uses weights averaged over training steps. This is a fixed trained model, not a correction fitted to the validation observations."],
    ddpm: ["DDPM and one generation", "The model starts from noise and refines it over 1,000 steps. It produces one temperature field per case. One field cannot tell us the spread of possible predictions."],
    sst: ["SST", "Sea-surface temperature is an additional model input. It describes the surface, not a subsurface validation cast."],
    budget: ["Conditioning budget", "The number of EN4 input locations retained per patch: 0, 1, 2, 4, 8, or all available locations. Other inputs, such as sea-surface temperature, remain available."],
    potential: ["Potential temperature / TEOS-10", "Water temperature changes with pressure. We convert the XBT measurements to the same pressure-adjusted temperature convention as the model, using standard seawater calculations and GLORYS salinity."],
    interpolation: ["Native observation depths", "Model values above and below an observation are used to estimate temperature at its exact depth. Profile curves use observation depths; spatial images use model depth levels."],
    clusters: ["Clusters and uncertainty", "Casts from the same mission can share similar errors. The Italian set has 102 casts but only two missions. We show errors for each mission and do not claim reliable uncertainty ranges from so few missions. EN4 uses 5-degree areas within each month as statistical groups; these are not identified platforms or missions."],
    quantization: ["Quantization", "The stored GLORYS temperatures change in steps of about 0.15 °C. That storage rounding is separate from the reanalysis error. A native-data comparison is needed before claiming that a model outperforms the original GLORYS product."],
    patch: ["Patch", "A cropped area of the model grid at one date. All panels show the same area and depth. The selector lists how many XBT casts and EN4 input cells each area contains."],
    scale: ["Colour scale and missing support", "Colours represent temperature. The scale stays fixed while changing depth or model within a patch, but can change between patches. Blank pixels mean missing data, not zero temperature."],
    replay: ["Verified spatial replay", "The images were regenerated with the original inputs, model weights and random seeds. Predictions at the saved observation points were checked against the original run."],
    markers: ["Spatial markers", "Cyan circles mark EN4 temperature input cells. A filled centre means input is available at the displayed depth. Crosses mark XBT casts; red marks the selected cast. A cast need not reach every displayed depth."],
    reference: ["Reference audit", "Checks how the stored GLORYS data are rounded and matched in time. Native GLORYS files are needed for a direct comparison with the original product."],
    comparators: ["Comparator audit", "Checks which baseline predictions are available at the required observations."],
    regional: ["Regional analysis", "Reports errors by area, depth and mission using the same observations for each method."],
    physical: ["Physical diagnostics", "Checks vertical temperature structure and related ocean quantities. It is separate from the main temperature-error comparison."],
    dependence: ["Dependence diagnostics", "Checks relationships across depths, locations or variables. One generated field limits the uncertainty analysis."],
    extremes: ["Extremes", "Checks performance on unusually high or low values. This is separate from the main temperature-error comparison."],
    selective: ["Selective prediction", "Checks whether an uncertainty estimate identifies more reliable predictions. One generated field does not provide a useful ensemble spread."],
    efficiency: ["Efficiency", "Reports computation time and related costs for the saved workload."],
    status: ["Analysis status", "Complete means results are available. Partial means some checks could not be completed. Blocked means the stage is unavailable or not used in this comparison. Pending means no results are published yet."]
  };
  explanations.glorys_packaged = explanations.glorys;
  function closeHelp() {
    clearTimeout(helpTimer);
    if (activeHelp) { activeHelp.removeAttribute("aria-describedby"); activeHelp.setAttribute("aria-expanded", "false"); }
    activeHelp = null;
    $("info-popup").hidden = true;
  }
  function showHelp(button, key) {
    clearTimeout(helpTimer);
    if (activeHelp && activeHelp !== button) closeHelp();
    const popup = $("info-popup"), [title, description] = explanations[key];
    popup.replaceChildren(element("strong", title), element("span", description));
    popup.hidden = false;
    activeHelp = button;
    button.setAttribute("aria-describedby", popup.id);
    button.setAttribute("aria-expanded", "true");
    // A single fixed-position tooltip escapes table overflow and stays within mobile viewports.
    const rect = button.getBoundingClientRect();
    const width = popup.offsetWidth, height = popup.offsetHeight;
    popup.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - width - 8))}px`;
    popup.style.top = `${Math.max(8, rect.bottom + height + 8 < innerHeight ? rect.bottom + 6 : rect.top - height - 6)}px`;
  }
  function infoButton(key) {
    if (!explanations[key]) throw new Error(`Missing explanation: ${key}`);
    const button = element("button", "i", "info-button");
    button.type = "button";
    button.dataset.info = key;
    button.setAttribute("aria-label", `Information: ${explanations[key][0]}`);
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("mouseenter", () => showHelp(button, key));
    button.addEventListener("mouseleave", () => { helpTimer = setTimeout(() => { if (document.activeElement !== button) closeHelp(); }, 150); });
    button.addEventListener("focus", () => showHelp(button, key));
    button.addEventListener("blur", closeHelp);
    // First tap focuses and opens; a subsequent tap dismisses without depending on hover.
    button.addEventListener("pointerdown", () => { button.dataset.wasOpen = String(activeHelp === button && document.activeElement === button); });
    button.addEventListener("click", () => { if (button.dataset.wasOpen === "true") closeHelp(); else showHelp(button, key); });
    return button;
  }
  function explained(text, key) {
    const node = element("span", text);
    node.append(infoButton(key));
    return node;
  }
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
      cells.forEach((cell) => {
        const td = element("td");
        td.append(cell instanceof Node ? cell : document.createTextNode(cell));
        row.append(td);
      });
      return row;
    }));
  }
  function depthChart() {
    const metric = $("metric").value;
    $("metric-help").replaceChildren(infoButton(metric));
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
      missionLayers.forEach((layer) => layer.remove());
      if (supportLayer) supportLayer.remove();
      if (layerControl) layerControl.remove();
      supportLayer = null; supportMapCohort = null; missionLayers = [];
      markers.clear();
      map.fitBounds(selected.profiles.map((profile) => [profile.latitude, profile.longitude]), { padding: [24, 24], maxZoom: 7 });
      const italian = selected.id.startsWith("italian");
      const groupName = (profile) => italian ? profile.mission || profile.cluster : "Screened EN4 evaluation profiles";
      const missions = [...new Set(selected.profiles.map(groupName))];
      const overlays = {};
      missions.forEach((mission, index) => {
        const profiles = selected.profiles.filter((profile) => groupName(profile) === mission);
        const group = L.featureGroup().addTo(map);
        missionLayers.push(group);
        const color = missionColors[index % missionColors.length];
        // Build the label with text nodes before Leaflet uses it as HTML.
        const label = element("span"), swatch = element("span", "● ");
        swatch.style.color = color;
        swatch.setAttribute("aria-hidden", "true");
        label.append(swatch, document.createTextNode(`${mission} · ${profiles.length} casts`));
        overlays[label.innerHTML] = group;
        profiles.forEach((profile) => {
          const marker = L.circleMarker([profile.latitude, profile.longitude], { radius: 5, color: "#e1edf5", weight: 1, fillColor: color, fillOpacity: .85 }).addTo(group);
          marker.bindTooltip(element("span", `${mission} · ${profile.id}`));
          marker.bindPopup(element("span", `${mission} · ${profile.id} · ${profile.time} · ${profile.points} measurements`));
          const choose = () => { $("profile").value = profile.id; profileChart().catch(showError); };
          marker.on("click", choose);
          const node = marker.getElement();
          node.setAttribute("tabindex", "0"); node.setAttribute("role", "button");
          node.setAttribute("aria-label", `Compare profile ${profile.id} from ${mission}`);
          node.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); marker.openPopup(); choose(); } });
          markers.set(profile.id, marker);
        });
      });
      layerControl = L.control.layers(null, overlays, { collapsed: false }).addTo(map);
      $("map-summary").textContent = italian
        ? `${selected.profiles.length} casts across ${missions.length} missions. Each mission has a separate map layer. Click a cast to compare its temperatures.`
        : `${selected.profiles.length} screened EN4 evaluation profiles. Click a location to compare temperatures. Statistical groups are 5-degree areas within each month, not identified missions.`;
      mapCohort = selected.id;
    }
    if (spatial && spatialCohort === selected.id && supportMapCohort !== selected.id) {
      supportLayer = L.featureGroup().addTo(map);
      // These are verified input grid-cell centres, not fabricated source-profile coordinates.
      (spatial.context_patches || spatial.patches).forEach((patch) => {
        const [west, south, east, north] = patch.bounds;
        (patch.en4_supports || []).forEach((support) => {
          const lon = west + (support.col + .5) * (east - west) / patch.width;
          const lat = north - (support.row + .5) * (north - south) / patch.height;
          L.circleMarker([lat, lon], { radius: 3, color: "#00e5ff", weight: 1.5, fillOpacity: 0 }).addTo(supportLayer)
            .bindPopup(element("span", `EN4 temperature input cell · ${patch.date} · patch ${patch.patch_index} · ${support.levels.length} supported depths`));
        });
      });
      layerControl.addOverlay(supportLayer, "EN4 conditioning cells · all saved patches");
      const bounds = L.latLngBounds(selected.profiles.map((profile) => [profile.latitude, profile.longitude]));
      if (supportLayer.getLayers().length) bounds.extend(supportLayer.getBounds());
      map.fitBounds(bounds, { padding: [24, 24] });
      supportMapCohort = selected.id;
    }
    map.invalidateSize();
    markers.forEach((marker, id) => marker.setStyle({ radius: id === $("profile").value ? 8 : 5, weight: id === $("profile").value ? 3 : 1, color: id === $("profile").value ? "#ffd166" : "#e1edf5" }));
  }

  async function profileChart() {
    // Ignore slow responses from a previously selected cast or cohort.
    const token = ++request;
    const cohort = selected;
    const profile = cohort.profiles.find((p) => p.id === $("profile").value);
    if (!profile) return;
    $("profile-meta").textContent = `${profile.id} · ${profile.time} · ${profile.latitude.toFixed(3)}° N, ${profile.longitude.toFixed(3)}° E · ${profile.points.toLocaleString()} depths · ${cohort.id.startsWith("italian") ? "mission" : "observation group"} ${profile.mission || profile.cluster} · evaluation`;
    locationMap();
    if (spatial && spatialCohort === cohort.id) {
      const patch = spatial.patches.find((item) => item.profiles.some((item) => item.profile_id === profile.id));
      if (patch && String(patch.patch_index) !== $("patch").value) { $("patch").value = patch.patch_index; loadPatch().catch(showError); }
      else drawDepth();
    }
    const data = await fetchJSON(`${base}${cohort.id}/${profile.file}`);
    if (token !== request || selected !== cohort) return;
    const methods = Object.keys(data).filter((key) => key !== "depth_m" && includeMethod(key));
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
    const supports = patch.en4_supports || [];
    $("support-count").textContent = `${supports.length} EN4 temperature support cells; ${supports.filter((item) => item.levels.includes(level)).length} valid at this depth · ${patch.profiles.length} XBT evaluation casts`;
    $("spatial-images").querySelectorAll("canvas").forEach((canvas) => {
      const image = stackImages.get(canvas.dataset.method);
      if (!image) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      // Each PNG contains one original-resolution raster per depth, stacked vertically.
      ctx.drawImage(image, 0, level * patch.height, patch.width, patch.height, 0, 0, canvas.width, canvas.height);
      // Show every retained horizontal support, with depth-specific availability inside the ring.
      (patch.en4_supports || []).forEach((support) => {
        const x = (support.col + .5) * canvas.width / patch.width, y = (support.row + .5) * canvas.height / patch.height;
        ctx.lineWidth = 2.5; ctx.strokeStyle = "#00131d";
        ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.stroke();
        ctx.lineWidth = 1.3; ctx.strokeStyle = "#00e5ff"; ctx.stroke();
        if (support.levels.includes(level)) { ctx.fillStyle = "#00e5ff"; ctx.beginPath(); ctx.arc(x, y, 1.5, 0, 2 * Math.PI); ctx.fill(); }
      });
      patch.profiles.forEach((profile) => {
        const x = (profile.col + .5) * canvas.width / patch.width, y = (profile.row + .5) * canvas.height / patch.height;
        ctx.strokeStyle = profile.profile_id === $("profile").value ? "#ff3939" : "white";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y); ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3); ctx.stroke();
      });
      canvas.setAttribute("aria-label", `${name(canvas.dataset.method)}, depth ${spatial.depths_m[level].toFixed(2)} metres, patch ${patch.patch_index}`);
    });
  }
  function pausePlayback() {
    clearInterval(playback);
    playback = null;
    $("depth-play").textContent = "Play";
    $("depth-play").setAttribute("aria-pressed", "false");
  }
  function togglePlayback() {
    if (playback) { pausePlayback(); return; }
    if (!spatial || !stackImages.size) return;
    $("depth-play").textContent = "Pause";
    $("depth-play").setAttribute("aria-pressed", "true");
    playback = setInterval(() => {
      $("depth-level").value = (Number($("depth-level").value) + 1) % spatial.depths_m.length;
      drawDepth();
    }, 500);
  }
  function stepDepth(delta) {
    pausePlayback();
    if (!spatial) return;
    $("depth-level").value = Math.max(0, Math.min(spatial.depths_m.length - 1, Number($("depth-level").value) + delta));
    drawDepth();
  }
  async function loadPatch() {
    pausePlayback();
    $("depth-play").disabled = true;
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
      const caption = element("figcaption");
      caption.append(explained(method === "idw" ? "IDW (incomplete support)" : name(method), method));
      figure.append(caption, canvas);
      $("spatial-images").append(figure);
    });
    $("spatial-status").textContent = `${spatial.patches.length} evaluation patches · ${spatial.depths_m.length} model depths · ${methods.length} fields. Colour range is fixed across methods and depths within the selected patch.`;
    $("depth-play").disabled = false;
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
    $("patch").replaceChildren(...data.patches.map((patch) => { const option = element("option", `${patch.date} · patch ${patch.patch_index} · ${patch.profiles.length} XBT / ${(patch.en4_supports || []).length} EN4`); option.value = patch.patch_index; return option; }));
    // Choose by observation/input coverage, never by prediction accuracy.
    const preferred = data.patches.filter((patch) => patch.profiles.length > 5 && (patch.en4_supports || []).length > 5)
      .sort((a, b) => b.profiles.length - a.profiles.length || b.en4_supports.length - a.en4_supports.length || a.patch_index - b.patch_index)[0] || data.patches[0];
    $("patch").value = preferred.patch_index;
    $("profile").value = preferred.profiles[0].profile_id;
    locationMap();
    $("patch").disabled = false;
    $("depth-level").max = data.depths_m.length - 1;
    $("depth-level").value = 0;
    $("spatial-controls").hidden = false;
    await Promise.all([loadPatch(), profileChart()]);
  }
  async function render() {
    pausePlayback();
    closeHelp();
    ++request;
    ++spatialRequest;
    spatial = null;
    selected = campaign.cohorts.find((cohort) => cohort.id === $("cohort").value);
    $("load-error").hidden = true;
    $("results").hidden = !selected.available;
    $("pending").hidden = selected.available;
    $("status").textContent = selected.pilot ? "Preliminary EN4 subset complete" : selected.evaluation_policy ? "All-profile comparison complete" : selected.available ? `Accuracy available · campaign ${selected.run_status}` : `Results pending · campaign ${selected.run_status}`;
    $("provenance").textContent = JSON.stringify({ campaign: campaign.campaign, snapshot_utc: campaign.updated_utc, model: selected.model || "Pending", membership: selected.membership || "Pending", subset: selected.pilot || null }, null, 2);
    $("stages").replaceChildren(...selected.stages.map((stage) => {
      const node = element("div", undefined, "stage");
      const heading = element("strong");
      heading.append(explained(stage.name, stage.name));
      node.append(heading, element("span", stage.status, stage.status), element("p", stage.summary));
      return node;
    }));
    if (!selected.available) {
      $("pending-counts").textContent = selected.membership ? `${selected.membership.verified_profiles.toLocaleString()} selected profiles passed archive-screened registry membership checks.` : "Waiting for the screened cohort and completed accuracy artifacts.";
      return;
    }
    const model = selected.overall.find((row) => row.method === "residual");
    const glorys = selected.overall.find((row) => row.method === "glorys_packaged");
    const counts = selected.counts;
    $("profile-intro").textContent = selected.id.startsWith("italian")
      ? `All ${counts.profiles} Italian casts from both missions are available. Mission colours and map layers show where they were collected. Click a location to select its cast; curves retain all saved observation depths.`
      : `All ${counts.profiles} evaluated EN4 profiles in the frozen subset are shown. Click a location to compare the prediction, original GLORYS and the held-out EN4 observation at each saved model depth.`;
    $("conditioning-panel").hidden = !selected.conditioning?.length;
    $("spatial-description").hidden = !selected.spatial;
    $("stats").replaceChildren(...[[counts.profiles.toLocaleString(), "evaluation casts"], [counts.clusters.toLocaleString(), "missions / observation groups"], [fmt(model.mae), "DepthDif MAE · °C"], [signed(model.mae - glorys.mae), "MAE difference vs GLORYS · °C"]].map(([value, label]) => {
      const node = element("div", undefined, "stat");
      node.append(element("strong", value), explained(label, label.includes("difference") ? "delta" : label.includes("MAE") ? "mae" : "profiles"));
      return node;
    }));
    $("interpretation").textContent = selected.pilot
      ? `${selected.pilot.patches} of ${selected.pilot.source_patches} frozen EN4 patches were run. ${counts.patches} patches contribute ${counts.profiles} profiles and ${counts.points.toLocaleString()} measurements at ${counts.depths} depths across all 12 months. ${selected.pilot.points_before_support - counts.points} candidate measurements were excluded by time-match, reference or input-support checks before scoring. Selection used geographic and monthly coverage before inference. These preliminary results describe this subset, not the full ocean or all EN4 candidates.`
      : selected.evaluation_policy
      ? `${counts.profiles} casts from ${counts.clusters} missions; all ${counts.points.toLocaleString()} measurements are evaluated. DepthDif MAE is ${Math.abs(model.mae - glorys.mae).toFixed(4)} °C ${model.mae > glorys.mae ? "higher" : "lower"} than original packaged GLORYS. Results are also shown by mission below. Two missions provide limited evidence about performance elsewhere.`
      : `${counts.profiles} profiles from ${counts.clusters} observation groups; ${counts.points.toLocaleString()} measurements evaluated. These are development results.`;
    tableRows("mission-table", (selected.mission_scores || []).map((row) => ({ highlight: row.method === "residual", cells: [row.mission, explained(name(row.method), row.method), fmt(row.mae), fmt(row.rmse)] })));
    $("mission-results").hidden = !selected.mission_scores?.length;
    const missionCSV = selected.downloads?.find((item) => item.name === "mission_scores.csv");
    if (missionCSV) $("mission-csv").href = `${base}${selected.id}/${missionCSV.file}`;
    tableRows("overall-table", selected.overall.map((row) => ({ highlight: row.method === "residual", cells: [explained(name(row.method), row.method), fmt(row.mae), fmt(row.rmse), fmt(row.crps), signed(row.mae - glorys.mae)] })));
    $("overall-csv").href = `${base}${selected.id}/overall_scores.csv`;
    $("paper-download").href = `${base}${selected.id}/paper_artifacts.zip`;
    $("provenance-download").href = `${base}${selected.id}/provenance.json`;
    const depthCSV = selected.downloads.find((item) => item.name === "depth_scores.csv");
    $("depth-download").hidden = !depthCSV;
    if (depthCSV) $("depth-download").href = `${base}${selected.id}/${depthCSV.file}`;
    $("baseline-status").textContent = selected.baselines.filter((row) => row.status !== "available").map((row) => `${name(row.method)}: ${row.status}, ${row.invalid_points} invalid predictions across the full input cohort; excluded from the shared comparison.`).join(" ");
    $("depth-note").textContent = `${selected.id.startsWith("italian") ? "Measurements grouped into 25 m depth bands." : "Exact model depth levels."} Hover to see the number of casts and measurements. Click a legend label to show or hide a method.`;
    tableRows("depth-table", selected.depth.map((row) => ({ highlight: row.method === "residual", cells: [explained(name(row.method), row.method), row.depth_resolution === "bin" ? `${row.depth_lower_m}–${row.depth_upper_m}` : row.depth_m, row.n_profiles, row.n_points, fmt(row.mae), fmt(row.rmse), fmt(row.bias), fmt(row.crps)] })));
    $("method-help").replaceChildren(...[...new Set(selected.overall.map((row) => row.method))].map((method) => explained(name(method), method)));
    $("profile").replaceChildren(...selected.profiles.map((profile) => { const option = element("option", `${profile.mission || profile.cluster} · ${profile.id}`); option.value = profile.id; return option; }));
    // Render only after revealing the panel so Plotly measures the real layout width.
    if (typeof Plotly === "undefined") throw new Error("The chart library could not load. Tables and downloads remain available");
    await Promise.all([depthChart(), conditioningChart(), profileChart(), spatialViewer()]);
  }
  async function start() {
    document.querySelectorAll("[data-explain]").forEach((node) => node.append(infoButton(node.dataset.explain)));
    $("info-popup").addEventListener("mouseenter", () => clearTimeout(helpTimer));
    $("info-popup").addEventListener("mouseleave", closeHelp);
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeHelp(); });
    document.addEventListener("click", (event) => { if (!event.target.closest(".info-button, #info-popup")) closeHelp(); });
    window.addEventListener("resize", closeHelp);
    document.addEventListener("scroll", (event) => {
      if (!activeHelp || event.target === $("info-popup")) return;
      const rect = activeHelp.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > innerHeight) closeHelp();
      else showHelp(activeHelp, activeHelp.dataset.info);
    }, true);
    document.addEventListener("visibilitychange", () => { if (document.hidden) pausePlayback(); });
    window.addEventListener("pagehide", pausePlayback);
    $("depth-play").addEventListener("click", togglePlayback);
    $("map-fit").addEventListener("click", () => {
      const locations = selected.profiles.map((profile) => [profile.latitude, profile.longitude]);
      if (supportLayer) supportLayer.eachLayer((marker) => locations.push(marker.getLatLng()));
      map.fitBounds(locations, { padding: [24, 24] });
    });
    campaign = await fetchJSON(`${base}campaign.json`);
    // Apply the same publication selection to older cached campaign snapshots.
    campaign.cohorts.forEach((cohort) => {
      ["overall", "depth", "paired", "conditioning"].forEach((key) => {
        if (cohort[key]) cohort[key] = cohort[key].filter((row) => includeMethod(row.method ?? row.baseline));
      });
    });
    $("snapshot").textContent = `Snapshot ${new Date(campaign.updated_utc).toLocaleString()} · static snapshot`;
    $("cohort").replaceChildren(...campaign.cohorts.map((cohort) => { const option = element("option", `${cohort.label}${cohort.available ? "" : " · pending"}`); option.value = cohort.id; return option; }));
    const requestedCohort = new URLSearchParams(location.search).get("cohort");
    if (campaign.cohorts.some((cohort) => cohort.id === requestedCohort)) $("cohort").value = requestedCohort;
    $("cohort").disabled = false;
    $("cohort").addEventListener("change", () => render().catch(showError));
    $("metric").addEventListener("change", () => depthChart().catch(showError));
    $("profile").addEventListener("change", () => profileChart().catch(showError));
    $("patch").addEventListener("change", () => loadPatch().catch(showError));
    $("depth-level").addEventListener("input", () => { pausePlayback(); drawDepth(); });
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
