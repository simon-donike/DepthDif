/* Saved campaign data only: this page never runs inference or recomputes paper scores. */
"use strict";
(() => {
  const base = "../assets/validation/";
  const $ = (id) => document.getElementById(id);
  const labels = { residual: "DepthDif", glorys: "GLORYS (packaged)", glorys_packaged: "GLORYS (packaged)", climatology: "Climatology", lstm: "LSTM", cnn: "CNN", unet3d: "UNet3D", unet2d: "UNet2D", glorys_bias: "GLORYS + calibrated bias", baseline_oi_glorys_full: "OI · GLORYS", baseline_oi_climatology_full: "OI · climatology", observation: "Italian XBT observation" };
  labels.idw = "IDW";
  const colors = { residual: "#7de0bc", glorys: "#ffd166", glorys_packaged: "#ffd166", observation: "#f4fbff", climatology: "#8d9aa5", lstm: "#b39bff", cnn: "#ff8c91", unet3d: "#6ca9ff", unet2d: "#e1a4de", glorys_bias: "#f59d57", baseline_oi_glorys_full: "#4bc6ca", baseline_oi_climatology_full: "#b2ce6e" };
  const name = (method) => labels[method] || method;
  const fmt = (value) => Number.isFinite(value) ? value.toFixed(4) : "—";
  const signed = (value) => Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(4)}` : "—";
  let campaign, selected, request = 0, map, markers = new Map(), mapCohort;
  let spatial, spatialCohort, spatialRequest = 0, stackImages = new Map();
  let playback, activeHelp, helpTimer;
  const explanations = {
    mae: ["MAE", "Mean absolute error: average |prediction − observation| within each profile, then average across profiles. Units: °C. Smaller values indicate smaller typical errors."],
    rmse: ["RMSE", "Root mean squared error: compute mean squared error within each profile, average across profiles, then take the square root. Units: °C. Large errors receive more weight than in MAE."],
    bias: ["Signed bias", "Average prediction − observation within each profile, then across profiles. Positive values indicate predictions that are too warm; negative values indicate predictions that are too cold. Opposite errors can cancel."],
    crps: ["CRPS", "Continuous ranked probability score compares a predictive distribution with the observation. Smaller is better; units are °C. With this campaign’s single generated member, pointwise CRPS equals absolute error, so its aggregate equals MAE. It does not measure ensemble calibration here."],
    delta: ["Δ MAE vs GLORYS", "Method MAE minus packaged GLORYS MAE on the same evaluation support. Negative values favour the method; positive values favour GLORYS. This descriptive difference alone does not establish statistical significance."],
    weighting: ["Equal profile weighting", "First average errors over the measurements available within each profile (or depth bin), then average those profile scores equally. A densely sampled cast therefore does not outweigh a sparsely sampled one."],
    common: ["Common evaluation support", "Methods in a comparison are evaluated at the same retained observation points. A method lacking complete valid predictions is excluded from that shared comparison rather than receiving a score on easier or different points."],
    depths: ["Depth bins", "Italian XBT observations are grouped into 25 m intervals: the lower boundary is included, the upper boundary excluded. Curves are plotted at bin centres. Profile weights are recalculated within each bin; the set of profiles can change with depth. EN4 uses model depth levels."],
    profiles: ["Profiles and points", "A profile is a temperature-versus-depth cast. Points are individual retained temperature measurements, not independent replicate casts. Several profiles may belong to the same cruise."],
    residual: ["DepthDif", "The frozen diffusion model generates a temperature residual relative to climatology, then adds climatology back to reconstruct the temperature field. This campaign uses EMA weights, SST conditioning and one generated member per case."],
    glorys: ["GLORYS (packaged)", "GLORYS is an ocean reanalysis combining an ocean model with assimilated observations. Packaged means the dataset’s processed GeoTIFF reference: 0.1° grid and about 0.1496 °C temperature increments. Targets are daily means sampled every seven days, not weekly averages. Native GLORYS retains the source 1/12° grid and floating-point precision. Packaging does not change which observations were assimilated."],
    climatology: ["Climatology", "The dataset’s fitted temperature background describes recurring seasonal and spatial structure. It does not reconstruct the individual event from current EN4 context. DepthDif learns departures from this background."],
    idw: ["IDW", "Inverse-distance weighting interpolates retained observations, giving nearby supports greater weight. This run lacks valid predictions at some evaluation points, so IDW is shown descriptively in the spatial viewer and excluded from the complete-support score table."],
    lstm: ["LSTM", "Long short-term memory neural-network baseline. Recurrent processing represents temperature structure along depth. Predictions use the frozen selected checkpoint; this is a deterministic comparator."],
    cnn: ["CNN", "Convolutional neural-network baseline using learned local filters to reconstruct temperature. It uses its own frozen selected checkpoint and configuration; the label does not imply identical inputs or architecture to DepthDif."],
    unet3d: ["UNet3D", "Three-dimensional U-Net baseline with convolutions across depth and both horizontal dimensions. Encoder and decoder features are linked by skip connections. It uses a frozen selected checkpoint."],
    unet2d: ["UNet2D", "Two-dimensional U-Net baseline with spatial convolutions and skip connections between encoder and decoder. Depth levels are represented by channels rather than a third convolution dimension. It uses a frozen selected checkpoint."],
    glorys_bias: ["GLORYS + calibrated bias", "Packaged GLORYS adjusted by a regularised systematic-error correction fitted only on the calibration partition. The correction is evaluated on the separate evaluation partition."],
    baseline_oi_glorys_full: ["OI · GLORYS", "Optimal interpolation updates a GLORYS background using retained EN4 observations and specified covariance and observation-error assumptions. Full denotes all retained context after masking. Evaluation XBT values are not used for the update."],
    baseline_oi_climatology_full: ["OI · climatology", "Optimal interpolation updates a climatological background using retained EN4 observations and specified covariance and observation-error assumptions. Full denotes all retained context after masking. Evaluation XBT values are not used for the update."],
    conditioning: ["Conditioning", "Observations or surface fields supplied as inputs to a prediction. Here the full-context reconstruction uses available quality-controlled EN4 temperature supports and SST. EN4 context may include observations assimilated into GLORYS; it is not the Italian evaluation target."],
    en4: ["EN4 supports", "EN4 is the temperature and salinity profile dataset supplying context. Markers show occupied temperature input grid cells after quality filtering and holdout masking, verified against the original inference batch hashes. Multiple source profiles can share a cell; these are not unique profile identities."],
    xbt: ["Italian XBT", "Expendable bathythermograph casts measure temperature with depth. This evaluation uses corrected Italian XBT data with quality flag QF=1. The casts provide held-out observations, not conditioning inputs."],
    masking: ["Holdout buffer", "All target grid cells and their one-cell neighbourhood are removed from the context inputs, including overlapping patches. This reduces local leakage; complete cruise-level context exclusion cannot be verified from the compact store."],
    screening: ["Archive screening", "Selected profiles were checked against the documented archive-screened registry. No archive match is evidence about those archives, not proof that GLORYS never assimilated the observations. Assimilation-log verification is unavailable."],
    partition: ["Calibration and evaluation", "Complete clusters are assigned to calibration or evaluation. Calibration fits adjustments; evaluation supplies reported errors. Italian data have 14 calibration profiles from one cruise and 88 evaluation profiles from another."],
    discovery: ["Discovery campaign", "The 2016 data informed checkpoint selection. Scores on this year describe performance during development; they are not an independent confirmation of a predeclared claim."],
    ema: ["EMA weights", "Exponential moving average of model weights accumulated during training. The frozen EMA checkpoint is used for inference rather than the last raw training weights."],
    ddpm: ["DDPM and one generation", "Denoising diffusion probabilistic model sampling progressively removes noise over 1,000 steps to generate one temperature field per case. One generation is one ensemble member; it cannot characterize predictive spread or calibration."],
    sst: ["SST", "Sea-surface temperature supplied as an additional conditioning field. It is a surface input, not a subsurface evaluation profile."],
    budget: ["Conditioning budget", "A cap of 0, 1, 2, 4 or 8 retained horizontal EN4 context locations per patch, after holdout masking. Full retains all available locations. This ablates profile context; it does not remove every other input, such as SST or climatology."],
    potential: ["Potential temperature / TEOS-10", "Potential temperature accounts for the temperature change from pressure alone when seawater is brought to a reference pressure. Corrected XBT temperatures are converted with TEOS-10 seawater thermodynamics using GLORYS salinity. Model and observation temperatures must use the same convention."],
    interpolation: ["Native observation depths", "Model temperatures at the two surrounding model levels are linearly interpolated to each retained observation depth before scoring. Profile plots preserve the saved observation depths; spatial images show the original model levels."],
    clusters: ["Clusters and uncertainty", "Profiles collected on the same cruise can share errors and are grouped as a cluster. The Italian evaluation has only one cruise, so between-cruise uncertainty cannot be estimated reliably; confidence intervals and significance claims are withheld."],
    quantization: ["Quantization", "Storing temperatures in discrete increments loses numerical precision. The packaged reference increment is approximately 0.1496 °C, with a half-step rounding bound of 0.0748 °C. This is not GLORYS’s total physical error. Native floating-point verification is needed before claiming superiority over the delivered product."],
    patch: ["Patch", "A cropped model-grid domain at one target date. All spatial panels show that same patch and depth. Overlapping patches may contain the same geographic locations. Only patches containing evaluation casts are displayed."],
    scale: ["Colour scale and missing support", "Within a selected patch, the temperature colour range is fixed across all methods and depths, so colours remain comparable during playback. It can change when selecting another patch. Transparent pixels indicate missing reference or model support, not zero temperature."],
    replay: ["Verified spatial replay", "Full prediction grids were reconstructed using the recorded masked inputs, weights, batch sizes and random seeds. Input hashes and predictions at saved evaluation points were checked against the original run. IDW has no complete valid point comparator."],
    markers: ["Spatial markers", "Cyan circles mark all retained EN4 temperature support cells in the patch; a filled centre means that input is valid at the selected depth. Open circles have support only at other levels. White crosses locate XBT evaluation casts; red marks the selected cast. A cross does not imply a measurement at every model depth."],
    reference: ["Reference audit", "Checks packaged GLORYS precision and temporal matching, with native-product comparison when source NetCDF files are configured. A completed packaged audit does not replace native-product verification."],
    comparators: ["Comparator audit", "Checks availability and compatibility of saved comparator predictions before scoring. Missing or incomplete predictions can limit which methods share a comparison."],
    baselines: ["Calibrated baselines", "Fits permitted corrections or distributions on calibration data, then evaluates them on separate held-out data. Partial status means some requested comparator outputs are unavailable."],
    regional: ["Regional analysis", "Groups paired method errors by region, depth and other predefined strata. Available descriptive scores do not imply sufficient independent clusters for uncertainty or significance estimates."],
    physical: ["Physical diagnostics", "Examines reconstructed vertical structure and derived ocean quantities where temperature, salinity and depth coverage permit. A temperature-only model cannot establish joint temperature–salinity physical accuracy."],
    dependence: ["Dependence diagnostics", "Examines whether relationships across depth, space or variables are retained. A single generated member limits what can be inferred about ensemble dependence."],
    extremes: ["Extremes", "Evaluates predefined tail events and their forecast probabilities where the sample and ensemble support those estimates. One member provides very limited probability resolution."],
    selective: ["Selective prediction", "Examines whether forecast uncertainty can identify more reliable cases when retaining only part of the sample. One generated member does not provide an ensemble-spread estimate."],
    efficiency: ["Efficiency", "Summarizes measured computational cost alongside accuracy under the saved workload. Hardware, batch size and member count affect comparisons."],
    status: ["Analysis status", "Complete means the stage produced its configured outputs; partial means some outputs or evidence requirements remain unavailable. Pending means no result is available in this static site snapshot. These labels are not scientific success or significance claims."]
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
    pausePlayback();
    closeHelp();
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
    $("stats").replaceChildren(...[[counts.profiles.toLocaleString(), "evaluation profiles"], [counts.points.toLocaleString(), "temperature measurements"], [fmt(model.mae), "DepthDif MAE · °C"], [signed(model.mae - glorys.mae), "MAE difference vs GLORYS · °C"]].map(([value, label]) => {
      const node = element("div", undefined, "stat");
      node.append(element("strong", value), explained(label, label.includes("difference") ? "delta" : label.includes("MAE") ? "mae" : "profiles"));
      return node;
    }));
    const calibration = selected.partition.find((row) => row.partition === "calibration");
    $("interpretation").textContent = `${counts.clusters} evaluation cluster${counts.clusters === 1 ? " (one cruise)" : "s"}; ${calibration?.profiles ?? 0} profiles reserved for calibration. DepthDif MAE is ${Math.abs(model.mae - glorys.mae).toFixed(4)} °C ${model.mae > glorys.mae ? "higher" : "lower"} than packaged GLORYS. These are descriptive discovery results${counts.clusters === 1 ? "; one cruise does not support confidence intervals or significance claims" : ""}.`;
    tableRows("overall-table", selected.overall.map((row) => ({ highlight: row.method === "residual", cells: [explained(name(row.method), row.method), fmt(row.mae), fmt(row.rmse), fmt(row.crps), signed(row.mae - glorys.mae)] })));
    $("overall-csv").href = `${base}${selected.id}/overall_scores.csv`;
    $("baseline-status").textContent = selected.baselines.filter((row) => row.status !== "available").map((row) => `${name(row.method)}: ${row.status}, ${row.invalid_points} invalid predictions across the full input cohort; excluded from the shared comparison.`).join(" ");
    $("depth-note").textContent = `${selected.id.startsWith("italian") ? "25 m bins, left-closed and right-open, plotted at bin centers." : "Exact model depth levels."} Hover for profile and measurement counts. Click a legend label to show or hide a method. Curves are descriptive; no confidence bands are implied.`;
    tableRows("depth-table", selected.depth.map((row) => ({ highlight: row.method === "residual", cells: [explained(name(row.method), row.method), row.depth_resolution === "bin" ? `${row.depth_lower_m}–${row.depth_upper_m}` : row.depth_m, row.n_profiles, row.n_points, fmt(row.mae), fmt(row.rmse), fmt(row.bias), fmt(row.crps)] })));
    $("method-help").replaceChildren(...[...new Set(selected.overall.map((row) => row.method))].map((method) => explained(name(method), method)));
    $("profile").replaceChildren(...selected.profiles.map((profile) => { const option = element("option", profile.id); option.value = profile.id; return option; }));
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
    campaign = await fetchJSON(`${base}campaign.json`);
    $("snapshot").textContent = `Snapshot ${new Date(campaign.updated_utc).toLocaleString()} · static snapshot`;
    $("cohort").replaceChildren(...campaign.cohorts.map((cohort) => { const option = element("option", `${cohort.label}${cohort.available ? "" : " · pending"}`); option.value = cohort.id; return option; }));
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
