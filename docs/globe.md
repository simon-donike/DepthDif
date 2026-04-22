# 3D Globe

Use this page to inspect one hosted global inference export on a Cesium globe.

The page loads a `globe-config.json` manifest, then overlays:
- the stitched DepthDif top-band prediction
- the stitched GLORYS top-band ground truth
- the observed Argo points GeoJSON

By default the page reads the placeholder config at `docs/assets/globe/globe-config.json`. To point the viewer at a hosted export bundle, open this page with a `config` query parameter:

```text
globe/?config=https://<bucket-or-site>/<run_name>/globe/globe-config.json
```

The hosted asset bundle is produced by `inference/export_cesium_globe_assets.py`.

<div class="globe-page">
  <div class="globe-toolbar">
    <div class="globe-toolbar__group">
      <span class="globe-toolbar__label">Layers</span>
      <label class="globe-toggle">
        <input id="globe-toggle-prediction" type="checkbox" checked />
        <span>Prediction</span>
      </label>
      <label class="globe-toggle">
        <input id="globe-toggle-ground-truth" type="checkbox" checked />
        <span>Ground truth</span>
      </label>
      <label class="globe-toggle">
        <input id="globe-toggle-points" type="checkbox" checked />
        <span>Argo points</span>
      </label>
    </div>
    <div class="globe-toolbar__group">
      <label class="globe-range" for="globe-overlay-opacity">
        <span class="globe-toolbar__label">Overlay opacity</span>
        <input id="globe-overlay-opacity" type="range" min="0.15" max="1" step="0.05" value="0.85" />
      </label>
      <button id="globe-reset-camera" class="globe-reset-button" type="button">Reset View</button>
    </div>
    <div class="globe-toolbar__group globe-toolbar__group--status">
      <span class="globe-toolbar__label">Selected date</span>
      <span id="globe-selected-date">waiting for config</span>
      <span id="globe-status" class="globe-status" data-kind="info">Initializing globe viewer.</span>
    </div>
  </div>
  <div id="depthdif-cesium-globe" class="globe-canvas" aria-label="DepthDif Cesium globe viewer"></div>
</div>
