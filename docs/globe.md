# 3D Globe

<div class="globe-page">
  <div class="globe-stage">
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
          <span class="globe-toolbar__label">Opacity</span>
          <input id="globe-overlay-opacity" type="range" min="0.15" max="1" step="0.05" value="0.85" />
        </label>
        <button id="globe-reset-camera" class="globe-reset-button" type="button">Reset View</button>
      </div>
    </div>
    <div class="globe-meta">
      <span id="globe-selected-date" class="globe-chip">Date: --</span>
      <span id="globe-status" class="globe-chip globe-status" data-kind="info">Loading</span>
    </div>
    <div id="depthdif-cesium-globe" class="globe-canvas" aria-label="DepthDif Cesium globe viewer"></div>
  </div>
  <p class="globe-footnote">
    Default view: hosted 2015-06-15 export. Override via <code>?config=&lt;public-globe-config-url&gt;</code>.
  </p>
</div>
