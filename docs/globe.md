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
          <input id="globe-toggle-ground-truth" type="checkbox" />
          <span>GLORYS</span>
        </label>
        <label class="globe-toggle">
          <input id="globe-toggle-points" type="checkbox" />
          <span>Argo points</span>
        </label>
        <label class="globe-toggle">
          <input id="globe-toggle-full-samples" type="checkbox" />
          <span>Full sample locations</span>
        </label>
        <label class="globe-toggle">
          <input id="globe-toggle-patch-splits" type="checkbox" />
          <span>Train/val split</span>
        </label>
      </div>
      <div class="globe-toolbar__group">
        <label class="globe-range" for="globe-overlay-opacity">
          <span class="globe-toolbar__label">Opacity</span>
          <input id="globe-overlay-opacity" type="range" min="0.15" max="1" step="0.05" value="1" />
        </label>
        <button id="globe-toggle-spin" class="globe-reset-button" type="button" aria-pressed="true">
          Stop Spin
        </button>
        <button id="globe-reset-camera" class="globe-reset-button" type="button">Reset View</button>
      </div>
    </div>
    <div class="globe-legend" aria-label="Temperature color scale">
      <span class="globe-legend__title">Temperature</span>
      <div class="globe-legend__bar"></div>
      <div class="globe-legend__labels">
        <span>0°C</span>
        <span>30°C</span>
      </div>
    </div>
    <div id="depthdif-cesium-globe" class="globe-canvas" aria-label="DepthDif Cesium globe viewer"></div>
    <div id="globe-profile-popup" class="globe-profile-popup" hidden>
      <div class="globe-profile-popup__card">
        <div class="globe-profile-popup__header">
          <div>
            <strong id="globe-profile-popup-title">Full sample</strong>
            <div id="globe-profile-popup-subtitle" class="globe-profile-popup__subtitle"></div>
          </div>
          <button id="globe-profile-popup-close" class="globe-reset-button" type="button">Close</button>
        </div>
        <img id="globe-profile-popup-image" class="globe-profile-popup__image" alt="" />
      </div>
    </div>
  </div>
</div>
