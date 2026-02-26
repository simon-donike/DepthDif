# Data + Coordinate Injection
This page explains how location and date features are fused and injected into the denoiser using one FiLM-style conditioning path.

## Where This Happens in Code
- coordinate/date encoding logic: `models/difFF/DenoisingDiffusionProcess/DenoisingDiffusionProcess.py`  
- FiLM application in ConvNeXt block: `models/difFF/DenoisingDiffusionProcess/backbones/unet_convnext.py`  

## Enabling It
Main config flags (model section):
- `coord_conditioning.enabled`  
- `coord_conditioning.encoding`: `unit_sphere`, `sincos`, `raw`  
- `coord_conditioning.include_date`  
- `coord_conditioning.date_encoding`: currently `day_of_year_sincos`  
- `coord_conditioning.embed_dim`  

Data-side requirement:
- `dataset.output.return_coords: true` so batches include `coords`  

Runtime requirements enforced by code:
- if coord conditioning is enabled and `coords` is missing, inference/training raises an error  
- if date conditioning is enabled and `date` is missing, inference/training raises an error  

## Coordinate Encoding Options
### `unit_sphere` (default)
Input: latitude/longitude in degrees.

Transform:
- convert to radians  
- map to 3D unit sphere  
- output features: `(x, y, z)`  

Why it is useful:
- avoids longitude discontinuity at ±180°  

This 3D view shows one example coordinate encoded as `(x, y, z)` on the sphere. Red/green/blue arrows are the `x/y/z` components, and the black arrow is the resulting encoded vector.  
<p align="center">
  <img src="../assets/single_coord_encoding_3d.png" width="50%" />
</p>

This image evaluates all integer `(lat, lon)` combinations and maps the 3D unit-sphere encoding `(x, y, z)` to RGB. It visualizes how each coordinate is transformed into a 3-value vector before FiLM embedding, showing smooth transitions and proper wrap-around from -90 to +90 and -180 to +180.
<p align="center">
  <img src="../assets/coord_encoding_unit_sphere_rgb.png" width="80%" />
</p>


### `sincos`
Features:  
- `sin(lat), cos(lat), sin(lon), cos(lon)`  

Why it is useful:  
- periodic and wrap-safe representation  

### `raw`
Features:  
- `lat / 90`  
- `lon / 180`  

Why it is simple:
- minimal feature transform, but not wrap-safe at dateline transitions  


## Date Encoding
Current implementation supports one option:  
- `day_of_year_sincos`  

Pipeline:
1. parse date as integer `YYYYMMDD`  
2. validate month/day values  
3. compute non-leap day-of-year using fixed month offsets  
4. encode as:  
   - `sin(2*pi*doy/365)`  
   - `cos(2*pi*doy/365)`  

Dataset date parsing convention:  
- monthly source names (`YYYYMM`) are converted to mid-month (`YYYYMM15`)  
- in the OSTIA overlap setup, depth files still provide `source_file` while OSTIA timestamps are stored separately (`ostia_timestamp_utc`), so model date conditioning remains mid-month aligned

This plot shows all 365 day-of-year embeddings as points in the sin/cos plane. Each day maps to one point on the unit circle, which makes the representation periodic and year-wrap safe.  

<p align="center">
  <img src="../assets/time_encodings_365.png" width="50%" />
</p>

## Feature Fusion
When `include_date=true`:  
- encoded date features are concatenated with encoded coordinate features  
- the concatenated vector is passed through a small MLP (`coord_mlp`) to produce one joint conditioning embedding  

## FiLM Injection Mechanism
Inside each `ConvNextBlock`:  
```python
self.coord_mlp = nn.Sequential(nn.GELU(), nn.Linear(coord_emb_dim, dim * 2))
scale_shift = self.coord_mlp(coord_emb)
scale, shift = scale_shift.chunk(2, dim=1)
h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
```

Interpretation:  
- date and coordinate features are not injected separately; they first become one joint embedding (`coord_emb`) and are applied together in the same FiLM transform  
- per-sample, per-channel scale and shift.  
- values are broadcast across spatial dimensions.  
- `1 + scale` keeps identity behavior easy (`scale=0`, `shift=0` -> no change).  

## Interaction With Time Conditioning. 
Time embedding and coordinate conditioning are complementary:  
- time embedding is additive per channel  
- coordinate/date conditioning is scale-and-shift per channel  

So each block receives:  
- diffusion-step context from timestep embeddings  
- geophysical context from location/date embeddings  
