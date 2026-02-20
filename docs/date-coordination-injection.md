# Date + Coordination Injection

## A2: FiLM Coordinate Injection Details

## Coordinate Encoding Options
Encoding options (set with `model.coord_conditioning.encoding`):
- `unit_sphere`: Convert lat/lon to a 3D unit vector (x,y,z). This avoids lon wrap discontinuity and is the default.
- `sincos`: Use sin/cos for lat and lon (4D). Also wrap-safe, slightly higher dimensional.
- `raw`: Normalize degrees to [-1, 1] (lat/90, lon/180). Simplest but can be discontinuous at +/-180.

When `model.coord_conditioning.include_date=true`, `batch["date"]` is parsed as `YYYYMMDD`.
Monthly file names (`YYYYMM`) are mapped to `YYYYMM15` before encoding.

## Date Encoding And Fusion With Coordinates
Date encoding is controlled by `model.coord_conditioning.date_encoding`.

Current option:
- `day_of_year_sincos`: parse `YYYYMMDD` -> compute non-leap `day_of_year` -> encode as:
  - `sin(2*pi*day_of_year/365)`
  - `cos(2*pi*day_of_year/365)`

Fusion with coordinate encoding:
- First encode coordinates using `model.coord_conditioning.encoding`.
- If `include_date=true`, concatenate date features to the coordinate feature vector:
  - `fused = concat(coord_features, date_features)`
- The fused vector is passed through the coordinate MLP to produce one embedding used by FiLM.

## Exact Injection Mechanism (Scale-Shift)
The coordinate embedding is injected via a per-channel FiLM scale and shift inside each `ConvNextBlock`.

Inside `ConvNextBlock`:
```python
self.coord_mlp = nn.Sequential(nn.GELU(), nn.Linear(coord_emb_dim, dim * 2))
...
scale_shift = self.coord_mlp(coord_emb)   # (B, 2*dim)
scale, shift = scale_shift.chunk(2, dim=1) # each (B, dim)

h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
```

That is:
```
h[b,c,x,y] <- h[b,c,x,y] * (1 + s[b,c]) + t[b,c]
```

Notes:
- `scale` and `shift` are per-sample, per-channel and broadcast to `(B, C, H, W)`.
- Applied after the depthwise conv (`ds_conv`) and before the main conv stack (`self.net`).
- This is classic FiLM conditioning: coordinates decide how strongly each channel is amplified/suppressed and offset.
- Why `1 + scale`? It keeps the identity map easy: if `scale=0` and `shift=0`, coords do nothing. This is more stable than multiplying by `scale` directly.

## Interaction With Time Conditioning
Time conditioning is additive:
```python
condition = self.mlp(time_emb)   # (B, dim)
h = h + condition[:, :, None, None]
```

So:
- Time adds a bias per channel.
- Coords do a scale-and-shift per channel.
- These are compatible: time tells the block where it is in diffusion, coords tell it where on Earth the sample belongs.
