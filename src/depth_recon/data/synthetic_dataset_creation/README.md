# Vertical-Offset Pretraining Targets

The maintained synthetic target is deterministic:

```text
temperature(z,y,x) = observed_SST(y,x) + mean_GLORYS[temperature(z)-temperature(surface)]
salinity(z,y,x) = observed_SSS(y,x) + mean_GLORYS[salinity(z)-salinity(surface)]
```

Every depth therefore retains exactly the observed surface spatial pattern and
frequencies. Only one scalar changes per depth. GLORYS is used offline to fit
those coefficients; the training loader does not retrieve same-date GLORYS.

`vertical_offset_prior.py` implements the target,
`fit_vertical_offset_prior.py` fits it, and
`plot_vertical_offset_examples.py` writes qualitative comparisons. Sparse real
ARGO values override synthetic values at their exact cells and depths.

These targets are not subsurface truth. They intentionally copy surface fronts
unchanged through depth and serve only as a controlled initialization objective.
