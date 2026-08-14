# Smooth GLORYS-Delta Pretraining Targets

The synthetic Stage 1 target retains the observed EO surface exactly and adds an
offline-fitted GLORYS climatological depth delta:

```text
temperature(z,y,x) = observed_SST(y,x) + smooth_GLORYS_delta[month,z,y,x]
salinity(z,y,x) = observed_SSS(y,x) + smooth_GLORYS_delta[month,z,y,x]
```

The v2 prior fits the delta on a monthly 10° grid, shrinks sparse cells toward
the global monthly profile, smooths the grid with periodic longitude handling,
and bilinearly samples it at patch pixels. The loader never retrieves paired or
same-date GLORYS for this synthetic target. Sparse ARGO values still override
the target at their exact cells and levels.

Unsupported abyssal levels carry the deepest supported GLORYS delta. Every
bathymetrically valid synthetic cell is supervised equally with the standard
diffusion loss. These targets remain a controlled initialization objective, not
subsurface truth.
