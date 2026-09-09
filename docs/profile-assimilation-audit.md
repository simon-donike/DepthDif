# Profile Assimilation Audit

The `profile_audit` package implements the GLORYS12 profile audit pipeline. It
keeps archive membership, assimilation eligibility, confirmed GLORYS
acceptance, and demonstrated analysis influence as separate evidence levels.

Although OceanDepths uses `argo` in paths and variable names, its profile source
is the UK Met Office EN4.2.2 archive. The exporter therefore labels every row as
EN4 and retains the immutable `(profile_source_file, source_profile_idx)` key.

## Minimal Workflow

```bash
profile-audit inventory \
  --oceandepths-revision 4fd0400c40283073257246b26a7432b7e43a1da3 \
  --oceandepths-root /scratch/rcartuyvels/ocean_data \
  --output outputs/source_inventory.json

profile-audit export-oceandepths \
  --root /scratch/rcartuyvels/ocean_data \
  --output outputs/oceandepths_profiles.parquet

profile-audit parse-italian-xbt \
  --download-dir inputs/italian_xbt \
  --profiles-output outputs/italian_profiles.parquet \
  --observations-output outputs/italian_observations.parquet \
  --sensitivity-output outputs/italian_observations_qf12.parquet

profile-audit acquire-inputs \
  --from-profiles outputs/italian_profiles.parquet \
  --output-dir inputs/assimilation \
  --extract xbt

profile-audit index-inputs frozen_cora/*.nc \
  --archive-version CORA4.1 \
  --profiles-output outputs/cora41_profiles.parquet \
  --observations-output outputs/cora41_observations.parquet

profile-audit match \
  --targets outputs/italian_profiles.parquet \
  --target-observations outputs/italian_observations.parquet \
  --candidates outputs/cora41_profiles.parquet \
  --candidate-observations outputs/cora41_observations.parquet \
  --output outputs/matches.parquet

profile-audit classify \
  --profiles outputs/italian_profiles.parquet \
  --matches outputs/matches.parquet \
  --output outputs/classified.parquet

profile-audit report \
  --audit outputs/classified.parquet \
  --source-inventory outputs/source_inventory.json \
  --output-dir outputs/report
```

Use only frozen CORA/NRT snapshots corresponding to the GLORYS production
period. Matching a current archive does not prove historical input membership.
The unresolved 2017 through June 2021 source period is retained as unresolved,
and only returned Mercator profile feedback can promote a result to evidence
level C.

The default matching thresholds live in `src/profile_audit/config.yaml`.
Profile-value thresholds are provisional and should be recalibrated from known
positive and neighboring negative XBT controls before paper analysis.

## EN4-Wide Audit

The Italian casts are a focused independent dataset, but they are not the only
profiles worth auditing. OceanDepths contains 9,485,977 EN4-derived profiles.
Of these, 6,554,631 fall in 2000-2016, where the public CORA4.1, CORA5.0, and
CORA5.1 releases provide candidate historical inputs. Later profiles must be
reported separately because the exact source is unresolved or internal.

First build memory-efficient metadata indexes after extraction:

```bash
profile-audit index-inputs /scratch/rcartuyvels/glorys_inputs/extracted/cora41 \
  --archive-version CORA4.1 \
  --profiles-output outputs/cora41_profile_index.parquet \
  --metadata-only

profile-audit index-inputs /scratch/rcartuyvels/glorys_inputs/extracted/cora50 \
  --archive-version CORA5.0 \
  --profiles-output outputs/cora50_profile_index.parquet \
  --metadata-only

profile-audit index-inputs /scratch/rcartuyvels/glorys_inputs/extracted/cora51 \
  --archive-version CORA5.1 \
  --profiles-output outputs/cora51_profile_index.parquet \
  --metadata-only
```

For the complete multi-instrument audit, the indexer can read archive members
directly and only keeps one temporary NetCDF file on disk at a time. This avoids
extracting a second full copy of each CORA release:

```bash
profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/45999.tar.gz \
  --archive-version CORA4.1 \
  --profiles-output outputs/cora41_profile_index.parquet \
  --metadata-only
```

Use the analogous `56697.tar` and `56698.tar` paths for CORA5.0 and CORA5.1.
The `--extract xbt` acquisition option remains appropriate for the focused
Italian audit, while direct archive indexing is required for all EN4 instrument
families.

Then run the streaming global candidate audit:

```bash
profile-audit audit-en4 \
  --oceandepths-index /scratch/rcartuyvels/ocean_data/indices/profiles.parquet \
  --archive CORA4.1=outputs/cora41_profile_index.parquet \
  --archive CORA5.0=outputs/cora50_profile_index.parquet \
  --archive CORA5.1=outputs/cora51_profile_index.parquet \
  --revision 4fd0400c40283073257246b26a7432b7e43a1da3 \
  --output outputs/oceandepths_archive_candidate_audit.parquet
```

This stage streams OceanDepths in bounded batches and uses unit-sphere spatial
trees, so antimeridian and polar searches do not require loading either global
table into pandas. It emits exactly one row per EN4 profile with one of these
statuses:

- `no_spatiotemporal_candidate`: no profile exists in the period-correct CORA
  release within 24 hours and 25 km.
- `candidate_requires_fingerprint`: at least one candidate must be compared by
  profile values before archive membership can be claimed.
- `source_unresolved`: 2017 through June 2021 cannot currently be audited.
- `internal_snapshot_unavailable`: the post-July-2021 internal `013_047`
  snapshot is unavailable.
- `invalid_target_metadata` or `archive_not_supplied`: the target or local
  acquisition is incomplete.

`no_spatiotemporal_candidate` is the strongest automatic public-data evidence
of absence, but its exact meaning remains bounded by the supplied CORA release
and matching windows. Candidate rows are not archive matches until profile
fingerprinting succeeds, and archive matches are not proof of GLORYS acceptance.

## Historical Input Acquisition

`acquire-inputs` automatically maps target dates to CORA4.1, CORA5.0, and
CORA5.1 and downloads the official archived releases from SEANOE. These public
archives total approximately 86.6 GB. SEANOE does not currently support partial
archive retrieval, so `--extract xbt` reduces extracted storage and indexing
work but cannot reduce the archive download itself.

Inspect availability without downloading:

```bash
profile-audit acquire-inputs \
  --from-profiles outputs/italian_profiles.parquet \
  --output-dir inputs/assimilation \
  --discovery-only
```

Download one release explicitly:

```bash
profile-audit acquire-inputs \
  --release cora41 \
  --output-dir inputs/assimilation \
  --extract xbt
```

Downloads use temporary `.part` files, resume if the server supports byte
ranges, validate the published byte size, and record SHA-256 in
`assimilation_input_inventory.json`. The index command accepts extracted
directories and discovers NetCDF files recursively:

```bash
profile-audit index-inputs inputs/assimilation/extracted/cora41 \
  --archive-version CORA4.1 \
  --profiles-output outputs/cora41_profiles.parquet \
  --observations-output outputs/cora41_observations.parquet
```

No public frozen source was found for 2017 through June 2021. The documented
post-July-2021 product, `INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047`, was an
internal CMEMS assimilation feed, also documented under the older name
`INSITU_GLO_TS_ASSIM_NRT_OBSERVATIONS_013_047`. It was based on the Coriolis
In Situ TAC ecosystem and stored observations concatenated by day and
instrument/type, but neither a public endpoint nor its exact selection and
thinning rules were found. The acquisition manifest records both periods as
blocked and never substitutes the current evolving CORA or NRT products.
