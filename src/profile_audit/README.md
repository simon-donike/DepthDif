# GLORYS12 Profile Assimilation Audit

This package audits whether observational profiles used by DepthDif may also
have entered GLORYS12. It supports two related workflows:

1. A focused audit of the independent Italian PNRA XBT casts ([Zenodo record](https://zenodo.org/records/14848849)).
2. A global audit of the EN4-derived profiles distributed in OceanDepths.

The package deliberately separates archive membership from actual GLORYS use.

The Italian XBT workflow can also screen post-2016 casts against the public
`013_030` product. It is already supported by `acquire-013030-proxy`,
`index-inputs`, and `match`; follow [Italian XBT Audit](#italian-xbt-audit),
especially [Step 7](#7-optionally-screen-post-2017-casts-against-current-013_030).
This is a current Coriolis-product proxy search, not an EN4-wide audit and not
proof of membership in the internal historical `013_047` assimilation feed.

## Evidence Levels

| Level | Claim | Required evidence |
|---|---|---|
| A | Present in a candidate input archive | Match in the correct frozen CORA or NRT input release |
| B | Potentially eligible | Level A plus usable variables, levels, period, and input QC |
| C | Accepted by GLORYS | Profile-level Mercator used/rejected feedback |
| D | Influenced the analysis | Innovation, residual, or increment diagnostics |

A CORA match does not prove assimilation. A missing CORA match means only that
no qualifying match was found in the specified archive under the declared
matching rules.

## Known GLORYS Input Timeline

| Observation period | Best documented source | Public audit status |
|---|---|---|
| 1993-2013 | CORA4.1 | Archived public release available; the QUID says 2003, probably a typo requiring producer confirmation |
| 2014-2015 | CORA5.0 | Archived public release available |
| 2016 | CORA5.1 | Archived public release available |
| 2017-June 2021 | Unknown | Source unresolved |
| July 2021 onward | `INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047` | Internal snapshot unavailable publicly |

`013_047`, also called
`INSITU_GLO_TS_ASSIM_NRT_OBSERVATIONS_013_047`, was an internal CMEMS T/S
assimilation feed based on the Coriolis/In Situ TAC ecosystem. Available
documentation says it contained quality-controlled, homogenized observations
concatenated by day and instrument/type. No public endpoint, frozen membership
list, or exact selection and thinning specification has been found.

Therefore, public data cannot establish profile-level GLORYS use after 2017.
Definitive answers require Mercator's frozen input or observation-feedback
records.

## Environment

The repository uses `uv`; pip is not required.

To synchronize the already activated environment and install the command:

```bash
source /scratch/rcartuyvels/depth_env/bin/activate
uv sync --active --frozen
profile-audit --help
```

If the package is already synchronized, use `profile-audit` directly. Do not
wrap it in `uv run` while relying on the separately activated `depth_env`, since
plain `uv run` targets the project `.venv` unless `--active` is supplied.

Create working directories. Large archives and indexes should remain on
scratch:

```bash
mkdir -p outputs inputs/italian_xbt
mkdir -p /scratch/rcartuyvels/glorys_inputs
```

## OceanDepths Dataset

The local dataset is already available at:

```text
/scratch/rcartuyvels/ocean_data
```

No Hugging Face synchronization is required to use it. Although paths and
variables use the name `argo`, the profile source is the UK Met Office EN4.2.2
archive. Stable lineage within this snapshot is retained as:

```text
(profile_source_file, profile_idx)
```

For final reproducibility, a pinned Hugging Face snapshot can be downloaded to
a separate directory without changing the local copy:

```bash
uvx --from huggingface-hub hf download \
  ESA-philab/OceanDepths \
  --repo-type dataset \
  --revision 4fd0400c40283073257246b26a7432b7e43a1da3 \
  --local-dir /scratch/rcartuyvels/oceandepths-4fd0400c
```

## Freeze Provenance

Create the initial source inventory:

```bash
profile-audit inventory \
  --oceandepths-revision 4fd0400c40283073257246b26a7432b7e43a1da3 \
  --oceandepths-root /scratch/rcartuyvels/ocean_data \
  --italian-xbt-record 14848849 \
  --output outputs/source_inventory.json
```

Each historical archive acquisition command uses a workflow-specific manifest
containing URLs, expected byte sizes, local paths, and SHA-256 checksums. Never
rely on the default `assimilation_input_inventory.json` when using the shared
`/scratch/rcartuyvels/glorys_inputs` directory.

# Italian XBT Audit

## 1. Download And Parse The XBT Files

```bash
profile-audit parse-italian-xbt \
  --download-dir inputs/italian_xbt \
  --profiles-output outputs/italian_profiles.parquet \
  --observations-output outputs/italian_observations.parquet \
  --sensitivity-output outputs/italian_observations_qf12.parquet
```

The primary observations contain `QF=1`. The sensitivity output contains
`QF in {1,2}`.

The parser preserves raw and corrected depth and temperature columns. It also
records source depth reversals instead of silently sorting or deleting them.

PNRA XXXII contains a verified provider spreadsheet-format defect affecting 60
dates. The parser corrects only the exact known file, gated by its SHA-256, and
records `datetime_correction_applied` and `datetime_correction_reason`. All
other files are parsed according to the declared MDY convention.

## 2. Discover Historical Inputs

Check URLs, sizes, and blocked periods without downloading:

```bash
profile-audit acquire-inputs \
  --from-profiles outputs/italian_profiles.parquet \
  --output-dir /scratch/rcartuyvels/glorys_inputs \
  --manifest /scratch/rcartuyvels/glorys_inputs/italian_discovery_inventory.json \
  --discovery-only
```

Inspect:

```text
/scratch/rcartuyvels/glorys_inputs/italian_discovery_inventory.json
```

The discovery command above writes `italian_discovery_inventory.json`. Keep
this manifest separate from the acquisition manifest below; acquisition
commands sharing the same output directory otherwise overwrite the default
`assimilation_input_inventory.json`. Do not run discovery and download for the
same archive set concurrently.

## 3. Split Italian Targets By The Correct Archive Period

Create period-specific targets before acquiring or indexing the archives. Do not
match every cast against every release:

```bash
python - <<'PY'
import pandas as pd

profiles = pd.read_parquet("outputs/italian_profiles.parquet")
observations = pd.read_parquet("outputs/italian_observations.parquet")
time = pd.to_datetime(profiles["datetime_utc"], utc=True)

masks = {
    "cora41": time.dt.year <= 2013,
    "cora50": time.dt.year.isin([2014, 2015]),
    "cora51": time.dt.year == 2016,
    "proxy013030": time.dt.year >= 2017,
}

for name, mask in masks.items():
    selected = profiles.loc[mask].copy()
    selected.to_parquet(f"outputs/italian_{name}_profiles.parquet", index=False)
    ids = set(selected["profile_id"])
    observations[observations["profile_id"].isin(ids)].to_parquet(
        f"outputs/italian_{name}_observations.parquet", index=False
    )
PY
```

Profiles from 2017 onward remain historically unresolved or unavailable even
when they are screened against the current `013_030` proxy.

## 4. Download CORA Archives

```bash
profile-audit acquire-inputs \
  --from-profiles outputs/italian_profiles.parquet \
  --output-dir /scratch/rcartuyvels/glorys_inputs \
  --manifest /scratch/rcartuyvels/glorys_inputs/italian_assimilation_input_inventory.json \
  --workers 3
```

The three archives total approximately 86.6 GB:

| Release | Archive | Size |
|---|---|---:|
| CORA4.1 | `45999.tar.gz` | 51.4 GB |
| CORA5.0 | `56697.tar` | 18.4 GB |
| CORA5.1 | `56698.tar` | 16.8 GB |

Downloads use atomic `.part` files, retry interrupted transfers, validate the
published byte size, and compute SHA-256. SEANOE does not support useful partial
archive retrieval, so selecting XBT files reduces extraction and indexing work
but not download size. Existing files with the expected size are reused when an
interrupted command is rerun. Independent archives are processed concurrently
with `--workers`; start with one worker per archive and reduce it if the scratch
filesystem becomes the bottleneck.

## 5. Index The XBT Inputs

Choose one of these mutually exclusive routes. The direct route is faster and
does not create an extracted XBT tree. The extraction route is useful when the
individual NetCDF files are needed for manual inspection or other tools.

### Direct Streaming (Recommended)

Use the Step 4 acquisition command without `--extract`, then stream only
target-day XBT NetCDF members directly from each archive with these commands:

```bash
profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/45999.tar.gz \
  --archive-version CORA4.1 \
  --target-profiles outputs/italian_cora41_profiles.parquet \
  --xbt-only \
  --profiles-output outputs/cora41_xbt_profiles.parquet \
  --observations-output outputs/cora41_xbt_observations.parquet

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/56697.tar \
  --archive-version CORA5.0 \
  --target-profiles outputs/italian_cora50_profiles.parquet \
  --xbt-only \
  --profiles-output outputs/cora50_xbt_profiles.parquet \
  --observations-output outputs/cora50_xbt_observations.parquet

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/56698.tar \
  --archive-version CORA5.1 \
  --target-profiles outputs/italian_cora51_profiles.parquet \
  --xbt-only \
  --profiles-output outputs/cora51_xbt_profiles.parquet \
  --observations-output outputs/cora51_xbt_observations.parquet
```

This route still scans compressed archive streams when necessary, but avoids an
extraction pass and the second read of extracted files.

### Extract Then Index

Run this acquisition command only instead of the direct route above. It creates
an extracted XBT tree, which can then be indexed with the commands below:

```bash
profile-audit acquire-inputs \
  --from-profiles outputs/italian_profiles.parquet \
  --output-dir /scratch/rcartuyvels/glorys_inputs \
  --manifest /scratch/rcartuyvels/glorys_inputs/italian_xbt_extraction_inventory.json \
  --extract xbt \
  --workers 3
```

CORA5.0 and CORA5.1 contain nested year archives; this pass opens only years
represented by the period-correct targets and extracts matching `_PR_XB.nc`
members directly. Then index the extracted files:

```bash
profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/extracted/cora41 \
  --archive-version CORA4.1 \
  --profiles-output outputs/cora41_xbt_extracted_profiles.parquet \
  --observations-output outputs/cora41_xbt_extracted_observations.parquet

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/extracted/cora50 \
  --archive-version CORA5.0 \
  --profiles-output outputs/cora50_xbt_extracted_profiles.parquet \
  --observations-output outputs/cora50_xbt_extracted_observations.parquet

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/extracted/cora51 \
  --archive-version CORA5.1 \
  --profiles-output outputs/cora51_xbt_extracted_profiles.parquet \
  --observations-output outputs/cora51_xbt_extracted_observations.parquet
```

The direct-streaming and extract-then-index routes now have distinct output
names. Run either route, or match the extracted indexes explicitly by using
the `_extracted_` candidate filenames in Step 6. Do not write both routes to
the same Parquet paths.

## 6. Match Each Period

```bash
profile-audit match \
  --targets outputs/italian_cora41_profiles.parquet \
  --target-observations outputs/italian_cora41_observations.parquet \
  --candidates outputs/cora41_xbt_profiles.parquet \
  --candidate-observations outputs/cora41_xbt_observations.parquet \
  --output outputs/italian_cora41_matches.parquet

profile-audit match \
  --targets outputs/italian_cora50_profiles.parquet \
  --target-observations outputs/italian_cora50_observations.parquet \
  --candidates outputs/cora50_xbt_profiles.parquet \
  --candidate-observations outputs/cora50_xbt_observations.parquet \
  --output outputs/italian_cora50_matches.parquet

profile-audit match \
  --targets outputs/italian_cora51_profiles.parquet \
  --target-observations outputs/italian_cora51_observations.parquet \
  --candidates outputs/cora51_xbt_profiles.parquet \
  --candidate-observations outputs/cora51_xbt_observations.parquet \
  --output outputs/italian_cora51_matches.parquet
```

Concatenate available-period matches:

```bash
python - <<'PY'
import pandas as pd

paths = [
    "outputs/italian_cora41_matches.parquet",
    "outputs/italian_cora50_matches.parquet",
    "outputs/italian_cora51_matches.parquet",
]
pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).to_parquet(
    "outputs/italian_matches.parquet", index=False
)
PY
```

## 7. Optionally Screen Post-2017 Casts Against Current `013_030`

The public `INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030` product can provide a
drop-in Coriolis proxy while access to internal `013_047` is being requested.
It is continuously updated and is not a historical snapshot. Positive matches
support common Coriolis provenance; negative matches do not prove historical
absence.

Configure credentials once:

```bash
copernicusmarine login
copernicusmarine login --check-credentials-valid
```

Create and inspect a target-specific file list without downloading NetCDFs:

```bash
profile-audit acquire-013030-proxy \
  --from-profiles outputs/italian_proxy013030_profiles.parquet \
  --output-dir /scratch/rcartuyvels/glorys_inputs/proxy_013030 \
  --part history
```

The resulting files are:

```text
/scratch/rcartuyvels/glorys_inputs/proxy_013030/proxy_inventory.json
/scratch/rcartuyvels/glorys_inputs/proxy_013030/selected_history_profile_files.txt
```

Download the selected files explicitly:

```bash
profile-audit acquire-013030-proxy \
  --from-profiles outputs/italian_proxy013030_profiles.parquet \
  --output-dir /scratch/rcartuyvels/glorys_inputs/proxy_013030 \
  --part history \
  --download-files
```

Index profile values and run matching:

```bash
profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/proxy_013030/files/history \
  --archive-version 013_030_PROXY \
  --profiles-output outputs/proxy013030_profiles.parquet \
  --observations-output outputs/proxy013030_observations.parquet

profile-audit match \
  --targets outputs/italian_proxy013030_profiles.parquet \
  --target-observations outputs/italian_proxy013030_observations.parquet \
  --candidates outputs/proxy013030_profiles.parquet \
  --candidate-observations outputs/proxy013030_observations.parquet \
  --output outputs/italian_proxy013030_matches.parquet
```

Keep `italian_proxy013030_matches.parquet` separate from frozen CORA evidence.
Do not promote proxy matches to evidence level A or B.

## 8. Classify Historical Evidence

```bash
profile-audit classify \
  --profiles outputs/italian_profiles.parquet \
  --matches outputs/italian_matches.parquet \
  --output outputs/italian_classified.parquet
```

This can establish archive membership and potential eligibility only. It cannot
set `confirmed_accepted` without Mercator feedback.

## 9. Generate The Italian Report

```bash
profile-audit report \
  --audit outputs/italian_classified.parquet \
  --source-inventory outputs/source_inventory.json \
  --output-dir outputs/italian_report
```

The report includes `mercator_feedback_request.csv`. Send this to Mercator and
request used/rejected status, rejection reason, cycle, QC1/QC2 results,
innovation, analysis residual, and source reference.

## 10. Run The QF Sensitivity Experiment

Repeat matching with `outputs/italian_observations_qf12.parquet` as the target
observation table. Keep primary and sensitivity outputs separate.

External NCEI, WOD, or accession crosswalks are optional. Users do not need to
manually create an identifier CSV to run this workflow.

# EN4-Wide OceanDepths Audit

OceanDepths contains 9,485,977 EN4-derived profiles from 2000 through July
2024. Public historical inputs currently make 6,554,631 profiles from
2000-2016 auditable at archive-membership level.

If the CORA archives have not already been acquired, download the three frozen
releases concurrently. Explicit releases are used here because the global
OceanDepths index is not an Italian XBT target table:

```bash
profile-audit acquire-inputs \
  --release cora41 \
  --release cora50 \
  --release cora51 \
  --output-dir /scratch/rcartuyvels/glorys_inputs \
  --manifest /scratch/rcartuyvels/glorys_inputs/en4_assimilation_input_inventory.json \
  --workers 3
```

The Italian and EN4-wide acquisition commands share the downloaded CORA
archives but use separate manifests. Complete archives are reused; do not run
the two acquisition commands concurrently while an archive is being
downloaded, because they share the `.part` download paths.

This downloads the same approximately 86.6 GB of CORA archives described in
the Italian workflow. Existing complete archives are reused. Do not add
`--extract`; the metadata indexer below streams archive members through bounded
temporary storage.

## 1. Build Complete Multi-Instrument CORA Metadata Indexes

The global audit must include all CORA instrument classes, not only XBT. The
indexer can stream NetCDF members directly from each archive and retains only
one temporary member at a time, avoiding a second full extracted copy.
The unrestricted archive-input behavior used below is intentionally preserved;
do not add `--xbt-only` because the EN4-wide audit requires every CORA
instrument class.

```bash
profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/45999.tar.gz \
  --archive-version CORA4.1 \
  --profiles-output /scratch/rcartuyvels/glorys_inputs/cora41_profile_index.parquet \
  --metadata-only

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/56697.tar \
  --archive-version CORA5.0 \
  --profiles-output /scratch/rcartuyvels/glorys_inputs/cora50_profile_index.parquet \
  --metadata-only

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/56698.tar \
  --archive-version CORA5.1 \
  --profiles-output /scratch/rcartuyvels/glorys_inputs/cora51_profile_index.parquet \
  --metadata-only
```

These commands can take substantial time because every NetCDF member must be
opened, but memory and temporary disk usage remain bounded.

## 2. Optionally Acquire The Current `013_030` Proxy

Use the lightweight OceanDepths profile index to select current `013_030`
files relevant to post-2017 targets:

```bash
profile-audit acquire-013030-proxy \
  --from-profiles /scratch/rcartuyvels/ocean_data/indices/profiles.parquet \
  --output-dir /scratch/rcartuyvels/glorys_inputs/proxy_013030_global \
  --part history
```

This global target covers almost the whole ocean and will select most relevant
temperature-profile files. Review
`selected_history_profile_files.txt` before starting the potentially large
download:

```bash
profile-audit acquire-013030-proxy \
  --from-profiles /scratch/rcartuyvels/ocean_data/indices/profiles.parquet \
  --output-dir /scratch/rcartuyvels/glorys_inputs/proxy_013030_global \
  --part history \
  --download-files
```

Run the selection and download commands sequentially. They intentionally share
the global proxy inventory and selected-file list, while the download command
also populates the shared `files/history` tree.

This is an evolving current proxy, not a frozen `013_047` parent snapshot.

## 3. Index The Current Proxy

```bash
profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/proxy_013030_global/files/history \
  --archive-version 013_030_PROXY \
  --profiles-output /scratch/rcartuyvels/glorys_inputs/proxy_013030_profile_index.parquet \
  --metadata-only
```

## 4. Run The Streaming Global Candidate Audit

```bash
profile-audit audit-en4 \
  --oceandepths-index /scratch/rcartuyvels/ocean_data/indices/profiles.parquet \
  --archive CORA4.1=/scratch/rcartuyvels/glorys_inputs/cora41_profile_index.parquet \
  --archive CORA5.0=/scratch/rcartuyvels/glorys_inputs/cora50_profile_index.parquet \
  --archive CORA5.1=/scratch/rcartuyvels/glorys_inputs/cora51_profile_index.parquet \
  --proxy-013030 /scratch/rcartuyvels/glorys_inputs/proxy_013030_profile_index.parquet \
  --revision 4fd0400c40283073257246b26a7432b7e43a1da3 \
  --output /scratch/rcartuyvels/glorys_inputs/oceandepths_candidate_audit_with_proxy.parquet
```

The command streams OceanDepths in bounded batches and uses unit-sphere spatial
trees. It writes exactly one row per profile.

## 5. Interpret Global Audit Statuses

| Status | Meaning |
|---|---|
| `no_spatiotemporal_candidate` | No candidate in the period-correct CORA release within 24 hours and 25 km |
| `candidate_requires_fingerprint` | Nearby candidate exists; profile values must still be compared |
| `source_unresolved` | Exact 2017-June 2021 source is unknown |
| `internal_snapshot_unavailable` | Frozen post-July-2021 `013_047` membership is unavailable |
| `proxy_no_spatiotemporal_candidate` | No nearby profile in current `013_030`; not definitive historical absence |
| `proxy_candidate_requires_fingerprint` | Nearby current proxy candidate requires value comparison |
| `archive_not_supplied` | Required local CORA index was not provided |
| `invalid_target_metadata` | EN4 target has invalid time or coordinates |

Summarize the first-stage result:

```bash
python - <<'PY'
import pandas as pd

audit = pd.read_parquet(
    "/scratch/rcartuyvels/glorys_inputs/oceandepths_candidate_audit_with_proxy.parquet",
    columns=["audit_status", "expected_archive"],
)
print(audit.groupby(["expected_archive", "audit_status"], dropna=False).size())
PY
```

Profiles marked `no_spatiotemporal_candidate` are the strongest automatically
identified independent candidates. The defensible claim is:

> No qualifying profile was found in the specified archived CORA release within
> 24 hours and 25 km.

Do not shorten this to “not used by GLORYS.”

Proxy rows retain the original `historical_coverage_status`, set
`evidence_scope=current_013030_proxy`, and must not be merged into definitive
historical absence counts.

### Export EN4 Profiles Without A Spatiotemporal Candidate

Export the stable OceanDepths profile ID and lineage for rows with no candidate
in either the period-correct CORA release or the current `013_030` proxy. Keep
`audit_status` in the output so proxy results remain distinguishable from
historical-release results:

```bash
python - <<'PY'
import pandas as pd

statuses = [
    "no_spatiotemporal_candidate",
    "proxy_no_spatiotemporal_candidate",
]
columns = [
    "target_profile_id", "target_profile", "profile_source_file",
    "source_profile_idx", "datetime_utc", "latitude", "longitude",
    "expected_archive", "historical_coverage_status", "evidence_scope",
    "audit_status",
]
no_match = pd.read_parquet(
    "/scratch/rcartuyvels/glorys_inputs/oceandepths_candidate_audit_with_proxy.parquet",
    columns=columns,
    filters=[("audit_status", "in", statuses)],
)
no_match.to_parquet(
    "outputs/en4_no_spatiotemporal_candidate_profiles.parquet", index=False
)
print(f"Wrote {len(no_match):,} profiles")
PY
```

`target_profile_id` is the stable ID for the pinned OceanDepths revision. The
file deliberately includes both status classes, but only
`no_spatiotemporal_candidate` supports the documented historical CORA
candidate-absence claim; the proxy status remains non-historical evidence.

## 6. Fingerprint Nearby Candidates

`candidate_requires_fingerprint` is not a match. Those pairs require lazy
temperature and salinity comparison between OceanDepths Zarr rows and the
corresponding CORA NetCDF profiles. The small-profile `match` command already
implements the profile metrics, but a global lazy-value fingerprint stage is
not yet exposed as a single production command. Until that stage is completed,
the global output is a conservative candidate-absence screen rather than a
complete archive-membership classification.

## Italian XBT Versus EN4.2.2

The GLORYS/CORA audit and the EN4 provenance audit answer different questions.
Use the command below to test whether the Italian XBT casts have a matching
representation in the pinned EN4.2.2-derived OceanDepths snapshot. This does
not establish that a profile was assimilated by GLORYS.

The local snapshot starts in 2000. Casts before 2000 are retained in the
summary with `outside_oceandepths_en4_snapshot`; they must not be counted as
EN4 non-matches.

```bash
profile-audit audit-italian-en4 \
  --targets outputs/italian_profiles.parquet \
  --observations outputs/italian_observations.parquet \
  --en4-index /scratch/rcartuyvels/ocean_data/indices/profiles.parquet \
  --oceandepths-root /scratch/rcartuyvels/ocean_data \
  --matches-output outputs/italian_en4_matches.parquet \
  --summary-output outputs/italian_en4_summary.parquet
```

The command searches all EN4 instrument classes within 24 hours and 25 km,
extracts only matching OceanDepths profiles, and applies the existing
temperature-curve fingerprint metrics. `italian_en4_matches.parquet` retains
all candidates and their source `(profile_source_file, profile_idx)` lineage;
`italian_en4_summary.parquet` contains one best-candidate row per Italian cast.

Interpret the classifications as EN4 provenance labels:

| Classification | Meaning |
|---|---|
| `Near-exact` or `Probable duplicate` | EN4 candidate with a strong profile fingerprint |
| `Candidate` | Spatiotemporal EN4 candidate requiring review or stronger controls |
| `Ambiguous` | Multiple similarly ranked EN4 candidates |
| `No archive match` | No EN4 profile within 24 hours and 25 km |
| `Outside EN4 snapshot coverage` | Italian cast predates the local 2000-onward snapshot |

An EN4 match supports common data provenance only. It is not evidence of
GLORYS acceptance, assimilation, or analysis influence.

### Export Italian Casts Without A Close EN4 Match

The one-row-per-profile file `outputs/italian_en4_summary.parquet` contains the
Italian `profile_id`, date, source file, cruise, station, EN4 coverage status,
classification, and `candidate_count`. To select only profiles from 2000
onward with no EN4 candidate in the 24-hour/25-km window:

```bash
python - <<'PY'
import pandas as pd

summary = pd.read_parquet("outputs/italian_en4_summary.parquet")
no_match = summary[
    (summary["en4_coverage_status"] == "within_oceandepths_en4_snapshot")
    & (summary["candidate_count"] == 0)
].copy()
no_match[
    ["profile_id", "cruise", "station", "datetime_utc", "source_file",
     "latitude", "longitude_raw"]
].to_parquet("outputs/italian_en4_no_match_profiles.parquet", index=False)
print(f"Wrote {len(no_match)} profiles")
PY
```

The resulting `profile_id` values have the form:

```text
italian-xbt:<cruise>:<station>
```

For example, `italian-xbt:PNRAXXXIV:0` identifies station `0` from cruise
`PNRAXXXIV`. These are stable identifiers within the parsed Italian dataset,
but they are not numeric row offsets into the Zenodo record. Use the
`cruise`, `station`, and `source_file` columns to locate the original cast in
the downloaded Zenodo text files. After downloading the record, the parser
can be rerun on the source files and the same `profile_id` can be used to
filter `outputs/italian_profiles.parquet` and
`outputs/italian_observations.parquet`.

The current audit produces 2,137 such post-2000 profiles. This is a statement
about the local pinned EN4.2.2/OceanDepths snapshot under the declared matching
window, not proof that the casts are absent from every EN4 release.

# E4/E5 Local Profile Audit

E4 uses the California Underwater Glider Network (CUGN); E5 uses the BGEP 2018
CTD casts and ITP-107. Normalize the frozen local sources before any archive
or EN4 audit:

```bash
profile-audit audit-e4-e5-inputs --output-dir outputs/e4_e5_audit
```

This writes standard `profile_id`, time, position, source/platform metadata,
and native-depth T/S tables. CUGN is restricted to `2007-01-01 <= time <
2019-01-01`, exactly matching the Amaya et al. evaluation period. The command
does not resample any source profile onto GLORYS levels. It retains CUGN's
10-m bins, BGEP's 1-dbar downcasts, and ITP pressure samples; ITP also retains
`pressure_dbar`, because latitude-aware pressure-to-depth conversion belongs
in the E5 observation operator.

## GLORYS Assimilation Status

| Source | GLORYS status | Permitted interpretation |
|---|---|---|
| CUGN, 2007-2018 | Amaya et al. state that these glider T/S profiles were not assimilated by GLORYS. Record this as source/paper evidence, not Mercator acceptance feedback. | E4-P independent reference, subject to the checkpoint leakage and complete-mission conditioning holdout gates. |
| BGEP 2018-81 CTD | Unresolved. August-September 2018 falls in the 2017-June 2021 period whose frozen GLORYS input source is not publicly identified. | Same-date reconstruction diagnostic only, unless Mercator supplies frozen input/feedback showing absence. |
| ITP-107, September 2018 | Unresolved and potentially assimilated. CORA has an ITP ingestion pathway, while the applicable frozen GLORYS input is unavailable. | Same-date reconstruction diagnostic only, unless profile-level producer evidence resolves it. |

Do not translate a CUGN paper statement, archive non-match, or EN4 non-match
into a claim that a profile was accepted or influenced GLORYS. For the two
2018 Beaufort sets, `013_030` can be used only as the non-historical proxy
described above; it cannot resolve the `013_047`/2017-2021 historical gap.

## EN4.2.2 Training-Provenance Check

`audit-external-en4` is the source-neutral form of the Italian EN4 matcher. It
uses the local OceanDepths EN4.2.2 snapshot, searches within 24 hours and
25 km, then fingerprints T profiles and retains the stable EN4 lineage
`(profile_source_file, profile_idx)`. Run it separately for each source:

```bash
for source in cugn bgep_2018 itp107_2018_09; do
  profile-audit audit-external-en4 \
    --targets "outputs/e4_e5_audit/${source}_profiles.parquet" \
    --observations "outputs/e4_e5_audit/${source}_observations.parquet" \
    --en4-index /scratch/rcartuyvels/ocean_data/indices/profiles.parquet \
    --oceandepths-root /scratch/rcartuyvels/ocean_data \
    --matches-output "outputs/e4_e5_audit/${source}_en4_matches.parquet" \
    --summary-output "outputs/e4_e5_audit/${source}_en4_summary.parquet"
done
```

The CUGN run is much larger (116,490 profiles) than the two Beaufort runs;
run it as a separate batch and preserve its output rather than sampling or
silently dropping glider profiles. Each invocation prints its output paths
followed by the best-match counts, so the loop is a result rather than merely
a file-generation step. The 2026-08-21 local audit found:

| Source | Strong EN4 duplicate evidence | Requires review | No EN4 candidate within 24 h/25 km |
|---|---:|---:|---:|
| CUGN | 38,878 (`Near-exact` or `Probable duplicate`) | 9,241 (`Ambiguous` or `Candidate`) | 68,371 |
| BGEP 2018 | 0 | 0 | 67 |
| ITP-107, September 2018 | 16 (9 `Near-exact`, 7 `Probable duplicate`) | 8 (6 `Ambiguous`, 2 `Candidate`) | 0 |

Treat profiles with strong EN4 duplicate evidence as potentially represented
in the training snapshot; exclude them from strict held-out training
evaluation. Review `Ambiguous` and `Candidate` profiles before making the same
decision. A `No archive match` is only an absence from the local EN4.2.2
snapshot under this search window.

An EN4 match answers whether DepthDif may have trained on the profile (or a
duplicate) through OceanDepths. It does not answer whether GLORYS assimilated
it. Conversely, `No archive match` applies only to the pinned local snapshot
and its 24-hour/25-km search window; it is not evidence of absence from every
EN4 release.

## CUGN Versus Historical CORA Inputs

Also run the archive-membership check for CUGN. This provides an independent
duplicate screen alongside the Amaya et al. non-assimilation statement. It is
necessary because an EN4 match is not the only route by which a CUGN profile
could appear in a CORA input release.

First split the source-normalized profiles into the releases documented for
the GLORYS12 period. Keep observations with their corresponding profile rows:

```bash
python - <<'PY'
import pandas as pd

root = "outputs/e4_e5_audit"
profiles = pd.read_parquet(f"{root}/cugn_profiles.parquet")
observations = pd.read_parquet(f"{root}/cugn_observations.parquet")
time = pd.to_datetime(profiles["datetime_utc"], utc=True)

masks = {
    "cora41": time.dt.year <= 2013,
    "cora50": time.dt.year.isin([2014, 2015]),
    "cora51": time.dt.year == 2016,
}
for release, mask in masks.items():
    selected = profiles.loc[mask].copy()
    selected.to_parquet(f"{root}/cugn_{release}_profiles.parquet", index=False)
    observations[observations["profile_id"].isin(set(selected["profile_id"]))].to_parquet(
        f"{root}/cugn_{release}_observations.parquet", index=False,
    )
PY
```

Index the correct frozen CORA archive for each period. `--target-profiles`
now applies the declared 24-hour/25-km candidate window while indexing, so the
output holds only candidate CORA profiles and their T/S levels, rather than a
full value copy of the archive. CUGN samples on most days, so the CORA archives
must still be scanned; the filter reduces output size, not archive read time.

```bash
profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/45999.tar.gz \
  --archive-version CORA4.1 \
  --target-profiles outputs/e4_e5_audit/cugn_cora41_profiles.parquet \
  --profiles-output outputs/e4_e5_audit/cora41_cugn_candidates.parquet \
  --observations-output outputs/e4_e5_audit/cora41_cugn_candidate_observations.parquet

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/56697.tar \
  --archive-version CORA5.0 \
  --target-profiles outputs/e4_e5_audit/cugn_cora50_profiles.parquet \
  --profiles-output outputs/e4_e5_audit/cora50_cugn_candidates.parquet \
  --observations-output outputs/e4_e5_audit/cora50_cugn_candidate_observations.parquet

profile-audit index-inputs \
  /scratch/rcartuyvels/glorys_inputs/archives/56698.tar \
  --archive-version CORA5.1 \
  --target-profiles outputs/e4_e5_audit/cugn_cora51_profiles.parquet \
  --profiles-output outputs/e4_e5_audit/cora51_cugn_candidates.parquet \
  --observations-output outputs/e4_e5_audit/cora51_cugn_candidate_observations.parquet
```

Fingerprint each release's candidates and assign archive-evidence classes:

```bash
for release in cora41 cora50 cora51; do
  profile-audit match \
    --targets "outputs/e4_e5_audit/cugn_${release}_profiles.parquet" \
    --target-observations "outputs/e4_e5_audit/cugn_${release}_observations.parquet" \
    --candidates "outputs/e4_e5_audit/${release}_cugn_candidates.parquet" \
    --candidate-observations "outputs/e4_e5_audit/${release}_cugn_candidate_observations.parquet" \
    --output "outputs/e4_e5_audit/cugn_${release}_matches.parquet"
  profile-audit classify \
    --profiles "outputs/e4_e5_audit/cugn_${release}_profiles.parquet" \
    --matches "outputs/e4_e5_audit/cugn_${release}_matches.parquet" \
    --output "outputs/e4_e5_audit/cugn_${release}_classified.parquet"
done
```

The historical public releases do not resolve CUGN profiles from 2017-2018.
Keep those rows as `source_unresolved` (2017 through June 2021) and do not
combine them with `No archive match`. A `Near-exact`, `Probable duplicate`, or
exact-identifier CORA match establishes candidate archive membership and may
establish eligibility after QC review, but it does not overturn the paper's
claim or prove that GLORYS accepted the observation. Mercator feedback remains
required for accepted/influenced status.

## 7. Treat Unavailable Periods Separately

Never combine `source_unresolved` or `internal_snapshot_unavailable` with
`no_spatiotemporal_candidate`. Missing evidence is not evidence that a profile
was absent.

# Definitive Post-2017 Evidence

There is no verified public route to profile-level certainty after 2017.
Request the following from Mercator Ocean and the In Situ TAC:

1. Exact source and frozen input snapshot for 2017 through June 2021.
2. Frozen daily `013_047` files from July 2021 onward.
3. The retired/internal `013_047` metadata record or PUM.
4. Mapping between `013_047` and the public `013_030` Coriolis product.
5. Temporal, horizontal, vertical, and platform thinning rules.
6. Profile-level used, rejected, subsampled, QC1, and QC2 status.
7. Observation-minus-background innovations and analysis residuals.

Useful contacts:

| Organization | Contact |
|---|---|
| Copernicus Marine support | <https://marine.copernicus.eu/contact> |
| Mercator Ocean | <https://www.mercator-ocean.eu/contact-us/> |
| In Situ TAC coordination | `instacco@ifremer.fr` |
| In Situ TAC operations | `cmems-service@ifremer.fr` |

# Weak Influence Analysis

If daily GLORYS samples have been prepared for observation dates, offsets, and
pseudo-profile controls, summarize them with:

```bash
profile-audit influence \
  --samples outputs/influence_samples.parquet \
  --residuals-output outputs/influence_residuals.parquet \
  --summary-output outputs/influence_summary.csv
```

An unusually small residual near an observation date is consistent with
assimilation but is not level-C proof because public daily fields do not expose
the background, accepted-observation list, or analysis increment separately.

# Verification

Run the targeted tests:

```bash
PYTHONPATH=src python -m unittest tests.test_profile_audit -v
```

The tests cover XBT parsing and date correction, archive acquisition and retry,
safe XBT extraction, direct tar indexing, EN4 export, antimeridian matching,
profile fingerprints, evidence classification, reporting, and the streaming
EN4-wide candidate audit.
