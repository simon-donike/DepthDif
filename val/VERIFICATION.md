# Verification on 3 November 2026

The existing application source, tests, configs, and validation-loader shuffle
setting are unchanged. `git status --short` lists only the new `val/` directory;
`git diff --check` passes.

- Required formatter: `/work/envs/depth/bin/python -m black .`, exit 0,
  145 files unchanged on the final pass. Black prints an existing Python
  target-version warning. The sandbox formatter stalled; host execution completed.
- Full repository suite: **409 tests passed**, final run 34.909 seconds.
- Additional validation suite: **28 tests passed**, final run 5.780 seconds.
- Logs: `results/verification/{repository_tests,validation_tests,black_host}.log`.
  The sandboxed repository suite stalled during dataset setup; the host run passed.
- Both-variable synthetic report and saved-bundle rescoring passed. Open
  [the demonstration](results/demo/report/report.html); its scores are synthetic.
- Real temperature checkpoint/data smoke passed: four candidate patches,
  two members, two DDIM steps, GPU 1; **237 scored values, seven profiles,
  three bootstrap blocks**. This short chain is a functional check,
  not an accuracy experiment. Evidence is under `results/smoke_real_attempt04/`.
- Earlier smoke attempts caught an incorrect sampler-override path and mixed
  fractional-second timestamp parsing; both are fixed and regression-tested.
  One selected patch had no wet GLORYS reference support and was excluded.
  Failed-attempt logs remain under `results/verification/`.

The configured 2016 temperature, SST, profile store, mask, checkpoint, and
climatology paths are present. A raster read and the checkpoint smoke succeeded.
This does not verify every data chunk or constitute a complete regional evaluation.

## Italian data recovered

Raw PNRA files exist inside `evaluations/focus_evaluation.tar.gz`; the extracted
Italian directory is empty. The companion screened identity table has 2,137
profiles. The 2016 import matched all 102 requested profiles and retained 108,428
corrected `QF=1` measurements from two cruises. Evidence:
`results/xbt_import/{profiles.csv,manifest.json}`.

Reference-salinity preparation retained 79,367 rows across all 102 profiles;
29,061 rows lacked a permitted vertical bracket or wet salinity support.
The default 100 m model-level-gap cap causes most exclusions. The preparation
manifest records counts and every used raster hash. Evidence:
`results/xbt_reference/{profiles.csv,manifest.json}`.

GSW 3.6.19 was installed with `--no-deps --target val/vendor`, leaving the shared
Python environment unchanged. Using `PYTHONPATH=val/vendor:src`, real TEOS-10
conversion passed for all 79,367 prepared rows; the maximum correction from
in-situ to potential temperature was approximately 0.05268°C. Reference salinity
comes from GLORYS and remains an observation-operator assumption requiring a
sensitivity analysis for small gains.

## Scientific results still pending

The complete 20-member regional benchmark has not been run. The selected QC
checkpoint is temperature-only; salinity and joint capability is implemented and
tested using distinct physical outputs, but requires trained compatible weights.
No trained AI baseline checkpoints were supplied to the default configuration.
Italian screening is not verified non-assimilation evidence. The compact store
lacks platform/cruise identities for complete conditioning exclusion, and 2016
was used in prior model selection. Confirmation gates therefore remain false.
No accuracy, uncertainty, global-superiority, or efficiency benefit is claimed
from these software checks.
