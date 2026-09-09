from pathlib import Path
import io
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
import xarray as xr

from profile_audit.classify_matches import classify_matches, source_for_time
from profile_audit.acquire_assimilation_inputs import (
    ArchiveSource,
    acquire_inputs,
    download_archive,
    extract_archive,
    sources_for_dates,
)
from profile_audit.acquire_013030_proxy import select_proxy_files
from profile_audit.audit_oceandepths import audit_oceandepths_profiles
from profile_audit.audit_italian_en4 import audit_italian_en4, format_en4_audit_summary
from profile_audit.common import haversine_km
from profile_audit.export_oceandepths_profiles import export_oceandepths_profiles
from profile_audit.index_assimilation_inputs import index_assimilation_inputs, iter_targeted_xbt_inputs
from profile_audit.match_profiles import MatchThresholds, match_profiles, profile_fingerprint
from profile_audit.make_report import main as report_main
from profile_audit.make_report import make_report
from profile_audit.parse_italian_xbt import parse_xbt_files
from profile_audit.parse_italian_xbt import PNRA_XXXII_SHA256, _parse_profile_datetime


class TestProfileAudit(unittest.TestCase):
    def test_italian_en4_audit_separates_snapshot_coverage_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "data" / "argo_glors_ostia_ssh.zarr"
            store.parent.mkdir(parents=True)
            date = pd.Timestamp("2000-01-01T12:00:00Z")
            ds = xr.Dataset(
                {
                    "profile_source_file": (("profile",), np.array(["EN.4.2.2.200001.nc"])),
                    "profile_idx": (("profile",), np.array([3])),
                    "profile_juld": (("profile",), np.array([18262.5])),
                    "latitude": (("profile",), np.array([-60.0])),
                    "longitude": (("profile",), np.array([170.0])),
                    "argo_temp_on_glorys_depth": (("profile", "glorys_depth"), np.array([[10.0, 9.0, 8.0]], dtype=np.float32)),
                },
                coords={"profile": np.array([7]), "glorys_depth": np.array([0.0, 10.0, 20.0], dtype=np.float32)},
            )
            ds.to_zarr(store, mode="w", zarr_format=2)
            index = root / "en4_index.parquet"
            pd.DataFrame({
                "profile": [7], "profile_source_file": ["EN.4.2.2.200001.nc"], "profile_idx": [3],
                "profile_juld": [18262.5], "latitude": [-60.0], "longitude": [170.0],
            }).to_parquet(index, index=False)
            targets = root / "targets.parquet"
            pd.DataFrame({
                "profile_id": ["inside", "outside"],
                "datetime_utc": [date, pd.Timestamp("1999-01-01T00:00:00Z")],
                "latitude": [-60.0, -60.0], "longitude_180": [170.0, 170.0],
            }).to_parquet(targets, index=False)
            observations = root / "observations.parquet"
            pd.DataFrame({
                "profile_id": ["inside"] * 3,
                "depth_3": [0.0, 10.0, 20.0],
                "temperature_2": [10.0, 9.0, 8.0],
                "depth_2": [0.0, 10.0, 20.0],
                "depth_1": [0.0, 10.0, 20.0],
                "temperature_1": [10.0, 9.0, 8.0],
            }).to_parquet(observations, index=False)
            matches = root / "matches.parquet"
            summary = root / "summary.parquet"
            audit_italian_en4(targets, observations, index, root, matches, summary)
            match_frame = pd.read_parquet(matches)
            summary_frame = pd.read_parquet(summary).sort_values("profile_id")
            self.assertIn("Candidate", match_frame["classification"].tolist())
            self.assertEqual(summary_frame.loc[summary_frame["profile_id"] == "outside", "classification"].iloc[0], "Outside EN4 snapshot coverage")
            self.assertEqual(summary_frame.loc[summary_frame["profile_id"] == "inside", "candidate_count"].iloc[0], 1)

    def test_italian_en4_audit_handles_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "data" / "argo_glors_ostia_ssh.zarr"
            store.parent.mkdir(parents=True)
            xr.Dataset(
                {
                    "profile_source_file": (("profile",), np.array(["EN.4.2.2.200001.nc"])),
                    "profile_idx": (("profile",), np.array([3])),
                    "profile_juld": (("profile",), np.array([18262.5])),
                    "latitude": (("profile",), np.array([0.0])),
                    "longitude": (("profile",), np.array([0.0])),
                },
                coords={"profile": np.array([7])},
            ).to_zarr(store, mode="w", zarr_format=2)
            index = root / "en4_index.parquet"
            pd.DataFrame({
                "profile": [7], "profile_source_file": ["EN.4.2.2.200001.nc"], "profile_idx": [3],
                "profile_juld": [18262.5], "latitude": [0.0], "longitude": [0.0],
            }).to_parquet(index, index=False)
            targets = root / "targets.parquet"
            pd.DataFrame({
                "profile_id": ["target"], "datetime_utc": [pd.Timestamp("2018-01-01T00:00:00Z")],
                "latitude": [75.0], "longitude": [-140.0],
            }).to_parquet(targets, index=False)
            observations = root / "observations.parquet"
            pd.DataFrame(columns=["profile_id", "depth_m", "temperature_c"]).to_parquet(observations, index=False)
            matches = root / "matches.parquet"
            summary = root / "summary.parquet"
            audit_italian_en4(targets, observations, index, root, matches, summary)
            self.assertEqual(pd.read_parquet(summary).loc[0, "classification"], "No archive match")

    def test_en4_audit_summary_explains_match_classes(self) -> None:
        output = format_en4_audit_summary(pd.DataFrame({
            "classification": ["Near-exact", "Probable duplicate", "Ambiguous", "Candidate", "No archive match"],
            "en4_coverage_status": ["within_oceandepths_en4_snapshot"] * 5,
            "candidate_count": [1, 2, 3, 4, 0],
        }))
        self.assertIn("Strong EN4 duplicate evidence: 2", output)
        self.assertIn("Candidate evidence requiring review: 2", output)
        self.assertIn("No EN4 candidate within 24 hours/25 km: 1", output)
        self.assertIn("Spatiotemporal EN4 candidate pairs inspected: 10", output)

    def test_013030_proxy_file_selection_is_spatiotemporal(self) -> None:
        index_text = """# Format version : 3.0
# product_id,file_name,geospatial_lat_min,geospatial_lat_max,geospatial_lon_min,geospatial_lon_max,time_coverage_start,time_coverage_end,institution,date_update,data_mode,parameters
COP,history/XB/AR_PR_XB_NEAR.nc,-61,-59,169,171,2021-12-31T00:00:00Z,2022-01-02T00:00:00Z,IFREMER,2022-01-03,D,PRES TEMP
COP,history/XB/AR_PR_XB_FAR.nc,10,20,20,30,2021-12-31T00:00:00Z,2022-01-02T00:00:00Z,IFREMER,2022-01-03,D,PRES TEMP
COP,history/DB/AR_TS_DB_NEAR.nc,-61,-59,169,171,2021-12-31T00:00:00Z,2022-01-02T00:00:00Z,IFREMER,2022-01-03,D,TEMP
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index_history.txt"
            index.write_text(index_text, encoding="utf-8")
            targets = pd.DataFrame({
                "time": pd.to_datetime(["2022-01-01T12:00:00Z"], utc=True),
                "latitude": [-60.0],
                "longitude": [170.0],
            })
            output = root / "files.txt"
            selected = select_proxy_files(index, targets, output)
            self.assertEqual(selected, ["history/XB/AR_PR_XB_NEAR.nc"])

    def test_post_2021_timeline_marks_internal_snapshot_unavailable(self) -> None:
        source, status = source_for_time(pd.Timestamp("2022-01-01", tz="UTC"))
        self.assertEqual(source, "INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047")
        self.assertEqual(status, "documented_internal_snapshot_unavailable")

    def test_streaming_en4_audit_separates_absence_and_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ocean_index = root / "ocean.parquet"
            candidate_index = root / "cora50.parquet"
            origin = pd.Timestamp("1950-01-01", tz="UTC")
            dates = pd.to_datetime(
                ["2014-06-01T12:00:00Z", "2015-06-01T12:00:00Z", "2018-01-01T00:00:00Z"],
                utc=True,
            )
            pd.DataFrame({
                "profile": [0, 1, 2],
                "profile_source_file": ["EN.201406.nc", "EN.201506.nc", "EN.201801.nc"],
                "profile_idx": [0, 0, 0],
                "profile_juld": [(date - origin).total_seconds() / 86400 for date in dates],
                "latitude": [-60.0, -20.0, 10.0],
                "longitude": [179.99, 20.0, 30.0],
            }).to_parquet(ocean_index, index=False)
            pd.DataFrame({
                "profile_id": ["cora:1"],
                "datetime_utc": pd.to_datetime(["2014-06-01T12:04:00Z"], utc=True),
                "latitude": [-60.0],
                "longitude": [-179.99],
                "source_file": ["XB.nc"],
                "source_profile_index": [1],
                "calendar_day": ["20140601"],
            }).to_parquet(candidate_index, index=False)
            output = root / "audit.parquet"
            audit_oceandepths_profiles(
                ocean_index,
                {"CORA5.0": candidate_index},
                output,
                revision="test-revision",
                batch_size=2,
            )
            audit = pd.read_parquet(output).sort_values("target_profile")
            self.assertEqual(
                audit["audit_status"].tolist(),
                ["candidate_requires_fingerprint", "no_spatiotemporal_candidate", "source_unresolved"],
            )
            self.assertEqual(audit.iloc[0]["primary_candidate_count"], 1)

    def test_en4_audit_labels_013030_as_nonhistorical_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ocean_index = root / "ocean.parquet"
            proxy_index = root / "proxy.parquet"
            date = pd.Timestamp("2022-01-01T12:00:00Z")
            origin = pd.Timestamp("1950-01-01", tz="UTC")
            pd.DataFrame({
                "profile": [0],
                "profile_source_file": ["EN.202201.nc"],
                "profile_idx": [0],
                "profile_juld": [(date - origin).total_seconds() / 86400],
                "latitude": [-50.0],
                "longitude": [170.0],
            }).to_parquet(ocean_index, index=False)
            pd.DataFrame({
                "profile_id": ["proxy:1"],
                "datetime_utc": pd.to_datetime(["2022-01-01T12:05:00Z"], utc=True),
                "latitude": [-50.0],
                "longitude": [170.01],
                "source_file": ["proxy.nc"],
                "source_profile_index": [1],
                "calendar_day": ["20220101"],
            }).to_parquet(proxy_index, index=False)
            output = root / "audit.parquet"
            audit_oceandepths_profiles(
                ocean_index,
                {},
                output,
                revision="test",
                proxy_013030=proxy_index,
            )
            row = pd.read_parquet(output).iloc[0]
            self.assertEqual(row["audit_status"], "proxy_candidate_requires_fingerprint")
            self.assertEqual(row["historical_coverage_status"], "internal_snapshot_unavailable")
            self.assertEqual(row["evidence_scope"], "current_013030_proxy")

    def test_archive_download_retries_from_partial_file(self) -> None:
        source = ArchiveSource(
            key="test", archive_version="TEST", start_year=2000, end_year=2000,
            url="https://example.test/archive.tar", filename="archive.tar", expected_size=6,
            doi=None, status="public_archive", note="test fixture",
        )

        class Response:
            def __init__(self, status_code: int, blocks: list[bytes | Exception]) -> None:
                self.status_code = status_code
                self.blocks = blocks

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                for block in self.blocks:
                    if isinstance(block, Exception):
                        raise block
                    yield block

        with tempfile.TemporaryDirectory() as temporary:
            responses = iter([
                Response(200, [b"abc", requests.ConnectionError("timed out")]),
                Response(206, [b"def"]),
            ])
            with patch("profile_audit.acquire_assimilation_inputs.requests.get", side_effect=lambda *args, **kwargs: next(responses)) as get:
                with patch("profile_audit.acquire_assimilation_inputs.time.sleep"):
                    archive = download_archive(source, Path(temporary))
            self.assertEqual(archive.read_bytes(), b"abcdef")
            self.assertEqual(get.call_args_list[1].kwargs["headers"], {"Range": "bytes=3-"})

    def test_archive_selection_reports_public_and_blocked_periods(self) -> None:
        dates = pd.to_datetime(
            ["2013-01-01", "2015-01-01", "2016-01-01", "2019-01-01", "2022-01-01"],
            utc=True,
        )
        sources = sources_for_dates(dates)
        self.assertEqual([source.key for source in sources], ["cora41", "cora50", "cora51", "2017_202106", "nrt013047"])
        self.assertEqual([source.status for source in sources[-2:]], ["blocked_unresolved_source", "blocked_internal_product"])

    def test_xbt_archive_extraction_filters_target_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "sample.tar"
            with tarfile.open(archive, "w") as bundle:
                for name in (
                    "CORA/CO_DMQCGL01_20140101_PR_XB.nc",
                    "CORA/CO_DMQCGL01_20140201_PR_XB.nc",
                    "CORA/CO_DMQCGL01_20140101_PR_PF.nc",
                ):
                    payload = name.encode()
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    bundle.addfile(member, io.BytesIO(payload))
            extracted = extract_archive(
                archive,
                root / "extracted",
                mode="xbt",
                target_dates=pd.to_datetime(["2014-01-01"], utc=True),
                margin_days=0,
            )
            self.assertEqual([path.name for path in extracted], ["CO_DMQCGL01_20140101_PR_XB.nc"])

    def test_xbt_archive_extraction_recurses_into_selected_year_archives(self) -> None:
        def nested_archive(year: int) -> bytes:
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as bundle:
                for suffix in ("PR_XB", "PR_PF"):
                    name = f"CORA/{year}/CO_DMQCGL01_{year}0101_{suffix}.nc"
                    content = name.encode()
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
            return payload.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "sample.tar"
            with tarfile.open(archive, "w") as bundle:
                for year in (2014, 2015):
                    content = nested_archive(year)
                    member = tarfile.TarInfo(f"CORA5.0/CORA5.0_data-{year}.tar.gz")
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
            extracted = extract_archive(
                archive,
                root / "extracted",
                mode="xbt",
                target_dates=pd.to_datetime(["2014-01-01"], utc=True),
                margin_days=0,
            )
            self.assertEqual(len(extracted), 1)
            self.assertEqual(extracted[0].name, "CO_DMQCGL01_20140101_PR_XB.nc")
            self.assertIn("CORA5.0_data-2014", extracted[0].parts)
            self.assertFalse(any("2015" in str(path) for path in extracted))

    def test_targeted_xbt_stream_skips_other_members_and_year_archives(self) -> None:
        def nested_archive(year: int) -> bytes:
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as bundle:
                for date, suffix in ((f"{year}0101", "PR_XB"), (f"{year}0101", "PR_PF"), (f"{year}0201", "PR_XB")):
                    name = f"CORA/{year}/CO_DMQCGL01_{date}_{suffix}.nc"
                    content = name.encode()
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
            return payload.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "sample.tar"
            with tarfile.open(archive, "w") as bundle:
                for year in (2014, 2015):
                    content = nested_archive(year)
                    member = tarfile.TarInfo(f"CORA5.0/CORA5.0_data-{year}.tar.gz")
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
            paths = list(iter_targeted_xbt_inputs(
                [archive],
                pd.to_datetime(["2014-01-01"], utc=True),
                margin_days=0,
            ))
            self.assertEqual([path.name for path in paths], ["CO_DMQCGL01_20140101_PR_XB.nc"])

    def test_xbt_parser_accepts_station_zero_and_filters_qf(self) -> None:
        content = """// Cruise ID\tPNRAXXXIX
// Platform Name\tR/V Laura Bassi
// Probe Type\tT7
Cruise\tStation\tType\tmon/day/yr\thh:mm\tLongitude [degrees_east]\tLatitude [degrees_north]\tBot. Depth [m]\tElapsed Time [s]\tDepth 1 [m]\tDepth 2 [m]\tDepth 3 [m]\tTemperature 1 [C]\tTemperature 2 [C]\tQF
PNRAXXXIX\t0\tXBT\t01/07/2024\t11:30\t181.0\t-48.0\t760\t0.1\t1.0\t1.1\t0.5\t11.6\t11.5\t2
PNRAXXXIX\t0\tXBT\t01/07/2024\t11:30\t181.0\t-48.0\t760\t0.2\t2.0\t2.1\t1.5\t11.5\t11.4\t1
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "xbt.txt"
            source.write_text(content, encoding="utf-8")
            profiles_path = root / "profiles.parquet"
            observations_path = root / "observations.parquet"
            sensitivity_path = root / "sensitivity.parquet"
            parse_xbt_files([source], profiles_output=profiles_path, observations_output=observations_path, sensitivity_output=sensitivity_path)
            profiles = pd.read_parquet(profiles_path)
            observations = pd.read_parquet(observations_path)
            sensitivity = pd.read_parquet(sensitivity_path)
            self.assertEqual(profiles.loc[0, "station"], "0")
            self.assertAlmostEqual(profiles.loc[0, "longitude_180"], -179.0)
            self.assertEqual(profiles.loc[0, "number_of_good_levels"], 1)
            self.assertEqual(observations["qf"].tolist(), [1])
            self.assertEqual(sensitivity["qf"].tolist(), [2, 1])

    def test_xbt_parser_records_depth_reversals_without_aborting(self) -> None:
        content = """Cruise\tStation\tType\tmon/day/yr\thh:mm\tLongitude\tLatitude\tBot Depth\tElapsed\tDepth 1\tDepth 2\tDepth 3\tTemperature 1\tTemperature 2\tQF
TEST\t2\tXBT\t01/01/2010\t00:00\t170\t-60\t100\t0.1\t1\t1\t1.001\t5\t5\t1
TEST\t2\tXBT\t01/01/2010\t00:00\t170\t-60\t100\t0.2\t2\t2\t1.999\t5\t5\t1
TEST\t2\tXBT\t01/01/2010\t00:00\t170\t-60\t100\t0.3\t3\t3\t1.998\t5\t5\t1
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reversal.txt"
            source.write_text(content, encoding="utf-8")
            profiles_path = root / "profiles.parquet"
            observations_path = root / "observations.parquet"
            parse_xbt_files([source], profiles_output=profiles_path, observations_output=observations_path)
            profile = pd.read_parquet(profiles_path).iloc[0]
            self.assertEqual(profile["primary_depth_reversal_count"], 1)
            self.assertAlmostEqual(profile["primary_maximum_depth_reversal_m"], 0.001)
            self.assertFalse(profile["primary_corrected_depth_monotonic"])

    def test_xbt_parser_infers_mixed_date_formats_from_coverage(self) -> None:
        timestamp, label, reason = _parse_profile_datetime(
            "02/01/2017",
            "00:42",
            source_file="xbt_PNRA_XXXII.txt",
            source_checksum=PNRA_XXXII_SHA256,
        )
        self.assertEqual(timestamp.strftime("%Y-%m-%d"), "2017-01-02")
        self.assertEqual(label, "DMY_source_correction")
        self.assertIsNotNone(reason)

        unchanged, label, reason = _parse_profile_datetime(
            "02/01/2017",
            "00:42",
            source_file="unknown.txt",
            source_checksum="unknown",
        )
        self.assertEqual(unchanged.strftime("%Y-%m-%d"), "2017-02-01")
        self.assertEqual(label, "MDY")
        self.assertIsNone(reason)

    def test_haversine_wraps_antimeridian(self) -> None:
        self.assertLess(float(haversine_km(0, 179.9, 0, -179.9)), 23)

    def test_profile_fingerprint_and_matching(self) -> None:
        target_profiles = pd.DataFrame({
            "profile_id": ["target"], "datetime_utc": ["2014-06-01T12:00:00Z"],
            "latitude": [-60.0], "longitude_180": [179.99],
        })
        candidates = pd.DataFrame({
            "profile_id": ["candidate"], "datetime_utc": ["2014-06-01T12:04:00Z"],
            "latitude": [-60.0], "longitude": [-179.99], "archive_version": ["CORA5.0"],
            "source_file": ["XB.nc"], "source_profile_index": [4], "dc_reference": ["ref-4"],
        })
        depth = np.arange(40, dtype=float) * 5
        target_obs = pd.DataFrame({
            "profile_id": "target", "depth_3": depth, "temperature_2": 10 - depth / 100,
            "depth_2": depth, "depth_1": depth, "temperature_1": 10 - depth / 100,
        })
        candidate_obs = pd.DataFrame({
            "profile_id": "candidate", "depth_m": depth, "temperature_c": 10 - depth / 100 + 0.01,
        })
        metrics = profile_fingerprint(target_obs, candidate_obs)
        self.assertEqual(metrics["overlap_levels"], 40)
        self.assertLess(metrics["temperature_mae_c"], 0.02)
        matches = match_profiles(target_profiles, candidates, target_observations=target_obs, candidate_observations=candidate_obs)
        self.assertEqual(matches.loc[0, "classification"], "Near-exact")
        audit = classify_matches(target_profiles, matches)
        self.assertTrue(audit.loc[0, "input_archive_match"])
        self.assertTrue(audit.loc[0, "potentially_assimilated"])
        self.assertEqual(audit.loc[0, "evidence_level"], "B")
        self.assertTrue(pd.isna(audit.loc[0, "confirmed_accepted"]))

    def test_exact_source_identity_wins(self) -> None:
        targets = pd.DataFrame({
            "profile_id": ["EN.nc:3"], "profile_source_file": ["EN.nc"], "source_profile_idx": [3],
            "profile_date": [20000101], "latitude": [0.0], "longitude": [0.0],
        })
        candidates = pd.DataFrame({
            "profile_id": ["archive:3"], "source_file": ["EN.nc"], "source_profile_index": [3],
            "profile_date": [20000102], "latitude": [10.0], "longitude": [10.0], "archive_version": ["EN4.2.2"],
        })
        result = match_profiles(targets, candidates)
        self.assertEqual(result.loc[0, "classification"], "Exact")
        self.assertEqual(result.loc[0, "exact_identifier"], "source_file_index")

    def test_index_generic_en4_netcdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "EN.nc"
            ds = xr.Dataset({
                "JULD": (("N_PROF",), np.array([18262.5])),
                "LATITUDE": (("N_PROF",), np.array([1.0])),
                "LONGITUDE": (("N_PROF",), np.array([2.0])),
                "DEPH_CORRECTED": (("N_PROF", "N_LEVELS"), np.array([[0.0, 10.0]], dtype=np.float32)),
                "TEMP": (("N_PROF", "N_LEVELS"), np.array([[20.0, 19.0]], dtype=np.float32)),
                "TEMP_QC": (("N_PROF", "N_LEVELS"), np.array([[b"1", b"2"]])),
            })
            ds["JULD"].attrs["units"] = "days since 1950-01-01 00:00:00 utc"
            ds.to_netcdf(source, engine="scipy")
            profiles_path = root / "indexed_profiles.parquet"
            observations_path = root / "indexed_observations.parquet"
            index_assimilation_inputs([source], archive_version="EN4.2.2", profiles_output=profiles_path, observations_output=observations_path)
            profiles = pd.read_parquet(profiles_path)
            observations = pd.read_parquet(observations_path)
            self.assertEqual(profiles.loc[0, "archive_version"], "EN4.2.2")
            self.assertEqual(observations["temperature_qc"].tolist(), [1, 2])

    def test_index_target_profiles_retains_only_nearby_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "CORA.nc"
            dataset = xr.Dataset({
                "JULD": (("N_PROF",), np.array([18262.5, 18262.5])),
                "LATITUDE": (("N_PROF",), np.array([1.0, 40.0])),
                "LONGITUDE": (("N_PROF",), np.array([2.0, 40.0])),
                "DEPH_CORRECTED": (("N_PROF", "N_LEVELS"), np.array([[0.0], [0.0]], dtype=np.float32)),
                "TEMP": (("N_PROF", "N_LEVELS"), np.array([[20.0], [10.0]], dtype=np.float32)),
            })
            dataset["JULD"].attrs["units"] = "days since 1950-01-01 00:00:00 utc"
            dataset.to_netcdf(source, engine="scipy")
            profiles_path = root / "profiles.parquet"
            observations_path = root / "observations.parquet"
            targets = pd.DataFrame({
                "profile_id": ["target"], "datetime_utc": [pd.Timestamp("2000-01-01T12:00:00Z")],
                "latitude": [1.0], "longitude": [2.0],
            })
            index_assimilation_inputs(
                [source], archive_version="CORA4.1", profiles_output=profiles_path,
                observations_output=observations_path, target_dates=targets["datetime_utc"].tolist(),
                target_profiles=targets,
            )
            self.assertEqual(len(pd.read_parquet(profiles_path)), 1)
            self.assertEqual(len(pd.read_parquet(observations_path)), 1)

    def test_index_target_profiles_allows_candidates_without_valid_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "CORA.nc"
            dataset = xr.Dataset({
                "JULD": (("N_PROF",), np.array([18262.5])),
                "LATITUDE": (("N_PROF",), np.array([1.0])),
                "LONGITUDE": (("N_PROF",), np.array([2.0])),
            })
            dataset["JULD"].attrs["units"] = "days since 1950-01-01 00:00:00 utc"
            dataset.to_netcdf(source, engine="scipy")
            profiles_path = root / "profiles.parquet"
            observations_path = root / "observations.parquet"
            targets = pd.DataFrame({
                "profile_id": ["target"], "datetime_utc": [pd.Timestamp("2000-01-01T12:00:00Z")],
                "latitude": [1.0], "longitude": [2.0],
            })

            index_assimilation_inputs(
                [source], archive_version="CORA4.1", profiles_output=profiles_path,
                observations_output=observations_path, target_dates=targets["datetime_utc"].tolist(),
                target_profiles=targets,
            )

            self.assertEqual(len(pd.read_parquet(profiles_path)), 1)
            self.assertEqual(pd.read_parquet(observations_path).columns.tolist(), [
                "profile_id", "level_index", "depth_m", "temperature_c", "salinity", "temperature_qc",
                "salinity_qc", "source_vertical_variable",
            ])

    def test_index_broadcasts_shared_vertical_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "CORA.nc"
            dataset = xr.Dataset({
                "JULD": (("N_PROF",), np.array([18262.0, 18262.5])),
                "LATITUDE": (("N_PROF",), np.array([1.0, 1.0])),
                "LONGITUDE": (("N_PROF",), np.array([2.0, 2.0])),
                "DEPH": (("N_LEVELS",), np.array([0.0, 10.0], dtype=np.float32)),
                "TEMP": (("N_PROF", "N_LEVELS"), np.array([[20.0, 19.0], [21.0, 20.0]], dtype=np.float32)),
            })
            dataset["JULD"].attrs["units"] = "days since 1950-01-01 00:00:00 utc"
            dataset.to_netcdf(source, engine="scipy")
            profiles_path = root / "profiles.parquet"
            observations_path = root / "observations.parquet"
            index_assimilation_inputs(
                [source], archive_version="CORA4.1", profiles_output=profiles_path,
                observations_output=observations_path,
            )
            observations = pd.read_parquet(observations_path)
            self.assertEqual(len(observations), 4)
            self.assertEqual(observations.groupby("profile_id")["depth_m"].max().tolist(), [10.0, 10.0])

    def test_index_broadcasts_scalar_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "profile.nc"
            dataset = xr.Dataset({
                "TIME": (("TIME",), np.array([0.0, 1.0])),
                "LATITUDE": ((), np.float32(12.5)),
                "LONGITUDE": ((), np.float32(45.0)),
            })
            dataset["TIME"].attrs["units"] = "days since 1950-01-01 00:00:00"
            dataset.to_netcdf(source, engine="scipy")
            profiles_path = root / "profiles.parquet"

            index_assimilation_inputs(
                [source],
                archive_version="013_030_PROXY",
                profiles_output=profiles_path,
                observations_output=None,
                metadata_only=True,
            )

            indexed = pd.read_parquet(profiles_path)
            self.assertEqual(indexed["latitude"].tolist(), [12.5, 12.5])
            self.assertEqual(indexed["longitude"].tolist(), [45.0, 45.0])

    def test_metadata_index_streams_netcdf_from_tar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            netcdf = root / "CO_DMQCGL01_20140101_PR_XB.nc"
            empty = root / "CO_DMQCGL01_20140101_TS_MO.nc"
            dataset = xr.Dataset({
                "JULD": (("N_PROF",), np.array([23376.0])),
                "LATITUDE": (("N_PROF",), np.array([-60.0])),
                "LONGITUDE": (("N_PROF",), np.array([170.0])),
            })
            dataset["JULD"].attrs["units"] = "days since 1950-01-01 00:00:00 utc"
            dataset.to_netcdf(netcdf, engine="scipy")
            empty.write_bytes(b"\0" * 32)
            archive = root / "cora.tar"
            with tarfile.open(archive, "w") as bundle:
                bundle.add(empty, arcname=f"data1/cora/{empty.name}")
                bundle.add(netcdf, arcname=f"data1/cora/{netcdf.name}")
            profiles = root / "profiles.parquet"
            index_assimilation_inputs(
                [archive],
                archive_version="CORA5.0",
                profiles_output=profiles,
                observations_output=None,
                metadata_only=True,
            )
            indexed = pd.read_parquet(profiles)
            self.assertEqual(indexed.loc[0, "source_file"], netcdf.name)
            self.assertEqual(indexed.loc[0, "archive_version"], "CORA5.0")

    def test_metadata_index_streams_nested_tar_netcdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            netcdf = root / "profile.nc"
            dataset = xr.Dataset({
                "JULD": (("N_PROF",), np.array([0.0])),
                "REFERENCE_DATE_TIME": ((), np.bytes_("19500101T000000Z")),
                "LATITUDE": (("N_PROF",), np.array([-60.0])),
                "LONGITUDE": (("N_PROF",), np.array([170.0])),
            })
            dataset["JULD"].attrs["units"] = "days since REFERENCE_DATE_TIME"
            dataset.to_netcdf(netcdf, engine="scipy")

            nested_bytes = io.BytesIO()
            with tarfile.open(fileobj=nested_bytes, mode="w:gz") as nested_bundle:
                nested_bundle.add(netcdf, arcname="profiles/profile.nc")
            nested_bytes.seek(0)
            archive = root / "cora5.tar"
            with tarfile.open(archive, "w") as bundle:
                member = tarfile.TarInfo("CORA5.0/CORA5.0_data-1950.tar.gz")
                member.size = len(nested_bytes.getbuffer())
                bundle.addfile(member, nested_bytes)

            profiles = root / "profiles.parquet"
            index_assimilation_inputs(
                [archive],
                archive_version="CORA5.0",
                profiles_output=profiles,
                observations_output=None,
                metadata_only=True,
            )

            indexed = pd.read_parquet(profiles)
            self.assertEqual(len(indexed), 1)
            self.assertEqual(indexed.loc[0, "archive_version"], "CORA5.0")

    def test_index_resolves_reference_date_time_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "CORA.nc"
            dataset = xr.Dataset({
                "JULD": (("N_PROF",), np.array([0.5])),
                "REFERENCE_DATE_TIME": ((), np.bytes_("19500101T000000Z")),
                "LATITUDE": (("N_PROF",), np.array([1.0])),
                "LONGITUDE": (("N_PROF",), np.array([2.0])),
            })
            dataset["JULD"].attrs["units"] = "days since REFERENCE_DATE_TIME"
            dataset.to_netcdf(source, engine="scipy")
            profiles_path = root / "profiles.parquet"

            index_assimilation_inputs(
                [source],
                archive_version="CORA4.1",
                profiles_output=profiles_path,
                observations_output=None,
                metadata_only=True,
            )

            indexed = pd.read_parquet(profiles_path)
            self.assertEqual(indexed.loc[0, "datetime_utc"], pd.Timestamp("1950-01-01T12:00:00Z"))

    def test_export_oceandepths_rejects_bad_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "data" / "argo_glors_ostia_ssh.zarr"
            store.parent.mkdir(parents=True)
            ds = xr.Dataset(
                {
                    "profile_source_file": (("profile",), np.array(["EN.4.2.2.200001.nc", "EN.4.2.2.200001.nc"])),
                    "profile_idx": (("profile",), np.array([0, 1])),
                    "profile_date": (("profile",), np.array([20000101, 20000101])),
                    "profile_juld": (("profile",), np.array([18262.0, 18262.0])),
                    "latitude": (("profile",), np.array([1.0, -999.99])),
                    "longitude": (("profile",), np.array([2.0, -999.99])),
                    "argo_temp_on_glorys_depth": (("profile", "glorys_depth"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
                },
                coords={"profile": np.array([0, 1]), "glorys_depth": np.array([0.5, 1.5], dtype=np.float32)},
            )
            ds.to_zarr(store, mode="w", zarr_format=2)
            output = root / "export.parquet"
            export_oceandepths_profiles(root, output, max_profiles=2)
            frame = pd.read_parquet(output)
            self.assertEqual(frame["valid_coordinate"].tolist(), [True, False])
            self.assertEqual(frame.loc[0, "source_product"], "EN4.2.2 profile archive")

    def test_report_outputs(self) -> None:
        audit = pd.DataFrame({
            "profile_id": ["italian-xbt:PNRAX:0"], "cruise": ["PNRAX"],
            "datetime_utc": ["2014-01-01T00:00:00Z"], "classification": ["No archive match"],
            "input_archive_match": [False], "potentially_assimilated": [False],
            "confirmed_accepted": pd.array([pd.NA], dtype="boolean"), "evidence_level": ["none"],
            "latitude": [-60.0], "longitude_180": [170.0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            outputs = make_report(audit, Path(temporary), make_plots=False)
            self.assertTrue(all(path.exists() for path in outputs.values()))

    def test_report_prints_summary(self) -> None:
        audit = pd.DataFrame({
            "profile_id": ["italian-xbt:PNRAX:0", "other:1"],
            "cruise": ["PNRAX", "other"],
            "datetime_utc": ["2014-01-01T00:00:00Z", "2015-01-01T00:00:00Z"],
            "classification": ["No archive match", "Ambiguous"],
            "input_archive_match": [False, True],
            "potentially_assimilated": [False, True],
            "confirmed_accepted": pd.array([pd.NA, True], dtype="boolean"),
            "evidence_level": ["none", "A"],
            "source_timeline_status": ["documented_internal_snapshot_unavailable", "unresolved"],
            "candidate_profile_id": [None, None],
            "latitude": [-60.0, -61.0],
            "longitude_180": [170.0, 171.0],
        })
        with tempfile.TemporaryDirectory() as temporary, patch("sys.stdout", new_callable=io.StringIO) as stdout:
            audit_path = Path(temporary) / "audit.parquet"
            audit.to_parquet(audit_path, index=False)
            report_main(["--audit", str(audit_path), "--output-dir", temporary, "--no-plots"])
            output = stdout.getvalue()
            self.assertIn("Report summary", output)
            self.assertIn("Profiles audited: 2", output)
            self.assertIn("Italian XBT profiles: 1", output)
            self.assertIn("Ambiguous matches: 1", output)
            self.assertIn("Checked against current 013_030 proxy (July 2021 onward): 1", output)
            self.assertIn("No proxy candidate by ID or 24-hour/25-km search: 1", output)


if __name__ == "__main__":
    unittest.main()
