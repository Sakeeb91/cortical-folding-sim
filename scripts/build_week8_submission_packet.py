"""Build the Week 8 submission packet, frozen bundle manifest, and summary artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default="PLAN.md", help="Execution plan path.")
    parser.add_argument(
        "--forward-summary",
        default="results/forward_sweep_summary.json",
        help="Forward-sweep summary artifact.",
    )
    parser.add_argument(
        "--forward-manifest",
        default="results/forward_sweep_manifest.json",
        help="Forward-sweep manifest artifact.",
    )
    parser.add_argument(
        "--forward-gate-report",
        default="results/validation_gate_report.json",
        help="Forward-sweep gate report artifact.",
    )
    parser.add_argument(
        "--week3-comparison",
        default="results/week3_anisotropy_comparison.json",
        help="Week 3 comparison artifact.",
    )
    parser.add_argument(
        "--week4-comparison",
        default="results/week4_collision_comparison.json",
        help="Week 4 comparison artifact.",
    )
    parser.add_argument(
        "--week5-comparison",
        default="results/week5_layered_comparison.json",
        help="Week 5 comparison artifact.",
    )
    parser.add_argument(
        "--week6-summary",
        default="results/week6_figure_pipeline_summary.json",
        help="Week 6 summary artifact.",
    )
    parser.add_argument(
        "--week6-manifest",
        default="docs/assets/week6_figure_manifest.json",
        help="Week 6 figure manifest artifact.",
    )
    parser.add_argument(
        "--week7-pack-summary",
        default="results/week7_animation_pack_summary.json",
        help="Week 7 orchestration summary artifact.",
    )
    parser.add_argument(
        "--week7-animation-summary",
        default="results/week7_animation_comparison_summary.json",
        help="Week 7 animation summary artifact.",
    )
    parser.add_argument(
        "--week7-manifest",
        default="docs/assets/week7_animation_manifest.json",
        help="Week 7 animation manifest artifact.",
    )
    parser.add_argument(
        "--week7-results-index",
        default="results/week7_results_index_summary.json",
        help="Week 7 results-index summary artifact.",
    )
    parser.add_argument(
        "--week7-hardened-validation",
        default="results/week7_hardened_validation.json",
        help="Week 7 hardened validation artifact.",
    )
    parser.add_argument(
        "--waiver-json",
        default="",
        help="Optional JSON file mapping criterion IDs to waiver rationales.",
    )
    parser.add_argument(
        "--output-packet-doc",
        default="docs/week8_methods_results_packet.md",
        help="Methods/results packet markdown output path.",
    )
    parser.add_argument(
        "--output-captions-doc",
        default="docs/assets/week8_figure_captions.md",
        help="Figure-caption markdown output path.",
    )
    parser.add_argument(
        "--output-methods-settings-json",
        default="results/week8_methods_settings_freeze.json",
        help="Methods/settings freeze JSON output path.",
    )
    parser.add_argument(
        "--output-tables-json",
        default="results/week8_final_tables.json",
        help="Final tables JSON output path.",
    )
    parser.add_argument(
        "--output-captions-json",
        default="results/week8_figure_captions.json",
        help="Figure captions JSON output path.",
    )
    parser.add_argument(
        "--output-commands-json",
        default="results/week8_reproducibility_commands.json",
        help="Submission-ready commands JSON output path.",
    )
    parser.add_argument(
        "--output-bundle-json",
        default="results/week8_frozen_artifact_bundle.json",
        help="Frozen artifact-bundle manifest output path.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="results/week8_submission_packet_summary.json",
        help="Week 8 packet summary output path.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    p = Path(path)
    with p.open() as f:
        return json.load(f)


def maybe_load_waivers(path: str) -> dict[str, str]:
    if not path:
        return {}
    return {str(k): str(v) for k, v in load_json(path).items()}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path_str: str) -> dict:
    path = Path(path_str)
    if path.exists():
        return {
            "path": path_str,
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    return {
        "path": path_str,
        "exists": False,
        "size_bytes": 0,
        "sha256": None,
    }


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict) -> None:
    ensure_parent(path)
    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)


def write_text(path: str | Path, text: str) -> None:
    ensure_parent(path)
    Path(path).write_text(text)


def methods_settings_freeze(config_paths: list[str]) -> dict:
    entries = []
    for config_path in config_paths:
        p = Path(config_path)
        entries.append(
            {
                "path": config_path,
                "exists": p.exists(),
                "sha256": sha256_path(p) if p.exists() else None,
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configs": entries,
    }


def build_top_level_criteria(
    *,
    forward_summary: dict,
    forward_gate_report: dict,
    week5_comparison: dict,
    week6_summary: dict,
    week6_manifest: dict,
    week7_pack_summary: dict,
    week7_animation_summary: dict,
    week7_manifest: dict,
    week7_results_index: dict,
    waivers: dict[str, str],
) -> list[dict]:
    improvements = {
        "gi_increase": float(week5_comparison.get("delta_gi_layered_minus_baseline", 0.0)) > 0.0,
        "outside_skull_frac_decrease": float(
            week5_comparison.get("delta_outside_skull_frac_layered_minus_baseline", 0.0)
        )
        < 0.0,
        "disp_p95_decrease": float(week5_comparison.get("delta_disp_p95_layered_minus_baseline", 0.0))
        < 0.0,
    }
    n_improved = sum(int(v) for v in improvements.values())

    criteria = [
        {
            "id": "S1",
            "description": "Stability rate >= 95% on benchmark grid.",
            "passed": float(forward_summary.get("stability_rate", 0.0)) >= 0.95
            and bool(forward_gate_report.get("passed")),
            "observed": {
                "forward_stability_rate": float(forward_summary.get("stability_rate", 0.0)),
                "validation_gate_passed": bool(forward_gate_report.get("passed")),
            },
        },
        {
            "id": "S2",
            "description": "Improved realism beats baseline on at least two morphology metrics.",
            "passed": n_improved >= 2,
            "observed": {
                "n_improved_metrics": n_improved,
                "improvements": improvements,
                "week7_delta_gi": float(week7_animation_summary.get("delta_gi_improved_minus_baseline", 0.0)),
                "week7_delta_disp_p95": float(
                    week7_animation_summary.get("delta_disp_p95_improved_minus_baseline", 0.0)
                ),
                "week7_delta_outside_skull_frac": float(
                    week7_animation_summary.get("delta_outside_skull_frac_improved_minus_baseline", 0.0)
                ),
            },
        },
        {
            "id": "S3",
            "description": "Main figures and animations regenerate from one documented command path.",
            "passed": bool(week6_summary.get("acceptance_one_command_regeneration_succeeds"))
            and bool(week6_manifest.get("all_figures_have_source_run_ids"))
            and bool(week7_pack_summary.get("acceptance_animation_regeneration_succeeds"))
            and bool(week7_manifest.get("all_assets_have_source_run_ids")),
            "observed": {
                "week6_regen": bool(week6_summary.get("acceptance_one_command_regeneration_succeeds")),
                "week6_figure_mapping": bool(week6_manifest.get("all_figures_have_source_run_ids")),
                "week7_regen": bool(week7_pack_summary.get("acceptance_animation_regeneration_succeeds")),
                "week7_animation_mapping": bool(week7_manifest.get("all_assets_have_source_run_ids")),
            },
        },
        {
            "id": "S4",
            "description": "Every reported metric links to saved config + summary artifact.",
            "passed": bool(forward_gate_report.get("passed"))
            and bool(week6_manifest.get("all_figures_have_source_run_ids"))
            and bool(week7_manifest.get("all_assets_have_source_run_ids"))
            and bool(week7_results_index.get("all_claims_have_linked_artifacts")),
            "observed": {
                "forward_gate_passed": bool(forward_gate_report.get("passed")),
                "week6_manifest_mapped": bool(week6_manifest.get("all_figures_have_source_run_ids")),
                "week7_manifest_mapped": bool(week7_manifest.get("all_assets_have_source_run_ids")),
                "results_index_claims_linked": bool(
                    week7_results_index.get("all_claims_have_linked_artifacts")
                ),
            },
        },
    ]

    for criterion in criteria:
        waiver_reason = waivers.get(criterion["id"], "")
        criterion["waived"] = bool(waiver_reason)
        criterion["waiver_reason"] = waiver_reason
        criterion["effective_pass"] = bool(criterion["passed"] or criterion["waived"])
    return criteria


def build_tables(
    *,
    forward_summary: dict,
    week3_comparison: dict,
    week4_comparison: dict,
    week5_comparison: dict,
    week7_animation_summary: dict,
) -> dict:
    return {
        "morphology_deltas": [
            {
                "comparison": "Week 3 anisotropic vs isotropic",
                "delta_gi": float(week3_comparison.get("delta_gi", 0.0)),
                "delta_curv_p90": float(week3_comparison.get("delta_curv_p90", 0.0)),
                "delta_disp_p95": float(week3_comparison.get("delta_disp_p95", 0.0)),
                "stable_both": bool(
                    int(week3_comparison.get("baseline_stable", 0))
                    and int(week3_comparison.get("anisotropic_stable", 0))
                ),
            },
            {
                "comparison": "Week 4 spatial-hash collision vs sampled",
                "delta_collision_overlap_count": float(
                    week4_comparison.get("reduction_overlap_count_vs_sampled", 0.0)
                ),
                "runtime_ratio_spatial_over_sampled": float(
                    week4_comparison.get("runtime_ratio_spatial_over_sampled", 0.0)
                ),
                "acceptance_collision_outliers_reduced": bool(
                    week4_comparison.get("acceptance_collision_outliers_reduced")
                ),
            },
            {
                "comparison": "Week 5 layered vs non-layered",
                "delta_gi": float(week5_comparison.get("delta_gi_layered_minus_baseline", 0.0)),
                "delta_outside_skull_frac": float(
                    week5_comparison.get("delta_outside_skull_frac_layered_minus_baseline", 0.0)
                ),
                "delta_disp_p95": float(
                    week5_comparison.get("delta_disp_p95_layered_minus_baseline", 0.0)
                ),
                "runtime_ratio_layered_over_baseline": float(
                    week5_comparison.get("runtime_ratio_layered_over_baseline", 0.0)
                ),
            },
        ],
        "final_reference_metrics": {
            "forward_stability_rate": float(forward_summary.get("stability_rate", 0.0)),
            "forward_gi_plausible_rate": float(forward_summary.get("gi_plausible_rate", 0.0)),
            "week7_delta_gi_improved_minus_baseline": float(
                week7_animation_summary.get("delta_gi_improved_minus_baseline", 0.0)
            ),
            "week7_delta_disp_p95_improved_minus_baseline": float(
                week7_animation_summary.get("delta_disp_p95_improved_minus_baseline", 0.0)
            ),
            "week7_delta_outside_skull_frac_improved_minus_baseline": float(
                week7_animation_summary.get("delta_outside_skull_frac_improved_minus_baseline", 0.0)
            ),
        },
    }


def build_captions(
    *,
    week3_comparison: dict,
    week4_comparison: dict,
    week5_comparison: dict,
    week7_animation_summary: dict,
) -> list[dict]:
    return [
        {
            "figure_id": "week3_anisotropy_delta",
            "artifact_path": "docs/assets/week3_anisotropy_delta.png",
            "caption": (
                "Directional growth anisotropy changes fold morphology relative to isotropic growth "
                f"(delta GI={float(week3_comparison.get('delta_gi', 0.0)):.4f}, "
                f"delta curvature p90={float(week3_comparison.get('delta_curv_p90', 0.0)):.4f})."
            ),
        },
        {
            "figure_id": "week4_collision_ablation",
            "artifact_path": "docs/assets/week4_collision_ablation.png",
            "caption": (
                "Spatial-hash collision handling reduces overlap-count outliers versus sampled collision checks "
                f"(delta overlap count={float(week4_comparison.get('reduction_overlap_count_vs_sampled', 0.0)):.1f}) "
                "while remaining within runtime budget."
            ),
        },
        {
            "figure_id": "week5_layered_ablation",
            "artifact_path": "docs/assets/week5_layered_ablation.png",
            "caption": (
                "Layered approximation improves GI and reduces skull penetration fraction on the reference setting "
                f"(delta GI={float(week5_comparison.get('delta_gi_layered_minus_baseline', 0.0)):.4f}, "
                f"delta outside skull frac={float(week5_comparison.get('delta_outside_skull_frac_layered_minus_baseline', 0.0)):.4f})."
            ),
        },
        {
            "figure_id": "week7_baseline_vs_improved",
            "artifact_path": "docs/assets/week7_baseline_vs_improved.gif",
            "caption": (
                "Deterministic baseline-vs-improved animation comparison from a shared source pipeline "
                f"(delta GI={float(week7_animation_summary.get('delta_gi_improved_minus_baseline', 0.0)):.4f}, "
                f"delta disp p95={float(week7_animation_summary.get('delta_disp_p95_improved_minus_baseline', 0.0)):.4f})."
            ),
        },
    ]


def build_command_list() -> list[dict]:
    return [
        {"id": "tests", "command": "python3.11 -m pytest tests -q"},
        {"id": "ci_quick", "command": "./scripts/run_validation_quick.sh"},
        {"id": "ci_full", "command": "./scripts/run_validation_full.sh"},
        {
            "id": "week6_regen",
            "command": "MPLBACKEND=Agg python3.11 scripts/regenerate_week6_figures.py --n-steps 140",
        },
        {
            "id": "week7_regen",
            "command": "MPLBACKEND=Agg python3.11 scripts/regenerate_week7_animation_pack.py --n-steps 140",
        },
        {
            "id": "week8_regen",
            "command": "MPLBACKEND=Agg python3.11 scripts/regenerate_week8_final_package.py --n-steps 140",
        },
        {
            "id": "week8_validate",
            "command": "MPLBACKEND=Agg python3.11 scripts/validate_week8_hardened.py --n-steps 140",
        },
    ]


def packet_markdown(
    *,
    criteria: list[dict],
    tables: dict,
    commands: list[dict],
    plan_path: str,
) -> str:
    morph_rows = tables["morphology_deltas"]
    lines = [
        "# Week 8 Methods and Results Packet",
        "",
        f"Source plan: `{plan_path}`",
        "",
        "## Acceptance Criteria Status",
        "",
        "| ID | Criterion | Passed | Waived | Effective pass |",
        "|---|---|---|---|---|",
    ]
    for criterion in criteria:
        lines.append(
            "| {id} | {desc} | {passed} | {waived} | {effective} |".format(
                id=criterion["id"],
                desc=criterion["description"],
                passed=str(bool(criterion["passed"])).lower(),
                waived=str(bool(criterion["waived"])).lower(),
                effective=str(bool(criterion["effective_pass"])).lower(),
            )
        )

    lines.extend(
        [
            "",
            "## Final Metrics Table",
            "",
            "| Comparison | Delta GI | Delta Curvature p90 | Delta Disp p95 | Notes |",
            "|---|---|---|---|---|",
            "| Week 3 anisotropic vs isotropic | "
            f"{morph_rows[0]['delta_gi']:.6f} | {morph_rows[0]['delta_curv_p90']:.6f} | "
            f"{morph_rows[0]['delta_disp_p95']:.6f} | stable_both={str(morph_rows[0]['stable_both']).lower()} |",
            "| Week 4 spatial-hash vs sampled collision | - | - | - | "
            f"delta_overlap_count={morph_rows[1]['delta_collision_overlap_count']:.1f}, "
            f"runtime_ratio={morph_rows[1]['runtime_ratio_spatial_over_sampled']:.4f} |",
            "| Week 5 layered vs non-layered | "
            f"{morph_rows[2]['delta_gi']:.6f} | - | {morph_rows[2]['delta_disp_p95']:.6f} | "
            f"delta_outside={morph_rows[2]['delta_outside_skull_frac']:.6f} |",
            "",
            "## Reproducibility Commands",
            "",
        ]
    )
    for cmd in commands:
        lines.append(f"1. `{cmd['command']}`")
    lines.append("")
    return "\n".join(lines)


def captions_markdown(captions: list[dict]) -> str:
    lines = ["# Week 8 Figure Captions", ""]
    for i, cap in enumerate(captions, start=1):
        lines.append(f"{i}. `{cap['artifact_path']}`")
        lines.append(f"   {cap['caption']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    forward_summary = load_json(args.forward_summary)
    forward_manifest = load_json(args.forward_manifest)
    forward_gate_report = load_json(args.forward_gate_report)
    week3_comparison = load_json(args.week3_comparison)
    week4_comparison = load_json(args.week4_comparison)
    week5_comparison = load_json(args.week5_comparison)
    week6_summary = load_json(args.week6_summary)
    week6_manifest = load_json(args.week6_manifest)
    week7_pack_summary = load_json(args.week7_pack_summary)
    week7_animation_summary = load_json(args.week7_animation_summary)
    week7_manifest = load_json(args.week7_manifest)
    week7_results_index = load_json(args.week7_results_index)
    week7_hardened_validation = load_json(args.week7_hardened_validation)
    waivers = maybe_load_waivers(args.waiver_json)

    criteria = build_top_level_criteria(
        forward_summary=forward_summary,
        forward_gate_report=forward_gate_report,
        week5_comparison=week5_comparison,
        week6_summary=week6_summary,
        week6_manifest=week6_manifest,
        week7_pack_summary=week7_pack_summary,
        week7_animation_summary=week7_animation_summary,
        week7_manifest=week7_manifest,
        week7_results_index=week7_results_index,
        waivers=waivers,
    )
    tables = build_tables(
        forward_summary=forward_summary,
        week3_comparison=week3_comparison,
        week4_comparison=week4_comparison,
        week5_comparison=week5_comparison,
        week7_animation_summary=week7_animation_summary,
    )
    captions = build_captions(
        week3_comparison=week3_comparison,
        week4_comparison=week4_comparison,
        week5_comparison=week5_comparison,
        week7_animation_summary=week7_animation_summary,
    )
    command_list = build_command_list()

    config_paths = [
        "configs/forward_sweep_baseline.json",
        "configs/forward_sweep_week3.json",
        "configs/week3_anisotropy_ab.json",
        "configs/week4_collision_ablation.json",
        "configs/week5_layered_ablation.json",
        "configs/validation_gates_default.json",
    ]
    methods_settings = methods_settings_freeze(config_paths)
    reference_config_hashes = {
        "forward_sweep_config_hash": str(forward_summary.get("sweep_config_hash", "")),
        "week3_sweep_config_hash": str(week6_manifest["figures"][0].get("source_sweep_config_hash", "")),
        "week4_sweep_config_hash": str(week6_manifest["figures"][1].get("source_sweep_config_hash", "")),
        "week5_sweep_config_hash": str(week6_manifest["figures"][2].get("source_sweep_config_hash", "")),
        "week7_sweep_config_hash": str(week7_animation_summary.get("sweep_config_hash", "")),
    }
    methods_settings["reference_config_hashes"] = reference_config_hashes

    write_json(args.output_methods_settings_json, methods_settings)
    write_json(args.output_tables_json, tables)
    write_json(
        args.output_captions_json,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "captions": captions,
        },
    )
    write_json(
        args.output_commands_json,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "commands": command_list,
        },
    )
    write_text(
        args.output_packet_doc,
        packet_markdown(
            criteria=criteria,
            tables=tables,
            commands=command_list,
            plan_path=args.plan_path,
        ),
    )
    write_text(args.output_captions_doc, captions_markdown(captions))

    bundle_paths = [
        args.plan_path,
        args.forward_summary,
        args.forward_manifest,
        args.forward_gate_report,
        args.week3_comparison,
        args.week4_comparison,
        args.week5_comparison,
        args.week6_summary,
        args.week6_manifest,
        args.week7_pack_summary,
        args.week7_animation_summary,
        args.week7_manifest,
        args.week7_results_index,
        args.week7_hardened_validation,
        "docs/results_index.md",
        "docs/assets/week3_anisotropy_delta.png",
        "docs/assets/week4_collision_ablation.png",
        "docs/assets/week5_layered_ablation.png",
        "docs/assets/week7_baseline_vs_improved.gif",
        "docs/assets/week7_baseline_vs_improved.mp4",
        args.output_packet_doc,
        args.output_captions_doc,
        args.output_methods_settings_json,
        args.output_tables_json,
        args.output_captions_json,
        args.output_commands_json,
    ]
    bundle_records = [file_record(path) for path in bundle_paths]
    bundle_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_artifacts": len(bundle_records),
        "n_missing_artifacts": sum(0 if r["exists"] else 1 for r in bundle_records),
        "all_artifacts_present": all(r["exists"] for r in bundle_records),
        "reference_config_hashes": reference_config_hashes,
        "artifacts": bundle_records,
    }
    write_json(args.output_bundle_json, bundle_payload)

    key_metrics = {
        "forward_stability_rate": float(forward_summary.get("stability_rate", 0.0)),
        "forward_gi_plausible_rate": float(forward_summary.get("gi_plausible_rate", 0.0)),
        "week5_delta_gi_layered_minus_baseline": float(
            week5_comparison.get("delta_gi_layered_minus_baseline", 0.0)
        ),
        "week5_delta_outside_skull_frac_layered_minus_baseline": float(
            week5_comparison.get("delta_outside_skull_frac_layered_minus_baseline", 0.0)
        ),
        "week5_delta_disp_p95_layered_minus_baseline": float(
            week5_comparison.get("delta_disp_p95_layered_minus_baseline", 0.0)
        ),
        "week7_delta_gi_improved_minus_baseline": float(
            week7_animation_summary.get("delta_gi_improved_minus_baseline", 0.0)
        ),
        "week7_delta_disp_p95_improved_minus_baseline": float(
            week7_animation_summary.get("delta_disp_p95_improved_minus_baseline", 0.0)
        ),
        "week7_delta_outside_skull_frac_improved_minus_baseline": float(
            week7_animation_summary.get("delta_outside_skull_frac_improved_minus_baseline", 0.0)
        ),
        "week7_runtime_ratio_week7_over_week6": float(
            week7_hardened_validation.get("runtime_budget_vs_week6", {}).get(
                "runtime_ratio_week7_over_week6", 0.0
            )
        ),
    }

    acceptance_top_level = all(bool(c["effective_pass"]) for c in criteria)
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "plan_path": args.plan_path,
            "forward_summary": args.forward_summary,
            "forward_manifest": args.forward_manifest,
            "forward_gate_report": args.forward_gate_report,
            "week3_comparison": args.week3_comparison,
            "week4_comparison": args.week4_comparison,
            "week5_comparison": args.week5_comparison,
            "week6_summary": args.week6_summary,
            "week6_manifest": args.week6_manifest,
            "week7_pack_summary": args.week7_pack_summary,
            "week7_animation_summary": args.week7_animation_summary,
            "week7_manifest": args.week7_manifest,
            "week7_results_index": args.week7_results_index,
            "week7_hardened_validation": args.week7_hardened_validation,
        },
        "output_paths": {
            "packet_doc": args.output_packet_doc,
            "captions_doc": args.output_captions_doc,
            "methods_settings_json": args.output_methods_settings_json,
            "tables_json": args.output_tables_json,
            "captions_json": args.output_captions_json,
            "commands_json": args.output_commands_json,
            "bundle_json": args.output_bundle_json,
        },
        "reference_config_hashes": reference_config_hashes,
        "key_metrics": key_metrics,
        "top_level_success_criteria": criteria,
        "waivers_used": waivers,
        "acceptance_top_level_success_criteria_met_or_waived": acceptance_top_level,
        "acceptance_methods_settings_frozen": all(c["exists"] for c in methods_settings["configs"]),
        "acceptance_final_tables_generated": True,
        "acceptance_figure_captions_generated": True,
        "acceptance_submission_repro_commands_ready": len(command_list) > 0,
        "acceptance_artifact_bundle_complete": bool(bundle_payload["all_artifacts_present"]),
    }
    summary_payload["acceptance_draft_packet_complete"] = bool(
        summary_payload["acceptance_top_level_success_criteria_met_or_waived"]
        and summary_payload["acceptance_methods_settings_frozen"]
        and summary_payload["acceptance_final_tables_generated"]
        and summary_payload["acceptance_figure_captions_generated"]
        and summary_payload["acceptance_submission_repro_commands_ready"]
        and summary_payload["acceptance_artifact_bundle_complete"]
    )
    summary_payload["passed"] = bool(summary_payload["acceptance_draft_packet_complete"])

    write_json(args.output_summary_json, summary_payload)
    print(f"Saved: {args.output_packet_doc}")
    print(f"Saved: {args.output_captions_doc}")
    print(f"Saved: {args.output_methods_settings_json}")
    print(f"Saved: {args.output_tables_json}")
    print(f"Saved: {args.output_captions_json}")
    print(f"Saved: {args.output_commands_json}")
    print(f"Saved: {args.output_bundle_json}")
    print(f"Saved: {args.output_summary_json}")


if __name__ == "__main__":
    main()
