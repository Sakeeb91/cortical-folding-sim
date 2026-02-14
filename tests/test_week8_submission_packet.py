"""Tests for Week 8 submission packet generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_build_week8_submission_packet_generates_complete_summary(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts/build_week8_submission_packet.py"

    (tmp_path / "PLAN.md").write_text("# plan\n")

    _write_json(
        tmp_path / "results/forward_sweep_summary.json",
        {"stability_rate": 1.0, "gi_plausible_rate": 0.7, "sweep_config_hash": "forward_hash"},
    )
    _write_json(tmp_path / "results/forward_sweep_manifest.json", {"ok": True})
    _write_json(tmp_path / "results/validation_gate_report.json", {"passed": True})
    _write_json(tmp_path / "results/week3_anisotropy_comparison.json", {"delta_gi": 0.5})
    _write_json(
        tmp_path / "results/week4_collision_comparison.json",
        {"reduction_overlap_count_vs_sampled": 8.0, "runtime_ratio_spatial_over_sampled": 1.2},
    )
    _write_json(
        tmp_path / "results/week5_layered_comparison.json",
        {
            "delta_gi_layered_minus_baseline": 0.2,
            "delta_outside_skull_frac_layered_minus_baseline": -0.01,
            "delta_disp_p95_layered_minus_baseline": -0.02,
            "runtime_ratio_layered_over_baseline": 1.1,
        },
    )
    _write_json(
        tmp_path / "results/week6_figure_pipeline_summary.json",
        {"acceptance_one_command_regeneration_succeeds": True},
    )
    _write_json(
        tmp_path / "docs/assets/week6_figure_manifest.json",
        {
            "all_figures_have_source_run_ids": True,
            "figures": [
                {"source_sweep_config_hash": "week3_hash"},
                {"source_sweep_config_hash": "week4_hash"},
                {"source_sweep_config_hash": "week5_hash"},
            ],
        },
    )
    _write_json(
        tmp_path / "results/week7_animation_pack_summary.json",
        {"acceptance_animation_regeneration_succeeds": True},
    )
    _write_json(
        tmp_path / "results/week7_animation_comparison_summary.json",
        {
            "sweep_config_hash": "week7_hash",
            "delta_gi_improved_minus_baseline": 0.1,
            "delta_disp_p95_improved_minus_baseline": -0.01,
            "delta_outside_skull_frac_improved_minus_baseline": -0.02,
        },
    )
    _write_json(
        tmp_path / "docs/assets/week7_animation_manifest.json",
        {"all_assets_have_source_run_ids": True},
    )
    _write_json(
        tmp_path / "results/week7_results_index_summary.json",
        {"all_claims_have_linked_artifacts": True},
    )
    _write_json(
        tmp_path / "results/week7_hardened_validation.json",
        {"runtime_budget_vs_week6": {"runtime_ratio_week7_over_week6": 1.0}},
    )

    (tmp_path / "docs/results_index.md").write_text("ok\n")
    (tmp_path / "docs/assets/week3_anisotropy_delta.png").write_text("png")
    (tmp_path / "docs/assets/week4_collision_ablation.png").write_text("png")
    (tmp_path / "docs/assets/week5_layered_ablation.png").write_text("png")
    (tmp_path / "docs/assets/week7_baseline_vs_improved.gif").write_text("gif")
    (tmp_path / "docs/assets/week7_baseline_vs_improved.mp4").write_text("mp4")

    for config_name in (
        "forward_sweep_baseline.json",
        "forward_sweep_week3.json",
        "week3_anisotropy_ab.json",
        "week4_collision_ablation.json",
        "week5_layered_ablation.json",
        "validation_gates_default.json",
    ):
        (tmp_path / "configs" / config_name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "configs" / config_name).write_text("{}")

    summary_out = tmp_path / "results/week8_submission_packet_summary.json"
    cmd = [
        "python3.11",
        str(script_path),
        "--plan-path",
        "PLAN.md",
        "--forward-summary",
        "results/forward_sweep_summary.json",
        "--forward-manifest",
        "results/forward_sweep_manifest.json",
        "--forward-gate-report",
        "results/validation_gate_report.json",
        "--week3-comparison",
        "results/week3_anisotropy_comparison.json",
        "--week4-comparison",
        "results/week4_collision_comparison.json",
        "--week5-comparison",
        "results/week5_layered_comparison.json",
        "--week6-summary",
        "results/week6_figure_pipeline_summary.json",
        "--week6-manifest",
        "docs/assets/week6_figure_manifest.json",
        "--week7-pack-summary",
        "results/week7_animation_pack_summary.json",
        "--week7-animation-summary",
        "results/week7_animation_comparison_summary.json",
        "--week7-manifest",
        "docs/assets/week7_animation_manifest.json",
        "--week7-results-index",
        "results/week7_results_index_summary.json",
        "--week7-hardened-validation",
        "results/week7_hardened_validation.json",
        "--output-packet-doc",
        "docs/week8_methods_results_packet.md",
        "--output-captions-doc",
        "docs/assets/week8_figure_captions.md",
        "--output-methods-settings-json",
        "results/week8_methods_settings_freeze.json",
        "--output-tables-json",
        "results/week8_final_tables.json",
        "--output-captions-json",
        "results/week8_figure_captions.json",
        "--output-commands-json",
        "results/week8_reproducibility_commands.json",
        "--output-bundle-json",
        "results/week8_frozen_artifact_bundle.json",
        "--output-summary-json",
        "results/week8_submission_packet_summary.json",
    ]
    subprocess.check_call(cmd, cwd=tmp_path)

    with summary_out.open() as f:
        payload = json.load(f)

    assert payload["acceptance_top_level_success_criteria_met_or_waived"] is True
    assert payload["acceptance_draft_packet_complete"] is True
    assert payload["passed"] is True

