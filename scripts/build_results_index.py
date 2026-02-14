"""Build a Week 7 results index that maps claims to evidence artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-doc",
        default="docs/results_index.md",
        help="Markdown results index output path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week7_results_index_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--week3-comparison",
        default="results/week3_anisotropy_comparison.json",
        help="Week 3 anisotropy comparison artifact.",
    )
    parser.add_argument(
        "--week4-comparison",
        default="results/week4_collision_comparison.json",
        help="Week 4 collision comparison artifact.",
    )
    parser.add_argument(
        "--week5-comparison",
        default="results/week5_layered_comparison.json",
        help="Week 5 layered comparison artifact.",
    )
    parser.add_argument(
        "--week6-summary",
        default="results/week6_figure_pipeline_summary.json",
        help="Week 6 figure pipeline summary artifact.",
    )
    parser.add_argument(
        "--week6-manifest",
        default="docs/assets/week6_figure_manifest.json",
        help="Week 6 figure manifest artifact.",
    )
    parser.add_argument(
        "--week7-summary",
        default="results/week7_animation_comparison_summary.json",
        help="Week 7 animation comparison summary artifact.",
    )
    parser.add_argument(
        "--week7-manifest",
        default="docs/assets/week7_animation_manifest.json",
        help="Week 7 animation manifest artifact.",
    )
    return parser.parse_args()


def claim_artifacts(args: argparse.Namespace) -> list[dict]:
    return [
        {
            "claim_id": "C1",
            "claim": "Directional anisotropy changes morphology versus isotropic growth.",
            "artifacts": [
                args.week3_comparison,
                "docs/assets/week3_anisotropy_delta.png",
                "results/week3_anisotropy_ab_summary.json",
            ],
        },
        {
            "claim_id": "C2",
            "claim": "Spatial-hash collision handling reduces penetration outliers versus sampled checks.",
            "artifacts": [
                args.week4_comparison,
                "docs/assets/week4_collision_ablation.png",
                "results/week4_collision_ablation_summary.json",
            ],
        },
        {
            "claim_id": "C3",
            "claim": "Layered approximation remains stable and improves GI on the reference setting.",
            "artifacts": [
                args.week5_comparison,
                "docs/assets/week5_layered_ablation.png",
                "results/week5_layered_ablation_summary.json",
            ],
        },
        {
            "claim_id": "C4",
            "claim": "Figure outputs regenerate deterministically with mapped source run IDs.",
            "artifacts": [
                args.week6_summary,
                args.week6_manifest,
                "docs/week6_figure_pipeline.md",
            ],
        },
        {
            "claim_id": "C5",
            "claim": "Week 7 baseline-vs-improved comparison animations are reproducible in GIF and MP4 formats.",
            "artifacts": [
                args.week7_summary,
                args.week7_manifest,
                "docs/assets/week7_baseline_vs_improved.gif",
                "docs/assets/week7_baseline_vs_improved.mp4",
            ],
        },
    ]


def main() -> None:
    args = parse_args()
    claims = claim_artifacts(args)

    rows: list[dict] = []
    for claim in claims:
        present = [p for p in claim["artifacts"] if Path(p).exists()]
        missing = [p for p in claim["artifacts"] if not Path(p).exists()]
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "artifacts": claim["artifacts"],
                "present_artifacts": present,
                "missing_artifacts": missing,
                "has_linked_artifact": len(present) > 0,
            }
        )

    all_claims_linked = all(row["has_linked_artifact"] for row in rows)

    output_doc = Path(args.output_doc)
    output_doc.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Results Index",
        "",
        "This index maps planned paper claims to concrete reproducible artifacts.",
        "",
        "| Claim ID | Claim | Linked artifacts | Missing artifacts |",
        "|---|---|---|---|",
    ]

    for row in rows:
        linked = "<br>".join(f"`{p}`" for p in row["present_artifacts"]) or "-"
        missing = "<br>".join(f"`{p}`" for p in row["missing_artifacts"]) or "-"
        lines.append(f"| {row['claim_id']} | {row['claim']} | {linked} | {missing} |")

    lines.extend(
        [
            "",
            f"Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
            f"Acceptance (all claims have >=1 linked artifact): `{str(all_claims_linked).lower()}`",
        ]
    )

    output_doc.write_text("\n".join(lines) + "\n")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_doc": str(output_doc),
        "n_claims": len(rows),
        "n_claims_with_linked_artifacts": sum(int(r["has_linked_artifact"]) for r in rows),
        "claims_missing_all_artifacts": [
            r["claim_id"] for r in rows if not r["has_linked_artifact"]
        ],
        "all_claims_have_linked_artifacts": all_claims_linked,
        "claims": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {output_doc}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
