#!/usr/bin/env python3
"""
Run All Experiments - Main Entry Point
=======================================
Execute complete experimental pipeline:
1. Run resilience experiments (reactive vs contemplative)
2. Run society experiments (baseline vs myco)
3. Generate comparison plots
4. Generate summary briefs

Usage:
    python run_all_experiments.py [--quick]

Options:
    --quick: Run quick experiments (fewer steps/runs)
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Import experiment components
from experiment_runner import run_resilience_experiment, run_society_experiment
from visualization import ExperimentVisualizer, load_results
from brief_generator import BriefGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main(quick_mode: bool = False):
    """
    Run complete experimental pipeline

    Args:
        quick_mode: If True, run shorter experiments for testing
    """

    logger.info("=" * 60)
    logger.info("MycoAgent Resilience & Society Experiments")
    logger.info("=" * 60)

    # Configuration
    if quick_mode:
        logger.info("Running in QUICK mode (reduced steps/runs)")
        num_steps = 50
        num_runs = 2
        output_dir = "quick_results"
    else:
        logger.info("Running FULL experiments")
        num_steps = 200
        num_runs = 3
        output_dir = "results"

    seed = 42

    # Create output directories
    Path(output_dir).mkdir(exist_ok=True)
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    briefs_dir = Path(output_dir) / "briefs"
    briefs_dir.mkdir(exist_ok=True)

    # ===== PHASE 1: RESILIENCE EXPERIMENTS =====

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Resilience Experiments (Flood Disaster)")
    logger.info("=" * 60)

    logger.info("\n--- Running Reactive Agents ---")
    resilience_reactive_results = run_resilience_experiment(
        agent_type='reactive',
        num_steps=num_steps,
        num_runs=num_runs,
        seed=seed,
        output_dir=output_dir
    )

    logger.info("\n--- Running Contemplative Agents ---")
    resilience_contemplative_results = run_resilience_experiment(
        agent_type='contemplative',
        num_steps=num_steps,
        num_runs=num_runs,
        seed=seed,
        output_dir=output_dir
    )

    logger.info("\n✓ Resilience experiments completed")

    # ===== PHASE 2: SOCIETY EXPERIMENTS =====

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Society Experiments (Policy-Making)")
    logger.info("=" * 60)

    logger.info("\n--- Running Baseline Policy Agent ---")
    society_baseline_results = run_society_experiment(
        agent_type='baseline',
        num_steps=num_steps,
        num_runs=num_runs,
        seed=seed,
        output_dir=output_dir
    )

    logger.info("\n--- Running Myco Policy Agent ---")
    society_myco_results = run_society_experiment(
        agent_type='myco',
        num_steps=num_steps,
        num_runs=num_runs,
        seed=seed,
        output_dir=output_dir
    )

    logger.info("\n✓ Society experiments completed")

    # ===== PHASE 3: GENERATE VISUALIZATIONS =====

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Generating Visualizations")
    logger.info("=" * 60)

    visualizer = ExperimentVisualizer(output_dir=str(plots_dir))

    # Resilience plots
    logger.info("\n--- Generating Resilience Comparison Plots ---")
    visualizer.plot_resilience_comparison(
        resilience_reactive_results,
        resilience_contemplative_results,
        save_path=plots_dir / "resilience_comparison.png"
    )

    visualizer.plot_compute_overhead(
        resilience_reactive_results,
        resilience_contemplative_results,
        save_path=plots_dir / "resilience_compute_overhead.png"
    )

    # Society plots
    logger.info("\n--- Generating Society Comparison Plots ---")
    visualizer.plot_society_comparison(
        society_baseline_results,
        society_myco_results,
        save_path=plots_dir / "society_comparison.png"
    )

    logger.info("\n✓ Visualizations completed")

    # ===== PHASE 4: GENERATE BRIEFS =====

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Generating Summary Briefs")
    logger.info("=" * 60)

    brief_gen = BriefGenerator(output_dir=str(briefs_dir))

    # Resilience brief
    logger.info("\n--- Generating Resilience Brief ---")
    brief_gen.generate_resilience_brief(
        resilience_reactive_results,
        resilience_contemplative_results,
        save_path=briefs_dir / "resilience_brief.md"
    )

    # Society brief
    logger.info("\n--- Generating Society Brief ---")
    brief_gen.generate_society_brief(
        society_baseline_results,
        society_myco_results,
        save_path=briefs_dir / "society_brief.md"
    )

    # Combined brief
    logger.info("\n--- Generating Combined Brief ---")
    brief_gen.generate_combined_brief(
        resilience_reactive_results,
        resilience_contemplative_results,
        society_baseline_results,
        society_myco_results,
        save_path=briefs_dir / "combined_brief.md"
    )

    logger.info("\n✓ Briefs completed")

    # ===== FINAL SUMMARY =====

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT PIPELINE COMPLETED")
    logger.info("=" * 60)

    logger.info(f"\nResults saved to: {output_dir}/")
    logger.info(f"  - Plots: {plots_dir}/")
    logger.info(f"  - Briefs: {briefs_dir}/")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\nResilience Experiment:")
    r_casualties_reactive = resilience_reactive_results['mean_metrics'].get('total_casualties', 0)
    r_casualties_contemp = resilience_contemplative_results['mean_metrics'].get('total_casualties', 0)
    casualty_reduction = ((r_casualties_reactive - r_casualties_contemp) / r_casualties_reactive * 100
                         if r_casualties_reactive > 0 else 0)

    print(f"  Reactive casualties: {r_casualties_reactive:.1f}")
    print(f"  Contemplative casualties: {r_casualties_contemp:.1f}")
    print(f"  → Reduction: {casualty_reduction:.1f}%")

    print("\nSociety Experiment:")
    s_inequality_baseline = society_baseline_results['mean_metrics'].get('final_inequality', 0)
    s_inequality_myco = society_myco_results['mean_metrics'].get('final_inequality', 0)
    inequality_reduction = ((s_inequality_baseline - s_inequality_myco) / s_inequality_baseline * 100
                           if s_inequality_baseline > 0 else 0)

    print(f"  Baseline inequality: {s_inequality_baseline:.3f}")
    print(f"  Myco inequality: {s_inequality_myco:.3f}")
    print(f"  → Reduction: {inequality_reduction:.1f}%")

    print("\n" + "=" * 60)
    print("For detailed analysis, see:")
    print(f"  {briefs_dir}/combined_brief.md")
    print("=" * 60 + "\n")

    return {
        'resilience': {
            'reactive': resilience_reactive_results,
            'contemplative': resilience_contemplative_results
        },
        'society': {
            'baseline': society_baseline_results,
            'myco': society_myco_results
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MycoAgent Resilience & Society experiments"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick experiments (fewer steps/runs for testing)'
    )

    args = parser.parse_args()

    try:
        results = main(quick_mode=args.quick)
        logger.info("All experiments completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Experiment pipeline failed: {e}", exc_info=True)
        sys.exit(1)
