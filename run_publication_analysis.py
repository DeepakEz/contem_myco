#!/usr/bin/env python3
"""
Run comprehensive publication analysis on MycoNet++ Contemplative AI results
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path.cwd()))

from publication_analysis_suite import analyze_publication_results

def main():
    print("🧠 MycoNet++ Contemplative AI - Publication Analysis")
    print("=" * 60)
    
    # Define paths
    results_dir = "publication_results"
    output_dir = "comprehensive_publication_analysis"
    
    # Check if results directory exists
    if not Path(results_dir).exists():
        print(f"❌ Results directory '{results_dir}' not found!")
        print("Please ensure your experimental results are in this directory.")
        return
    
    print(f"📂 Analyzing results from: {results_dir}")
    print(f"📊 Output will be saved to: {output_dir}")
    print()
    
    try:
        # Run comprehensive analysis
        results = analyze_publication_results(
            results_directory=results_dir,
            output_directory=output_dir
        )
        
        print("\n🎉 SUCCESS! Publication analysis completed!")
        print("=" * 60)
        print(f"📊 Statistical Tests: {len(results.statistical_tests)} configurations analyzed")
        print(f"📈 Effect Sizes: {len(results.effect_sizes)} comparisons calculated")
        print(f"🖼️  Visualizations: {len(results.visualizations)} publication-quality plots")
        print(f"🔍 XAI Insights: {len(results.xai_summaries.get('key_insights', []))} behavioral patterns")
        print()
        print("📁 Generated Files:")
        print(f"   📄 Main Report: {output_dir}/COMPREHENSIVE_PUBLICATION_REPORT.md")
        print(f"   📓 Jupyter Notebook: {output_dir}/interactive_analysis.ipynb")
        print(f"   📊 Visualizations: {output_dir}/visualizations/")
        print(f"   🔍 XAI Analysis: {output_dir}/xai_logs/")
        print()
        print("🎓 Ready for academic publication submission!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Please check your results format and try again.")
        return
    
    print("\n🚀 Next Steps:")
    print("1. Review the comprehensive report")
    print("2. Examine publication-quality visualizations")
    print("3. Use generated figures in your paper")
    print("4. Submit to target venues (AAMAS, IJCAI, Nature MI)")

if __name__ == "__main__":
    main()
