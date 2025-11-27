#!/usr/bin/env python3
"""
NBA Endgames Analysis - Main Script

This script fetches NBA game data for the 2024-25 season, builds regression models
to predict final scores from Q3 scores, and analyzes team performance.

Usage:
    python main.py [--max-games N] [--output-dir DIR] [--save-plots]
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

from src.data_fetcher import fetch_and_prepare_data
from src.models import analyze_all_teams, predict_final_from_q3
from src.analysis import (
    generate_analysis_report,
    plot_league_model,
    plot_team_comparison,
    plot_team_vs_league,
    create_summary_table
)


def main():
    """Main function to run the NBA endgame analysis."""
    parser = argparse.ArgumentParser(description='NBA Endgame Analysis')
    parser.add_argument(
        '--max-games', 
        type=int, 
        default=None,
        help='Maximum number of games to process (for testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to output directory'
    )
    parser.add_argument(
        '--min-games',
        type=int,
        default=10,
        help='Minimum games required for team model'
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2024-25',
        help='NBA season to analyze (format: YYYY-YY)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("NBA ENDGAME ANALYSIS")
    print(f"Season: {args.season}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Step 1: Fetch and prepare data
    print("STEP 1: Fetching and preparing data...")
    print("-" * 40)
    try:
        model_df = fetch_and_prepare_data(
            season=args.season,
            max_games=args.max_games
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Please ensure you have internet access and the nba_api package is installed.")
        sys.exit(1)
    
    if len(model_df) == 0:
        print("No valid data retrieved. Exiting.")
        sys.exit(1)
    
    print(f"\nData summary:")
    print(f"  - Total observations: {len(model_df)}")
    print(f"  - Unique teams: {model_df['team_abbrev'].nunique()}")
    print(f"  - Date range: {model_df['game_date'].min()} to {model_df['game_date'].max()}")
    print()
    
    # Step 2: Build models
    print("STEP 2: Building regression models...")
    print("-" * 40)
    league_model, team_models, comparison_df = analyze_all_teams(
        model_df,
        min_games=args.min_games
    )
    print()
    
    # Step 3: Generate report
    print("STEP 3: Generating analysis report...")
    print("-" * 40)
    report = generate_analysis_report(comparison_df, league_model)
    print(report)
    
    # Save report to file
    if args.save_plots:
        report_path = os.path.join(args.output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    # Step 4: Create visualizations
    if args.save_plots:
        print("\nSTEP 4: Creating visualizations...")
        print("-" * 40)
        
        # League model plot
        league_plot_path = os.path.join(args.output_dir, 'league_model.png')
        plot_league_model(model_df, league_model, save_path=league_plot_path)
        
        # Team comparison plot
        if len(comparison_df) > 0:
            comparison_plot_path = os.path.join(args.output_dir, 'team_comparison.png')
            plot_team_comparison(comparison_df, league_model, save_path=comparison_plot_path)
            
            # Individual team plots for top/bottom performers
            sorted_teams = comparison_df.sort_values('slope_diff', ascending=False)
            
            # Top 3 performers
            for team_abbrev in sorted_teams.head(3)['team_abbrev']:
                if team_abbrev in team_models:
                    team_plot_path = os.path.join(args.output_dir, f'team_{team_abbrev}.png')
                    plot_team_vs_league(
                        team_abbrev, model_df, team_models[team_abbrev], 
                        league_model, save_path=team_plot_path
                    )
            
            # Bottom 3 performers
            for team_abbrev in sorted_teams.tail(3)['team_abbrev']:
                if team_abbrev in team_models:
                    team_plot_path = os.path.join(args.output_dir, f'team_{team_abbrev}.png')
                    plot_team_vs_league(
                        team_abbrev, model_df, team_models[team_abbrev],
                        league_model, save_path=team_plot_path
                    )
    
    # Step 5: Save summary table
    if args.save_plots and len(comparison_df) > 0:
        print("\nSTEP 5: Saving summary data...")
        print("-" * 40)
        
        summary_table = create_summary_table(comparison_df)
        summary_path = os.path.join(args.output_dir, 'team_summary.csv')
        summary_table.to_csv(summary_path, index=False)
        print(f"Summary table saved to: {summary_path}")
        
        # Also save raw data
        data_path = os.path.join(args.output_dir, 'model_data.csv')
        model_df.to_csv(data_path, index=False)
        print(f"Model data saved to: {data_path}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    # Example predictions
    print("\nEXAMPLE PREDICTIONS:")
    print("-" * 40)
    example_scenarios = [
        (85, 75),  # Leading by 10
        (75, 85),  # Trailing by 10
        (80, 80),  # Tied
        (90, 70),  # Leading by 20
    ]
    
    for team_q3, opp_q3 in example_scenarios:
        prediction = predict_final_from_q3(team_q3, opp_q3, league_model=league_model)
        print(f"Q3 Score: {team_q3}-{opp_q3} (margin: {team_q3-opp_q3:+d})")
        print(f"  Predicted final margin: {prediction.get('league_predicted_margin', 'N/A'):.1f}")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
