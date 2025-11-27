"""
Analysis and visualization module for NBA endgame models.

This module provides functions to analyze team performance relative to
league average and visualize the results.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import EndgameModel


def generate_analysis_report(
    comparison_df: pd.DataFrame,
    league_model: EndgameModel
) -> str:
    """
    Generate a text report analyzing team performance.
    
    Args:
        comparison_df: DataFrame from analyze_all_teams()
        league_model: League-wide EndgameModel
    
    Returns:
        Formatted string report
    """
    report_lines = [
        "=" * 70,
        "NBA ENDGAME ANALYSIS REPORT",
        "=" * 70,
        "",
        "LEAGUE-WIDE MODEL",
        "-" * 40,
        f"Equation: Final Margin = {league_model.intercept:.3f} + {league_model.slope:.3f} × Q3 Margin",
        f"R-squared: {league_model.r2:.4f}",
        f"RMSE: {league_model.rmse:.2f} points",
        f"Sample size: {league_model.n_samples} games",
        "",
        "INTERPRETATION:",
        f"  - A team leading by 10 at end of Q3 is expected to win by {league_model.intercept + league_model.slope * 10:.1f} points",
        f"  - A team trailing by 10 at end of Q3 is expected to lose by {abs(league_model.intercept + league_model.slope * -10):.1f} points",
        f"  - Each point of Q3 margin translates to {league_model.slope:.2f} points of final margin",
        "",
    ]
    
    if len(comparison_df) > 0:
        report_lines.extend([
            "TEAM RANKINGS BY Q4 PERFORMANCE",
            "-" * 40,
            "",
            "Teams that OUTPERFORM expectations (extend leads / overcome deficits):",
            ""
        ])
        
        extenders = comparison_df[comparison_df['slope_diff'] > 0.02]
        for _, row in extenders.iterrows():
            report_lines.append(
                f"  {row['team_abbrev']:>4}: slope diff = {row['slope_diff']:+.4f} ({row['closing_type']})"
            )
        
        report_lines.extend([
            "",
            "Teams that UNDERPERFORM expectations (lose leads / can't overcome deficits):",
            ""
        ])
        
        closers = comparison_df[comparison_df['slope_diff'] < -0.02]
        for _, row in closers.iterrows():
            report_lines.append(
                f"  {row['team_abbrev']:>4}: slope diff = {row['slope_diff']:+.4f} ({row['closing_type']})"
            )
        
        report_lines.extend([
            "",
            "Teams performing NEAR AVERAGE:",
            ""
        ])
        
        average_teams = comparison_df[
            (comparison_df['slope_diff'] >= -0.02) & 
            (comparison_df['slope_diff'] <= 0.02)
        ]
        for _, row in average_teams.iterrows():
            report_lines.append(
                f"  {row['team_abbrev']:>4}: slope diff = {row['slope_diff']:+.4f}"
            )
    
    report_lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70
    ])
    
    return "\n".join(report_lines)


def plot_league_model(
    df: pd.DataFrame,
    league_model: EndgameModel,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot with league regression line.
    
    Args:
        df: Model DataFrame with q3_margin and final_margin
        league_model: Fitted league model
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot of actual data
    ax.scatter(
        df['q3_margin'], 
        df['final_margin'], 
        alpha=0.3, 
        s=20,
        label='Actual games'
    )
    
    # Regression line
    x_range = np.linspace(df['q3_margin'].min(), df['q3_margin'].max(), 100)
    y_pred = league_model.predict(x_range)
    ax.plot(x_range, y_pred, 'r-', linewidth=2, label='League average model')
    
    # Perfect prediction line (y = x)
    ax.plot(x_range, x_range, 'g--', linewidth=1, alpha=0.5, label='No change (y=x)')
    
    # Zero lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Q3 Margin (Team Score - Opponent Score)', fontsize=12)
    ax.set_ylabel('Final Margin (Team Score - Opponent Score)', fontsize=12)
    ax.set_title(
        f'NBA Endgame Analysis: Q3 Margin vs Final Margin\n'
        f'League Model: Final = {league_model.intercept:.2f} + {league_model.slope:.2f} × Q3 Margin '
        f'(R² = {league_model.r2:.3f})',
        fontsize=12
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_team_comparison(
    comparison_df: pd.DataFrame,
    league_model: EndgameModel,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart comparing team slopes to league average.
    
    Args:
        comparison_df: DataFrame from analyze_all_teams()
        league_model: Fitted league model
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    if len(comparison_df) == 0:
        print("No team data to plot")
        return None
    
    # Sort by slope difference
    df_sorted = comparison_df.sort_values('slope_diff', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.4)))
    
    colors = ['green' if x > 0 else 'red' for x in df_sorted['slope_diff']]
    
    bars = ax.barh(
        df_sorted['team_abbrev'],
        df_sorted['slope_diff'],
        color=colors,
        alpha=0.7
    )
    
    # Add league baseline
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='League Average')
    
    ax.set_xlabel('Slope Difference vs League Average', fontsize=12)
    ax.set_ylabel('Team', fontsize=12)
    ax.set_title(
        'Team Q4 Performance vs League Average\n'
        '(Positive = Extends leads better, Negative = Loses leads more)',
        fontsize=12
    )
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_team_vs_league(
    team_abbrev: str,
    df: pd.DataFrame,
    team_model: EndgameModel,
    league_model: EndgameModel,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comparison plot for a specific team vs league average.
    
    Args:
        team_abbrev: Team abbreviation
        df: Full model DataFrame
        team_model: Team's fitted model
        league_model: League-wide model
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    team_df = df[df['team_abbrev'] == team_abbrev]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Team's actual games
    ax.scatter(
        team_df['q3_margin'],
        team_df['final_margin'],
        alpha=0.6,
        s=50,
        color='blue',
        label=f'{team_abbrev} actual games'
    )
    
    # X range for lines
    x_min = min(df['q3_margin'].min(), team_df['q3_margin'].min())
    x_max = max(df['q3_margin'].max(), team_df['q3_margin'].max())
    x_range = np.linspace(x_min, x_max, 100)
    
    # League regression line
    y_league = league_model.predict(x_range)
    ax.plot(x_range, y_league, 'r-', linewidth=2, label='League average')
    
    # Team regression line
    y_team = team_model.predict(x_range)
    ax.plot(x_range, y_team, 'b--', linewidth=2, label=f'{team_abbrev} model')
    
    # Zero lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Q3 Margin', fontsize=12)
    ax.set_ylabel('Final Margin', fontsize=12)
    
    slope_diff = team_model.slope - league_model.slope
    perf_type = "OUTPERFORMS" if slope_diff > 0 else "UNDERPERFORMS"
    ax.set_title(
        f'{team_abbrev} vs League Average\n'
        f'Team slope: {team_model.slope:.3f}, League slope: {league_model.slope:.3f}\n'
        f'{perf_type} league by {abs(slope_diff):.3f} slope points',
        fontsize=12
    )
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def create_summary_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted summary table for display.
    
    Args:
        comparison_df: DataFrame from analyze_all_teams()
    
    Returns:
        Formatted DataFrame
    """
    if len(comparison_df) == 0:
        return pd.DataFrame()
    
    summary = comparison_df[[
        'team_abbrev', 'team_slope', 'league_slope', 
        'slope_diff', 'team_r2', 'team_n_samples', 'closing_type'
    ]].copy()
    
    summary.columns = [
        'Team', 'Team Slope', 'League Slope',
        'Slope Diff', 'R²', 'Games', 'Performance Type'
    ]
    
    # Round numeric columns
    summary['Team Slope'] = summary['Team Slope'].round(4)
    summary['League Slope'] = summary['League Slope'].round(4)
    summary['Slope Diff'] = summary['Slope Diff'].round(4)
    summary['R²'] = summary['R²'].round(4)
    
    return summary


if __name__ == "__main__":
    # Test with synthetic data
    import numpy as np
    from .models import EndgameModel
    
    # Create a mock league model
    league = EndgameModel()
    league.slope = 0.85
    league.intercept = 0.5
    league.r2 = 0.72
    league.rmse = 8.5
    league.n_samples = 500
    league.is_fitted = True
    
    # Create mock comparison data
    mock_comparisons = pd.DataFrame({
        'team_abbrev': ['LAL', 'BOS', 'MIA', 'CHI', 'NYK'],
        'team_slope': [0.90, 0.82, 0.85, 0.78, 0.88],
        'team_intercept': [0.6, 0.4, 0.5, 0.3, 0.7],
        'team_r2': [0.75, 0.70, 0.72, 0.68, 0.74],
        'team_n_samples': [50, 48, 52, 45, 49],
        'league_slope': [0.85] * 5,
        'league_intercept': [0.5] * 5,
        'slope_diff': [0.05, -0.03, 0.0, -0.07, 0.03],
        'intercept_diff': [0.1, -0.1, 0.0, -0.2, 0.2],
        'closing_type': ['Extender', 'Average', 'Average', 'Closer', 'Average'],
        'closing_description': [''] * 5
    })
    
    # Generate report
    report = generate_analysis_report(mock_comparisons, league)
    print(report)
