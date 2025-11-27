"""
Regression model module for NBA endgame prediction.

This module contains functions to build and evaluate regression models
that predict final scores from 3rd quarter scores.
"""

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class EndgameModel:
    """
    A regression model for predicting final score margin from Q3 margin.
    
    The model predicts: final_margin = intercept + slope * q3_margin
    
    Attributes:
        model: The underlying LinearRegression model
        slope: The coefficient for q3_margin
        intercept: The y-intercept
        r2: R-squared score on training data
        rmse: Root mean squared error on training data
        n_samples: Number of samples used for training
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None
        self.r2: Optional[float] = None
        self.rmse: Optional[float] = None
        self.n_samples: int = 0
        self.is_fitted: bool = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EndgameModel':
        """
        Fit the regression model.
        
        Args:
            X: Q3 margins (n_samples, 1)
            y: Final margins (n_samples,)
        
        Returns:
            self
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.model.fit(X, y)
        self.slope = self.model.coef_[0]
        self.intercept = self.model.intercept_
        self.n_samples = len(y)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        self.r2 = r2_score(y, y_pred)
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict final margin from Q3 margin.
        
        Args:
            X: Q3 margins
        
        Returns:
            Predicted final margins
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        return self.model.predict(X)
    
    def get_summary(self) -> Dict:
        """
        Get a summary of model parameters and metrics.
        
        Returns:
            Dictionary with model information
        """
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'r2': self.r2,
            'rmse': self.rmse,
            'n_samples': self.n_samples,
            'is_fitted': self.is_fitted
        }
    
    def __repr__(self) -> str:
        if not self.is_fitted:
            return "EndgameModel(not fitted)"
        return (
            f"EndgameModel(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
            f"R²={self.r2:.4f}, RMSE={self.rmse:.2f}, n={self.n_samples})"
        )


def build_league_model(df: pd.DataFrame) -> EndgameModel:
    """
    Build a league-wide regression model.
    
    Args:
        df: DataFrame with 'q3_margin' and 'final_margin' columns
    
    Returns:
        Fitted EndgameModel
    """
    X = df['q3_margin'].values.reshape(-1, 1)
    y = df['final_margin'].values
    
    model = EndgameModel()
    model.fit(X, y)
    
    return model


def build_team_models(df: pd.DataFrame, min_games: int = 10) -> Dict[str, EndgameModel]:
    """
    Build team-specific regression models.
    
    Args:
        df: DataFrame with 'team_abbrev', 'q3_margin', and 'final_margin' columns
        min_games: Minimum number of games required to build a model
    
    Returns:
        Dictionary mapping team abbreviation to fitted EndgameModel
    """
    team_models = {}
    
    for team_abbrev in df['team_abbrev'].unique():
        team_df = df[df['team_abbrev'] == team_abbrev]
        
        if len(team_df) < min_games:
            print(f"Skipping {team_abbrev}: only {len(team_df)} games (min: {min_games})")
            continue
        
        X = team_df['q3_margin'].values.reshape(-1, 1)
        y = team_df['final_margin'].values
        
        model = EndgameModel()
        model.fit(X, y)
        team_models[team_abbrev] = model
    
    return team_models


def compare_team_to_league(
    team_model: EndgameModel,
    league_model: EndgameModel,
    team_abbrev: str
) -> Dict:
    """
    Compare a team's model to the league-wide model.
    
    Key metrics:
    - Slope difference: How much more/less a team's final margin changes per point of Q3 lead
    - Intercept difference: Baseline advantage/disadvantage in Q4
    - "Closer" vs "Extender": Teams that perform better/worse than expected in Q4
    
    Args:
        team_model: Team's fitted EndgameModel
        league_model: League-wide fitted EndgameModel
        team_abbrev: Team abbreviation
    
    Returns:
        Dictionary with comparison metrics
    """
    slope_diff = team_model.slope - league_model.slope
    intercept_diff = team_model.intercept - league_model.intercept
    
    # Determine team characteristic
    # Positive slope_diff means team extends leads / overcomes deficits better than average
    # Negative slope_diff means team loses leads / struggles to overcome deficits more than average
    
    if slope_diff > 0.05:
        closing_type = "Extender"  # Better at maintaining/extending margins
        closing_desc = "Extends leads and overcomes deficits better than average"
    elif slope_diff < -0.05:
        closing_type = "Closer"  # Worse at maintaining margins (opponents close gap)
        closing_desc = "Prone to losing leads and struggling to overcome deficits"
    else:
        closing_type = "Average"
        closing_desc = "Performs close to league average in the 4th quarter"
    
    return {
        'team_abbrev': team_abbrev,
        'team_slope': team_model.slope,
        'team_intercept': team_model.intercept,
        'team_r2': team_model.r2,
        'team_n_samples': team_model.n_samples,
        'league_slope': league_model.slope,
        'league_intercept': league_model.intercept,
        'slope_diff': slope_diff,
        'intercept_diff': intercept_diff,
        'closing_type': closing_type,
        'closing_description': closing_desc
    }


def analyze_all_teams(
    df: pd.DataFrame,
    min_games: int = 10
) -> Tuple[EndgameModel, Dict[str, EndgameModel], pd.DataFrame]:
    """
    Build and compare all models.
    
    Args:
        df: Prepared model DataFrame
        min_games: Minimum games for team model
    
    Returns:
        Tuple of (league_model, team_models_dict, comparison_df)
    """
    print("Building league-wide model...")
    league_model = build_league_model(df)
    print(f"League model: {league_model}")
    
    print("\nBuilding team-specific models...")
    team_models = build_team_models(df, min_games)
    print(f"Built models for {len(team_models)} teams")
    
    print("\nComparing teams to league average...")
    comparisons = []
    for team_abbrev, team_model in team_models.items():
        comparison = compare_team_to_league(team_model, league_model, team_abbrev)
        comparisons.append(comparison)
    
    comparison_df = pd.DataFrame(comparisons)
    
    # Sort by slope difference (best closers first)
    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values('slope_diff', ascending=False)
    
    return league_model, team_models, comparison_df


def predict_final_from_q3(
    team_q3: int,
    opponent_q3: int,
    team_model: Optional[EndgameModel] = None,
    league_model: Optional[EndgameModel] = None
) -> Dict:
    """
    Predict final scores given Q3 scores.
    
    Args:
        team_q3: Team's score at end of Q3
        opponent_q3: Opponent's score at end of Q3
        team_model: Team-specific model (optional)
        league_model: League-wide model (required if team_model not provided)
    
    Returns:
        Dictionary with predictions
    """
    q3_margin = team_q3 - opponent_q3
    
    result = {
        'team_q3': team_q3,
        'opponent_q3': opponent_q3,
        'q3_margin': q3_margin
    }
    
    if league_model:
        league_pred_margin = league_model.predict(np.array([[q3_margin]]))[0]
        result['league_predicted_margin'] = league_pred_margin
        result['league_predicted_team_final'] = team_q3 + (league_pred_margin - q3_margin) / 2 + (q3_margin / 2)
        result['league_win_probability'] = "favorable" if league_pred_margin > 0 else "unfavorable"
    
    if team_model:
        team_pred_margin = team_model.predict(np.array([[q3_margin]]))[0]
        result['team_predicted_margin'] = team_pred_margin
        result['team_win_probability'] = "favorable" if team_pred_margin > 0 else "unfavorable"
        
        if league_model:
            result['margin_vs_league'] = team_pred_margin - league_pred_margin
    
    return result


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Generate synthetic data for testing
    n_samples = 100
    q3_margins = np.random.normal(0, 10, n_samples)
    # Final margin is roughly q3_margin * 0.85 + noise (regression to mean)
    final_margins = q3_margins * 0.85 + np.random.normal(0, 8, n_samples)
    
    test_df = pd.DataFrame({
        'team_abbrev': ['TEST'] * n_samples,
        'q3_margin': q3_margins,
        'final_margin': final_margins
    })
    
    # Build and test model
    model = build_league_model(test_df)
    print(f"Test model: {model}")
    print(f"Summary: {model.get_summary()}")
