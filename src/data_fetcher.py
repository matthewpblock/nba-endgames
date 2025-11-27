"""
Data fetching module for NBA game data.

This module uses the nba_api package to fetch game data from the 2024-2025 NBA season,
including 3rd quarter scores and final scores for all teams.
"""

import time
from typing import Optional

import pandas as pd

try:
    from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("Warning: nba_api package not installed. Install with: pip install nba_api")


def get_all_teams() -> pd.DataFrame:
    """
    Get all NBA teams.
    
    Returns:
        DataFrame with team information including id, full_name, abbreviation, etc.
    
    Raises:
        ImportError: If nba_api is not installed
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api package is required. Install with: pip install nba_api")
    nba_teams = teams.get_teams()
    return pd.DataFrame(nba_teams)


def get_season_games(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch all games for a given NBA season.
    
    Args:
        season: The NBA season in format "YYYY-YY" (e.g., "2024-25")
    
    Returns:
        DataFrame with game information
    
    Raises:
        ImportError: If nba_api is not installed
        ConnectionError: If unable to connect to NBA API
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api package is required. Install with: pip install nba_api")
    
    try:
        game_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",  # NBA league ID
            season_type_nullable="Regular Season"
        )
        games_df = game_finder.get_data_frames()[0]
        return games_df
    except Exception as e:
        raise ConnectionError(f"Unable to fetch data from NBA API: {e}")


def get_game_box_score(game_id: str, delay: float = 0.6) -> Optional[pd.DataFrame]:
    """
    Fetch the traditional box score for a specific game.
    
    Args:
        game_id: The NBA game ID
        delay: Delay in seconds before making API call (to avoid rate limiting)
    
    Returns:
        DataFrame with box score information or None if failed
    """
    time.sleep(delay)  # Rate limiting
    try:
        box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        return box_score.get_data_frames()
    except Exception as e:
        print(f"Error fetching box score for game {game_id}: {e}")
        return None


def get_quarter_scores_from_games(games_df: pd.DataFrame, max_games: Optional[int] = None) -> pd.DataFrame:
    """
    Extract quarter-by-quarter scores from games.
    
    This function parses game data to get Q1, Q2, Q3 cumulative scores and final scores
    for both home and away teams.
    
    Args:
        games_df: DataFrame from get_season_games()
        max_games: Optional limit on number of games to process (for testing)
    
    Returns:
        DataFrame with columns:
        - game_id, game_date
        - home_team_id, home_team_abbrev
        - away_team_id, away_team_abbrev
        - home_q3_score, away_q3_score (cumulative through Q3)
        - home_final_score, away_final_score
    """
    # Get unique game IDs
    unique_game_ids = games_df['GAME_ID'].unique()
    
    if max_games:
        unique_game_ids = unique_game_ids[:max_games]
    
    game_records = []
    
    for i, game_id in enumerate(unique_game_ids):
        if i % 50 == 0:
            print(f"Processing game {i + 1}/{len(unique_game_ids)}")
        
        # Get games for this game_id (should have 2 rows - one per team)
        game_data = games_df[games_df['GAME_ID'] == game_id]
        
        if len(game_data) != 2:
            continue
        
        # Identify home and away teams based on MATCHUP format
        # Home team matchup contains "vs.", away team contains "@"
        home_row = game_data[game_data['MATCHUP'].str.contains(' vs. ', na=False)]
        away_row = game_data[game_data['MATCHUP'].str.contains(' @ ', na=False)]
        
        if len(home_row) != 1 or len(away_row) != 1:
            continue
        
        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]
        
        # Extract Q3 cumulative scores (Q1 + Q2 + Q3)
        # The game data may have PTS_QTR1, PTS_QTR2, PTS_QTR3, PTS_QTR4 columns
        # or we need to calculate from the box score
        
        # Check if quarter columns exist
        if all(col in games_df.columns for col in ['PTS_QTR1', 'PTS_QTR2', 'PTS_QTR3']):
            home_q3_score = (
                home_row.get('PTS_QTR1', 0) + 
                home_row.get('PTS_QTR2', 0) + 
                home_row.get('PTS_QTR3', 0)
            )
            away_q3_score = (
                away_row.get('PTS_QTR1', 0) + 
                away_row.get('PTS_QTR2', 0) + 
                away_row.get('PTS_QTR3', 0)
            )
        else:
            # If quarter columns don't exist, we'll need to fetch from box score
            # For now, skip and handle in a separate pass
            home_q3_score = None
            away_q3_score = None
        
        record = {
            'game_id': game_id,
            'game_date': home_row['GAME_DATE'],
            'home_team_id': home_row['TEAM_ID'],
            'home_team_abbrev': home_row['TEAM_ABBREVIATION'],
            'away_team_id': away_row['TEAM_ID'],
            'away_team_abbrev': away_row['TEAM_ABBREVIATION'],
            'home_q3_score': home_q3_score,
            'away_q3_score': away_q3_score,
            'home_final_score': home_row['PTS'],
            'away_final_score': away_row['PTS']
        }
        game_records.append(record)
    
    return pd.DataFrame(game_records)


def prepare_model_data(quarter_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for regression modeling.
    
    Transforms game data into rows suitable for regression:
    - Each game produces 2 rows (one for each team's perspective)
    - Features: team_q3_score, opponent_q3_score, q3_margin
    - Target: team_final_score, final_margin
    
    Args:
        quarter_scores_df: DataFrame from get_quarter_scores_from_games()
    
    Returns:
        DataFrame with columns:
        - team_id, team_abbrev
        - opponent_id, opponent_abbrev  
        - is_home (boolean)
        - team_q3_score, opponent_q3_score, q3_margin
        - team_final_score, opponent_final_score, final_margin
        - q4_points_scored, q4_points_allowed, q4_margin
    """
    # Remove rows with missing Q3 scores
    df = quarter_scores_df.dropna(subset=['home_q3_score', 'away_q3_score']).copy()
    
    model_rows = []
    
    for _, row in df.iterrows():
        # Home team perspective
        home_record = {
            'game_id': row['game_id'],
            'game_date': row['game_date'],
            'team_id': row['home_team_id'],
            'team_abbrev': row['home_team_abbrev'],
            'opponent_id': row['away_team_id'],
            'opponent_abbrev': row['away_team_abbrev'],
            'is_home': True,
            'team_q3_score': row['home_q3_score'],
            'opponent_q3_score': row['away_q3_score'],
            'q3_margin': row['home_q3_score'] - row['away_q3_score'],
            'team_final_score': row['home_final_score'],
            'opponent_final_score': row['away_final_score'],
            'final_margin': row['home_final_score'] - row['away_final_score'],
        }
        home_record['q4_points_scored'] = home_record['team_final_score'] - home_record['team_q3_score']
        home_record['q4_points_allowed'] = home_record['opponent_final_score'] - home_record['opponent_q3_score']
        home_record['q4_margin'] = home_record['q4_points_scored'] - home_record['q4_points_allowed']
        
        # Away team perspective
        away_record = {
            'game_id': row['game_id'],
            'game_date': row['game_date'],
            'team_id': row['away_team_id'],
            'team_abbrev': row['away_team_abbrev'],
            'opponent_id': row['home_team_id'],
            'opponent_abbrev': row['home_team_abbrev'],
            'is_home': False,
            'team_q3_score': row['away_q3_score'],
            'opponent_q3_score': row['home_q3_score'],
            'q3_margin': row['away_q3_score'] - row['home_q3_score'],
            'team_final_score': row['away_final_score'],
            'opponent_final_score': row['home_final_score'],
            'final_margin': row['away_final_score'] - row['home_final_score'],
        }
        away_record['q4_points_scored'] = away_record['team_final_score'] - away_record['team_q3_score']
        away_record['q4_points_allowed'] = away_record['opponent_final_score'] - away_record['opponent_q3_score']
        away_record['q4_margin'] = away_record['q4_points_scored'] - away_record['q4_points_allowed']
        
        model_rows.append(home_record)
        model_rows.append(away_record)
    
    return pd.DataFrame(model_rows)


def fetch_and_prepare_data(season: str = "2024-25", max_games: Optional[int] = None) -> pd.DataFrame:
    """
    Main function to fetch and prepare all data for modeling.
    
    Args:
        season: The NBA season in format "YYYY-YY"
        max_games: Optional limit on games to process
    
    Returns:
        DataFrame ready for regression modeling
    """
    print(f"Fetching games for {season} season...")
    games_df = get_season_games(season)
    print(f"Found {len(games_df)} game records")
    
    print("Extracting quarter scores...")
    quarter_scores_df = get_quarter_scores_from_games(games_df, max_games)
    print(f"Processed {len(quarter_scores_df)} unique games")
    
    print("Preparing model data...")
    model_df = prepare_model_data(quarter_scores_df)
    print(f"Created {len(model_df)} model observations")
    
    return model_df


if __name__ == "__main__":
    # Test data fetching
    df = fetch_and_prepare_data(season="2024-25", max_games=10)
    print("\nSample data:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
