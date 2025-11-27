# nba-endgames

An exploration of closing performance by NBA Teams using regression analysis.

## Overview

This project analyzes NBA game data to understand how teams perform in the 4th quarter relative to their 3rd quarter leads or deficits. Using regression models, we can identify:

- **League-wide trends**: How much does a Q3 lead typically translate to a final victory margin?
- **Team-specific performance**: Which teams consistently extend leads or overcome deficits better than average?
- **Underperformers**: Which teams tend to lose leads or struggle to come back more than expected?

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full Analysis

Run the main script to fetch data and generate analysis:

```bash
python main.py --season 2024-25 --save-plots --output-dir output
```

### Options

- `--season`: NBA season to analyze (default: "2024-25")
- `--max-games`: Limit number of games to process (for testing)
- `--min-games`: Minimum games required for team model (default: 10)
- `--output-dir`: Directory for output files (default: "output")
- `--save-plots`: Save visualization plots

### Example Output

The analysis produces:
- `analysis_report.txt`: Full text report with interpretations
- `team_summary.csv`: CSV table of team comparisons
- `model_data.csv`: Raw data used for modeling
- `league_model.png`: Scatter plot with league regression line
- `team_comparison.png`: Bar chart comparing teams to league average
- `team_[ABBREV].png`: Individual team plots

## Project Structure

```
nba-endgames/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py     # NBA API data fetching
│   ├── models.py           # Regression model classes
│   └── analysis.py         # Analysis and visualization
└── README.md
```

## How It Works

### The Model

The core model is a simple linear regression:

```
Final Margin = intercept + slope × Q3 Margin
```

Where:
- **Q3 Margin** = Team's score minus opponent's score at end of 3rd quarter
- **Final Margin** = Team's score minus opponent's score at end of game

### Interpretation

- **League slope (~0.85)**: Each point of Q3 lead typically translates to ~0.85 points of final margin
- **Teams with higher slope**: Better at extending leads and overcoming deficits ("Extenders")
- **Teams with lower slope**: Prone to losing leads and struggling to come back ("Closers")

The slope being less than 1.0 indicates **regression to the mean** - leads tend to shrink in the 4th quarter on average.

## Data Source

Data is fetched from the NBA Stats API using the [nba_api](https://github.com/swar/nba_api) package.

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- matplotlib
- nba_api

## License

See LICENSE file.
