# IPL Analytics Lab 🏏

An interactive IPL data-analysis dashboard built with Python, Pandas, Plotly, SciPy and Streamlit.

## Features

- Tournament overview and match counts by season
- Toss decision win rates by early, middle and recent eras
- Chi-square association test for the recent era (where the data supports it)
- Team batting-first win rates
- Death-over bowler economy and bowler-credited wicket rankings
- Batter runs-per-match and consistency comparisons
- Dataset coverage checks and CSV downloads

## Project files

```text
ipl_analytics_project/
├── app.py
├── requirements.txt
├── README.md
├── ipl_matches.csv
└── ball_by_ball_ipl.csv
```

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).

   **Windows PowerShell**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

4. Streamlit opens the dashboard in your browser, usually at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and both CSV files.
2. Confirm the CSV files are in the same repository folder as `app.py`.
3. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
4. Select the repository, branch, and `app.py` as the main file.
5. Deploy and wait for the build to finish.

The ball-by-ball CSV is large. If GitHub rejects it or deployment storage limits are reached, use Git LFS or move the dataset to an appropriate hosted data source and update `app.py` to load it from there. Do not commit API keys or credentials.

## Analysis notes and limitations

- Era definitions preserve the season grouping in the original script: Early (2007/08–2012 seasons in the mapping), Middle (2013–2018), Recent (2019 onward in the mapping).
- The toss analysis reports association and descriptive win rates; it does not establish causation.
- The death-bowling view uses overs 16–20 and excludes dismissals that are not credited to the bowler from its wicket count. Economy is calculated from the supplied `Bowler Runs Conceded` and `Valid Ball` columns.
- The composite death score adds standardized lower economy and standardized bowler-credited wickets per over. It is an exploratory ranking, not an official cricket statistic.
- Batter consistency uses runs summed per batter per match. It does not adjust for batting position, not-outs, opposition strength or match context.
- The match and ball-by-ball files may cover different subsets of seasons/matches; the Data checks tab makes coverage gaps visible.
