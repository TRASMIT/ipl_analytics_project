from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="IPL Analytics Lab", page_icon="🏏", layout="wide")
BASE = Path(__file__).parent
MATCH_FILE = BASE / "ipl_matches.csv"
BALL_FILE = BASE / "ball_by_ball_ipl.csv"

ERA_MAP = {
    '2007/08': 'Early', '2009': 'Early', '2009/10': 'Early', '2011': 'Early', '2012': 'Early',
    '2013': 'Middle', '2014': 'Middle', '2015': 'Middle', '2016': 'Middle', '2017': 'Middle', '2018': 'Middle',
    '2019': 'Recent', '2020/21': 'Recent', '2021': 'Recent', '2022': 'Recent', '2023': 'Recent',
    '2024': 'Recent', '2025': 'Recent', '2026': 'Recent'
}
BOWLER_CREDITED = {'caught', 'bowled', 'lbw', 'caught and bowled', 'stumped', 'hit wicket'}

@st.cache_data(show_spinner="Loading IPL datasets...")
def load_data():
    matches = pd.read_csv(MATCH_FILE)
    balls = pd.read_csv(BALL_FILE, low_memory=False)
    matches['season'] = matches['season'].astype(str)
    matches['era'] = matches['season'].map(ERA_MAP)
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    balls['Date'] = pd.to_datetime(balls['Date'], errors='coerce')
    balls['Over'] = pd.to_numeric(balls['Over'], errors='coerce')
    for c in ['Batter Runs', 'Bowler Runs Conceded', 'Valid Ball', 'Wicket']:
        balls[c] = pd.to_numeric(balls[c], errors='coerce').fillna(0)
    return matches, balls

if not MATCH_FILE.exists() or not BALL_FILE.exists():
    st.error("Dataset files are missing. Put ipl_matches.csv and ball_by_ball_ipl.csv in the same folder as app.py.")
    st.stop()

matches, balls = load_data()

st.title("🏏 IPL Analytics Lab")
st.caption("Explore toss strategy, match outcomes, death-over bowling and batter consistency using IPL data.")

with st.sidebar:
    st.header("Dashboard filters")
    seasons = sorted(matches['season'].dropna().unique().tolist())
    chosen_seasons = st.multiselect("Seasons", seasons, default=seasons)
    teams = sorted(set(matches['team1'].dropna()) | set(matches['team2'].dropna()))
    chosen_teams = st.multiselect("Teams", teams, default=[] , help="Leave blank to include every team.")
    min_balls = st.slider("Minimum death-over legal balls", 30, 600, 120, 30)

filtered = matches[matches['season'].isin(chosen_seasons)].copy()
if chosen_teams:
    filtered = filtered[filtered['team1'].isin(chosen_teams) | filtered['team2'].isin(chosen_teams)]
filtered_ids = set(filtered['match_id'].dropna().astype(str))
# Ball-by-ball IDs and match-level IDs are normalized as strings for robust joins.
balls_filtered = balls[balls['Match ID'].astype(str).isin(filtered_ids)].copy()

# Overview metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches in selection", f"{len(filtered):,}")
c2.metric("Seasons", filtered['season'].nunique())
c3.metric("Teams represented", len(set(filtered['team1'].dropna()) | set(filtered['team2'].dropna())))
c4.metric("Toss decisions recorded", f"{filtered['toss_decision'].notna().sum():,}")

overview_tab, toss_tab, bowl_tab, bat_tab, predictor_tab, data_tab = st.tabs([
    "Overview", "Toss strategy", "Death bowling", "Batters", "Win predictor", "Data checks"
])

with overview_tab:
    st.subheader("Match outcomes by season")
    season_counts = filtered.groupby('season', as_index=False).agg(matches=('match_id', 'nunique'))
    season_counts['season_sort'] = season_counts['season'].astype(str)
    season_counts = season_counts.sort_values('season_sort')
    fig = px.bar(season_counts, x='season', y='matches', labels={'season':'Season','matches':'Matches'},
                 title='Matches in each season')
    st.plotly_chart(fig, width='stretch')
    st.subheader("Toss decision distribution")
    decision_counts = filtered['toss_decision'].value_counts().rename_axis('Decision').reset_index(name='Matches')
    if not decision_counts.empty:
        st.plotly_chart(px.pie(decision_counts, names='Decision', values='Matches', hole=.45), width='stretch')
    st.caption("Counts reflect the selected match seasons and optional team filter. A match can involve either selected team.")

with toss_tab:
    st.subheader("Does the toss decision relate to winning?")
    toss_data = filtered.dropna(subset=['era', 'toss_decision', 'toss_winner_won']).copy()
    if toss_data.empty:
        st.info("No toss data for the current filters.")
    else:
        rates = toss_data.groupby(['era','toss_decision'], as_index=False).agg(
            win_rate=('toss_winner_won','mean'), matches=('toss_winner_won','size'), wins=('toss_winner_won','sum'))
        rates['win_rate_pct'] = rates['win_rate'] * 100
        fig = px.bar(rates, x='era', y='win_rate', color='toss_decision', barmode='group',
                     category_orders={'era':['Early','Middle','Recent']},
                     labels={'era':'Era','win_rate':'Toss winner match-win rate','toss_decision':'Toss decision'},
                     hover_data={'matches':True,'wins':True,'win_rate':':.1%'}, title='Toss-winner win rate by era and decision')
        fig.add_hline(y=.5, line_dash='dash', line_color='gray', annotation_text='50% reference')
        st.plotly_chart(fig, width='stretch')
        st.dataframe(rates[['era','toss_decision','matches','wins','win_rate_pct']].rename(columns={
            'era':'Era','toss_decision':'Decision','matches':'Matches','wins':'Toss-winner wins','win_rate_pct':'Win rate (%)'}).round(2),
            width='stretch', hide_index=True)
        recent = toss_data[toss_data['era']=='Recent']
        if recent['toss_decision'].nunique() == 2 and recent['toss_winner_won'].nunique() == 2:
            ct = pd.crosstab(recent['toss_decision'], recent['toss_winner_won'])
            if ct.shape == (2,2):
                chi2, p, dof, expected = chi2_contingency(ct)
                a,b,c = st.columns(3)
                a.metric("Chi-square statistic", f"{chi2:.3f}")
                b.metric("p-value", f"{p:.4f}")
                c.metric("Degrees of freedom", str(dof))
                st.caption("Chi-square tests association between toss decision and whether the toss winner won, for the Recent era in the current selection. It does not prove causation. Check expected cell counts before relying on the test.")
            else:
                st.info("The recent-era contingency table does not contain both outcomes for both decisions; test skipped.")
        else:
            st.info("Need both toss decisions and both win/loss outcomes in the Recent era to run the chi-square test.")
        # Correctly infer whether team1 batted first from the toss decision and winner.
        filtered['team1_batted_first'] = np.where(
            filtered['toss_decision'].eq('bat'), filtered['toss_winner'].eq(filtered['team1']),
            np.where(filtered['toss_decision'].eq('field'), filtered['toss_winner'].ne(filtered['team1']), np.nan))
        valid = filtered.dropna(subset=['team1_batted_first','team1_won','era']).copy()
        valid['bat_first_team_won'] = np.where(valid['team1_batted_first'], valid['team1_won'].eq(1), valid['team1_won'].eq(0))
        bf = valid.groupby('era', as_index=False).agg(matches=('bat_first_team_won','size'), bat_first_win_rate=('bat_first_team_won','mean'))
        st.subheader("Did the team batting first win?")
        st.dataframe(bf.rename(columns={'era':'Era','matches':'Matches','bat_first_win_rate':'Bat-first win rate'}).assign(**{'Bat-first win rate':lambda x:(x['Bat-first win rate']*100).round(2)}), hide_index=True, width='stretch')

with bowl_tab:
    st.subheader("Death-over bowling (overs 16–20)")
    if balls_filtered.empty:
        st.info("No ball-by-ball rows match the selected seasons. The ball dataset may cover a different season range from the match dataset.")
    else:
        death = balls_filtered[balls_filtered['Over'].between(16,20)].copy()
        death['bowler_wicket'] = (death['Wicket'].eq(1) & death['Method'].astype(str).str.lower().isin(BOWLER_CREDITED)).astype(int)
        stats = death.groupby('Bowler', dropna=True).agg(
            runs_conceded=('Bowler Runs Conceded','sum'), legal_balls=('Valid Ball','sum'),
            bowler_wickets=('bowler_wicket','sum'), deliveries=('Bowler Runs Conceded','size')).reset_index()
        stats = stats[stats['legal_balls'] >= min_balls].copy()
        if stats.empty:
            st.warning("No bowlers meet the minimum legal-ball threshold. Reduce the filter in the sidebar.")
        else:
            stats['economy'] = stats['runs_conceded'] / stats['legal_balls'] * 6
            stats['wickets_per_over'] = stats['bowler_wickets'] / (stats['legal_balls'] / 6)
            econ_sd, wk_sd = stats['economy'].std(ddof=0), stats['wickets_per_over'].std(ddof=0)
            stats['economy_z'] = -(stats['economy'] - stats['economy'].mean()) / econ_sd if econ_sd else 0
            stats['wickets_z'] = (stats['wickets_per_over'] - stats['wickets_per_over'].mean()) / wk_sd if wk_sd else 0
            stats['death_score'] = stats['economy_z'] + stats['wickets_z']
            left, right = st.columns(2)
            with left:
                st.markdown("**Lowest economy rates**")
                st.dataframe(stats.nsmallest(10,'economy')[['Bowler','legal_balls','bowler_wickets','economy']].round(2), hide_index=True, width='stretch')
            with right:
                st.markdown("**Composite death score**")
                st.dataframe(stats.nlargest(10,'death_score')[['Bowler','legal_balls','bowler_wickets','economy','wickets_per_over','death_score']].round(2), hide_index=True, width='stretch')
            st.caption("Composite score = standardized low economy + standardized bowler-credited wickets per over. It is a descriptive ranking, not an official metric. Run-outs and other non-bowler-credit dismissals are excluded from bowler wickets.")
            st.download_button("Download death-bowling table (CSV)", stats.sort_values('death_score',ascending=False).to_csv(index=False), "death_bowling_rankings.csv", "text/csv")

with bat_tab:
    st.subheader("Batter runs and consistency")
    if balls_filtered.empty:
        st.info("No ball-by-ball rows match the selected seasons.")
    else:
        # Match-level batter runs; a zero-run appearance is not included unless a delivery row exists.
        per_match = balls_filtered.groupby(['Match ID','Batter'], dropna=True).agg(runs=('Batter Runs','sum')).reset_index()
        bstats = per_match.groupby('Batter').agg(matches=('Match ID','nunique'), mean_runs=('runs','mean'), std_runs=('runs','std'), total_runs=('runs','sum')).reset_index()
        bstats = bstats[bstats['matches'] >= 20].copy()
        bstats['cv'] = bstats['std_runs'] / bstats['mean_runs'].replace(0, np.nan)
        x,y = st.columns(2)
        with x:
            st.markdown("**Highest average runs per match**")
            st.dataframe(bstats.nlargest(10,'mean_runs')[['Batter','matches','total_runs','mean_runs','std_runs']].round(2), hide_index=True, width='stretch')
        with y:
            st.markdown("**Lowest run variability (CV)**")
            st.dataframe(bstats.dropna(subset=['cv']).nsmallest(10,'cv')[['Batter','matches','mean_runs','std_runs','cv']].round(2), hide_index=True, width='stretch')
        st.caption("CV is standard deviation divided by mean runs per match. These tables use the selected ball-by-ball records and a minimum of 20 batter-match appearances; they are not adjusted for batting position or not-outs.")
        st.download_button("Download batter statistics (CSV)", bstats.to_csv(index=False), "batter_statistics.csv", "text/csv")


with predictor_tab:
    st.subheader("Second-innings win predictor")
    st.write("Estimate the chasing team's chance of winning from a historical ball-by-ball model. This is not a live-score feed.")

    categorical_features = ['Bat First', 'Bat Second', 'Venue']
    numeric_features = ['Target Score', 'Innings Runs', 'Innings Wickets', 'Balls Remaining', 'Runs to Get', 'Current Run Rate', 'Required Run Rate']
    feature_columns = categorical_features + numeric_features

    @st.cache_resource(show_spinner="Training win-prediction model on historical IPL innings...")
    def train_win_model(ball_file):
        usecols = ['Match ID', 'Innings', 'Target Score', 'Runs to Get', 'Balls Remaining',
                   'Innings Runs', 'Innings Wickets', 'Bat First', 'Bat Second', 'Venue', 'Chased Successfully']
        raw = pd.read_csv(ball_file, usecols=usecols, low_memory=False)
        for col in ['Innings', 'Target Score', 'Runs to Get', 'Balls Remaining', 'Innings Runs', 'Innings Wickets', 'Chased Successfully']:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
        # Only second innings has a chase-success outcome with the desired meaning.
        raw = raw[(raw['Innings'] == 2) & raw['Chased Successfully'].isin([0, 1])].copy()
        raw = raw[raw['Balls Remaining'].between(0, 120) & raw['Target Score'].notna()]
        raw['balls_bowled'] = 120 - raw['Balls Remaining']
        raw['Current Run Rate'] = raw['Innings Runs'] / raw['balls_bowled'].clip(lower=1) * 6
        raw['Required Run Rate'] = raw['Runs to Get'].clip(lower=0) / raw['Balls Remaining'].clip(lower=1) * 6
        # Use a capped sample of deliveries to reduce redundant near-identical states and training time.
        raw = raw.sort_values(['Match ID', 'Balls Remaining'], ascending=[True, False])
        raw = raw[(raw['Balls Remaining'] % 6 == 0) | (raw['Balls Remaining'] <= 12) | (raw['Balls Remaining'] == 0)]
        raw = raw.dropna(subset=['Match ID', 'Bat First', 'Bat Second', 'Venue', 'Chased Successfully'])
        if raw['Match ID'].nunique() < 10 or raw['Chased Successfully'].nunique() < 2:
            raise ValueError('Not enough distinct second-innings matches and outcome classes to train the model.')
        X = raw[feature_columns].copy()
        y = raw['Chased Successfully'].astype(int)
        groups = raw['Match ID']
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        preprocessor = ColumnTransformer(transformers=[
            ('categorical', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
            ('numeric', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ]), numeric_features)
        ])
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(max_iter=500, class_weight=None))])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        metrics = {
            'accuracy': accuracy_score(y.iloc[test_idx], preds),
            'auc': roc_auc_score(y.iloc[test_idx], probs) if y.iloc[test_idx].nunique() == 2 else float('nan'),
            'brier': brier_score_loss(y.iloc[test_idx], probs),
            'train_matches': groups.iloc[train_idx].nunique(),
            'test_matches': groups.iloc[test_idx].nunique(),
            'rows': len(raw)
        }
        return model, metrics, raw

    try:
        model, model_metrics, training_rows = train_win_model(str(BALL_FILE))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Held-out accuracy", f"{model_metrics['accuracy']:.1%}")
        m2.metric("ROC-AUC", f"{model_metrics['auc']:.3f}" if pd.notna(model_metrics['auc']) else "N/A")
        m3.metric("Brier score", f"{model_metrics['brier']:.3f}")
        m4.metric("Held-out matches", f"{model_metrics['test_matches']:,}")
        st.caption(f"Trained on {model_metrics['rows']:,} sampled second-innings states from {model_metrics['train_matches']:,} training matches. The split is by match ID. Accuracy uses a 50% threshold; ROC-AUC measures ranking ability; lower Brier score is better for probability estimates.")

        known_teams = sorted(set(balls['Bat First'].dropna().astype(str)) | set(balls['Bat Second'].dropna().astype(str)))
        known_venues = sorted(balls['Venue'].dropna().astype(str).unique())
        c1, c2 = st.columns(2)
        with c1:
            bat_first = st.selectbox("Team batting first", known_teams, index=0, key='predict_bat_first')
            venue = st.selectbox("Venue", known_venues, key='predict_venue')
            target = st.number_input("Target to chase", min_value=1, max_value=400, value=180, step=1)
        with c2:
            bat_second_options = [t for t in known_teams if t != bat_first]
            bat_second = st.selectbox("Team chasing", bat_second_options, key='predict_bat_second')
            score = st.number_input("Chasing team's current score", min_value=0, max_value=400, value=60, step=1)
            wickets = st.slider("Wickets lost", min_value=0, max_value=10, value=2)
        overs_completed = st.slider("Overs completed", min_value=0.0, max_value=20.0, value=8.0, step=0.1)
        balls_bowled = min(120, max(0, int(round(overs_completed * 6))))
        balls_remaining = max(0, 120 - balls_bowled)
        runs_to_get = max(0, int(target) - int(score))
        current_rr = (int(score) / max(balls_bowled, 1)) * 6
        required_rr = (runs_to_get / max(balls_remaining, 1)) * 6 if runs_to_get > 0 else 0.0
        st.write(f"**Situation:** {runs_to_get} runs needed from {balls_remaining} balls · Required run rate: {required_rr:.2f}")
        if st.button("Predict win probability", type="primary", key='predict_win_button'):
            if score >= target:
                st.success(f"The chase is already complete based on the entered score. {bat_second} has reached the target.")
            elif balls_remaining == 0 or wickets >= 10:
                st.info("The innings is complete without reaching the target. The chasing team cannot win from this state.")
            else:
                prediction = pd.DataFrame([{
                    'Bat First': bat_first, 'Bat Second': bat_second, 'Venue': venue,
                    'Target Score': float(target), 'Innings Runs': float(score),
                    'Innings Wickets': float(wickets), 'Balls Remaining': float(balls_remaining),
                    'Runs to Get': float(runs_to_get), 'Current Run Rate': float(current_rr),
                    'Required Run Rate': float(required_rr)
                }])[feature_columns]
                chase_probability = float(model.predict_proba(prediction)[0, 1])
                first_probability = 1 - chase_probability
                p1, p2 = st.columns(2)
                p1.metric(f"{bat_second} — estimated win probability", f"{chase_probability:.1%}")
                p2.metric(f"{bat_first} — estimated win probability", f"{first_probability:.1%}")
                st.progress(chase_probability, text=f"Estimated chance of successful chase: {chase_probability:.1%}")
                st.caption("Model estimate from historical second-innings states. It does not account for current player line-ups, injuries, weather, or live match data; predictions may be less reliable for unusual venues or situations.")
    except Exception as exc:
        st.error(f"Could not train the win predictor: {exc}")
        st.info("Check that the ball-by-ball CSV contains the expected second-innings columns and that scikit-learn is installed.")


with data_tab:
    st.subheader("Dataset coverage and quality checks")
    d1,d2 = st.columns(2)
    with d1:
        st.markdown("**Match dataset**")
        st.write(f"Rows: {len(matches):,} · Columns: {matches.shape[1]}")
        st.write(f"Missing season: {matches['season'].isna().sum():,}")
        st.write(f"Unmapped seasons: {int(matches['era'].isna().sum()):,}")
        st.dataframe(matches.isna().sum().rename('Missing values').to_frame(), width='stretch')
    with d2:
        st.markdown("**Ball-by-ball dataset**")
        st.write(f"Rows: {len(balls):,} · Columns: {balls.shape[1]}")
        st.write(f"Unique match IDs: {balls['Match ID'].nunique():,}")
        st.write(f"Ball data match IDs not in match file: {len(set(balls['Match ID'].astype(str)) - set(matches['match_id'].astype(str))):,}")
        st.dataframe(balls[['Match ID','Date','Over','Batter','Bowler','Batter Runs','Wicket']].head(20), width='stretch', hide_index=True)
    st.download_button("Download filtered match data (CSV)", filtered.to_csv(index=False), "filtered_matches.csv", "text/csv")

st.divider()
st.caption("Built with Python, Pandas, Plotly, SciPy and Streamlit. Results depend on the coverage and definitions in the supplied datasets.")
