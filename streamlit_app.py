import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Set page configuration
st.set_page_config(
    page_title="EPL 2024/25 Player Similarity Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .position-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        color: white;
    }
    .goalkeeper { background-color: #ffd700; color: black; }
    .defender { background-color: #4caf50; }
    .midfielder { background-color: #2196f3; }
    .forward { background-color: #ff5722; }
    .stDataFrame {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚽ EPL 2024/25 Player Similarity Analysis</h1>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and merge the EPL data including shots data"""
    try:
        # Load data from the data folder
        gca = pd.read_csv('data/player_gca_2025_cleaned.csv')
        poss = pd.read_csv('data/player_possession_2025_cleaned.csv')
        defense = pd.read_csv('data/player_defense_2025_cleaned.csv')
        passes = pd.read_csv('data/player_pass_2025_cleaned.csv')
        
        # Try to load shots data
        try:
            shots = pd.read_csv('data/player_shot_2025_cleaned.csv')
            has_shots = True
        except FileNotFoundError:
            st.warning("Shots data not found. Please add 'player_shot_2025_cleaned.csv' to your data folder for complete analysis.")
            shots = None
            has_shots = False
        
        # Reset indices
        dataframes = [gca, poss, passes, defense]
        if has_shots:
            dataframes.append(shots)
            
        for df in dataframes:
            if df is not None:
                df.reset_index(drop=True, inplace=True)
        
        # Drop unnecessary columns
        cols_to_drop = ['Nation', 'Born', 'Matches']
        for df in dataframes:
            if df is not None:
                for col in cols_to_drop:
                    if col in df.columns:
                        df.drop(col, axis=1, inplace=True)
        
        # Convert data types
        text_cols = ['Player', 'Pos', 'Squad']
        
        for df in dataframes:
            if df is not None:
                for col in df.columns:
                    if col not in text_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Select features for merging
        gca_features = ['Player', 'Pos', 'Squad', 'Age', '90s', 'SCA_TO', 'SCA_PassLive', 'SCA_Shot']
        def_features = ['Player', 'Tkl_Mid', 'Tkl_Att', 'Interceptions']
        poss_features = ['Player', 'Touches_Att3rd', 'Touches_AttPen', 'PrgC', 'Carry1/3', 'Carry_PA']
        
        # Merge dataframes
        base_df = gca[gca_features]
        df = base_df.merge(defense[def_features], on='Player', how='left')
        df = df.merge(poss[poss_features], on='Player', how='left')
        
        # Add shots data if available
        if has_shots:
            shot_features = ['Player', 'Goals', 'Shots', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Dist', 'xG', 'npxG']
            df = df.merge(shots[shot_features], on='Player', how='left')
        
        return df, has_shots
        
    except FileNotFoundError as e:
        st.error(f"Data files not found. Please ensure the CSV files are in the correct directory: {e}")
        return None, False
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, False

def player_finder(df, target_player, features, position_filter=None, top_n=10):
    """Find similar players using cosine similarity"""
    # Check if target player exists
    if target_player not in df['Player'].values:
        return None, f'{target_player} not in the list'
    
    # Filter for more than 10 full games
    df_filtered = df[df['90s'] > 10].copy()
    
    # Apply position filter if specified
    if position_filter and position_filter != "All Positions":
        if position_filter == "Goalkeeper":
            df_filtered = df_filtered[df_filtered['Pos'].str.contains('GK', na=False)]
        elif position_filter == "Defender":
            df_filtered = df_filtered[df_filtered['Pos'].str.contains('DF', na=False)]
        elif position_filter == "Midfielder":
            df_filtered = df_filtered[df_filtered['Pos'].str.contains('MF', na=False)]
        elif position_filter == "Forward":
            df_filtered = df_filtered[df_filtered['Pos'].str.contains('FW', na=False)]
    
    # Check if target player is still in filtered data
    if target_player not in df_filtered['Player'].values:
        return None, f'{target_player} not available in the selected position filter'
    
    # Select only the features we want to compare
    feature_data = df_filtered[features].copy()
    
    # Handle missing data
    feature_data = feature_data.fillna(0)
    
    # Standardize the features
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    
    # Find index of the target player
    target_idx = df_filtered[df_filtered['Player'] == target_player].index[0]
    target_player_idx = df_filtered.index.get_loc(target_idx)
    
    # Calculate cosine similarity
    similarities = cosine_similarity([feature_data_scaled[target_player_idx]], feature_data_scaled)[0]
    
    # Create results DataFrame
    results = df_filtered[['Player', 'Squad', 'Pos', 'Age'] + features].copy()
    results['Similarity'] = similarities
    
    # Sort by similarity (descending) and exclude the target player
    results = results[results['Player'] != target_player].sort_values('Similarity', ascending=False)
    
    return results.head(top_n), None

# Load data
df_result = load_data()
if df_result[0] is not None:
    df, has_shots = df_result
else:
    df, has_shots = None, False

if df is not None:
    # Sidebar controls
    st.sidebar.header("🎯 Analysis Controls")
    
    # Player selection
    players = sorted(df['Player'].unique())
    selected_player = st.sidebar.selectbox(
        "Select Target Player:",
        players,
        index=players.index("Gabriel Martinelli") if "Gabriel Martinelli" in players else 0
    )
    
    # Position filter
    position_options = ["All Positions", "Goalkeeper", "Defender", "Midfielder", "Forward"]
    position_filter = st.sidebar.selectbox(
        "Filter by Position:",
        position_options,
        index=0
    )
    
    # KPI Selection
    st.sidebar.subheader("📊 Select KPIs for Comparison")
    
    # Base features (always available)
    base_features = ['SCA_TO', 'SCA_PassLive', 'SCA_Shot', 'Tkl_Mid', 'Tkl_Att', 
                    'Interceptions', 'Touches_Att3rd', 'Touches_AttPen', 'PrgC', 
                    'Carry1/3', 'Carry_PA']
    
    # Shot features (if available)
    shot_features = ['SoT%', 'G/SoT', 'npxG'] if has_shots else []
    
    all_features = base_features + shot_features
    
    # Feature descriptions
    feature_descriptions = {
        'SCA_TO': 'Shot Creating Actions - Take Ons',
        'SCA_PassLive': 'Shot Creating Actions - Live Passes',
        'SCA_Shot': 'Shot Creating Actions - Shots',
        'Tkl_Mid': 'Tackles in Middle Third',
        'Tkl_Att': 'Tackles in Attacking Third',
        'Interceptions': 'Interceptions',
        'Touches_Att3rd': 'Touches in Attacking Third',
        'Touches_AttPen': 'Touches in Penalty Area',
        'PrgC': 'Progressive Carries',
        'Carry1/3': 'Carries into Final Third',
        'Carry_PA': 'Carries into Penalty Area',
        'SoT%': 'Shots on Target Percentage',
        'G/SoT': 'Goals per Shot on Target',
        'npxG': 'Non-Penalty Expected Goals'
    }
    
    # Preset configurations
    preset_options = ["Custom", "Attacking Focus", "Defensive Focus", "Creative Focus", "All-Round"]
    if has_shots:
        preset_options.append("Finishing Focus")
    
    preset_config = st.sidebar.selectbox(
        "Choose Preset Configuration:",
        preset_options
    )
    
    if preset_config == "Attacking Focus":
        default_features = ['SCA_TO', 'SCA_Shot', 'Touches_Att3rd', 'Touches_AttPen', 'Carry_PA']
        if has_shots:
            default_features.extend(['SoT%', 'npxG'])
    elif preset_config == "Defensive Focus":
        default_features = ['Tkl_Mid', 'Tkl_Att', 'Interceptions']
    elif preset_config == "Creative Focus":
        default_features = ['SCA_PassLive', 'PrgC', 'Carry1/3']
    elif preset_config == "Finishing Focus" and has_shots:
        default_features = ['SCA_Shot', 'SoT%', 'G/SoT', 'npxG', 'Touches_AttPen']
    elif preset_config == "All-Round":
        default_features = all_features
    else:
        # Updated default to match your notebook
        default_features = ['SCA_TO', 'SCA_PassLive', 'SCA_Shot', 'Tkl_Mid', 'Tkl_Att', 
                           'Interceptions', 'Touches_Att3rd', 'Touches_AttPen', 'PrgC', 
                           'Carry1/3', 'Carry_PA']
        if has_shots:
            default_features.extend(['SoT%', 'G/SoT', 'npxG'])
    
    # Feature selection with checkboxes
    st.sidebar.markdown("### Individual KPI Selection")
    selected_features = []
    
    # Group features by category
    st.sidebar.markdown("**🎯 Shot Creating Actions**")
    for feature in ['SCA_TO', 'SCA_PassLive', 'SCA_Shot']:
        if st.sidebar.checkbox(
            feature_descriptions[feature], 
            value=feature in default_features,
            key=feature
        ):
            selected_features.append(feature)
    
    st.sidebar.markdown("**🛡️ Defensive Actions**")
    for feature in ['Tkl_Mid', 'Tkl_Att', 'Interceptions']:
        if st.sidebar.checkbox(
            feature_descriptions[feature], 
            value=feature in default_features,
            key=feature
        ):
            selected_features.append(feature)
    
    st.sidebar.markdown("**⚡ Attacking Movement**")
    for feature in ['Touches_Att3rd', 'Touches_AttPen', 'PrgC', 'Carry1/3', 'Carry_PA']:
        if st.sidebar.checkbox(
            feature_descriptions[feature], 
            value=feature in default_features,
            key=feature
        ):
            selected_features.append(feature)
    
    if has_shots:
        st.sidebar.markdown("**🥅 Finishing**")
        for feature in shot_features:
            if st.sidebar.checkbox(
                feature_descriptions[feature], 
                value=feature in default_features,
                key=feature
            ):
                selected_features.append(feature)
    
    # Number of similar players to show
    top_n = st.sidebar.slider("Number of Similar Players:", 5, 20, 10)
    
    # Analysis button
    if st.sidebar.button("🔍 Find Similar Players", type="primary"):
        if not selected_features:
            st.error("Please select at least one KPI for comparison.")
        else:
            with st.spinner("Analyzing player similarities..."):
                similar_players, error = player_finder(
                    df, selected_player, selected_features, position_filter, top_n
                )
                
                if error:
                    st.error(error)
                else:
                    # Display results in containers with background
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"Players Similar to {selected_player}")
                        
                        # Format the similarity score
                        similar_players['Similarity'] = similar_players['Similarity'].round(3)
                        
                        # Display the dataframe
                        st.dataframe(
                            similar_players,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col2:
                        st.subheader("Target Player Info")
                        target_info = df[df['Player'] == selected_player].iloc[0]
                        
                        st.metric("Position", target_info['Pos'])
                        st.metric("Team", target_info['Squad'])
                        st.metric("Age", int(target_info['Age']))
                        st.metric("Games (90s)", target_info['90s'])
                        
                        if has_shots and 'Goals' in df.columns:
                            st.metric("Goals", int(target_info.get('Goals', 0)))
                            st.metric("xG", round(target_info.get('xG', 0), 1))
                    
                    # Visualization
                    st.subheader("📊 Feature Comparison")
                    
                    # Radar chart
                    if len(selected_features) >= 3:
                        fig = go.Figure()
                        
                        # Target player data
                        target_data = df[df['Player'] == selected_player][selected_features].fillna(0).iloc[0]
                        
                        # Top similar player data
                        top_similar_data = similar_players.iloc[0][selected_features].fillna(0)
                        
                        # Normalize data for radar chart
                        max_vals = df[selected_features].max()
                        target_normalized = target_data / max_vals * 100
                        similar_normalized = top_similar_data / max_vals * 100
                        
                        # Add target player
                        fig.add_trace(go.Scatterpolar(
                            r=target_normalized.values.tolist() + [target_normalized.values[0]],
                            theta=selected_features + [selected_features[0]],
                            fill='toself',
                            name=selected_player,
                            line_color='#ff5722',
                            line_width=3
                        ))
                        
                        # Add most similar player
                        fig.add_trace(go.Scatterpolar(
                            r=similar_normalized.values.tolist() + [similar_normalized.values[0]],
                            theta=selected_features + [selected_features[0]],
                            fill='toself',
                            name=f"Most Similar: {similar_players.iloc[0]['Player']}",
                            line_color='#2196f3',
                            line_width=3
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100],
                                    gridcolor='lightgray'
                                ),
                                bgcolor='white'
                            ),
                            showlegend=True,
                            title="Player Comparison (Normalized %)",
                            font=dict(size=12),
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart of similarities
                    st.subheader("📈 Similarity Scores")
                    fig_bar = px.bar(
                        similar_players.head(10),
                        x='Similarity',
                        y='Player',
                        orientation='h',
                        color='Similarity',
                        color_continuous_scale='viridis',
                        title="Top 10 Most Similar Players"
                    )
                    fig_bar.update_layout(
                        height=400,
                        paper_bgcolor='white',
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Performance comparison chart if shots data available
                    if has_shots and 'Goals' in df.columns:
                        st.subheader("🎯 Performance Metrics")
                        
                        # Get top 5 similar players plus target
                        top_players = similar_players.head(5)
                        target_row = df[df['Player'] == selected_player].iloc[[0]]
                        
                        # Ensure consistent column structure before concatenating
                        target_cols = target_row.columns.tolist()
                        top_cols = top_players.columns.tolist()
                        common_cols = list(set(target_cols) & set(top_cols))
                        
                        target_subset = target_row[common_cols]
                        top_subset = top_players[common_cols]
                        
                        comparison_data = pd.concat([target_subset, top_subset], ignore_index=True)
                        comparison_data['Player_Type'] = ['Target'] + ['Similar'] * len(top_subset)
                        
                        if all(col in comparison_data.columns for col in ['Goals', 'npxG']):
                            fig_scatter = px.scatter(
                                comparison_data,
                                x='npxG',
                                y='Goals',
                                color='Player_Type',
                                hover_data=['Player', 'Squad'],
                                title="Goals vs Expected Goals (npxG)",
                                color_discrete_map={'Target': '#ff5722', 'Similar': '#2196f3'}
                            )
                            fig_scatter.update_layout(
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Display current squad information
    st.subheader("📋 League Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Players per Team:**")
        squad_counts = df.groupby('Squad').size().sort_values(ascending=False)
        st.dataframe(squad_counts.head(10), use_container_width=True)
    
    with col2:
        st.write("**Position Distribution:**")
        pos_counts = df['Pos'].value_counts()
        st.dataframe(pos_counts.head(10), use_container_width=True)
    
    # Feature availability info
    if has_shots:
        st.success("✅ Complete dataset loaded including shooting statistics!")
    else:
        st.info("ℹ️ Basic dataset loaded. Add 'player_shot_2025_cleaned.csv' for complete analysis.")

else:
    st.error("Unable to load data. Please check your CSV files are in the correct location.")
    
    # Instructions for data setup
    st.markdown("""
    ## 📁 Data Setup Instructions
    
    To use this app, you need to place the following CSV files in a `data/` folder:
    
    **Required files:**
    - `player_gca_2025_cleaned.csv`
    - `player_possession_2025_cleaned.csv`
    - `player_defense_2025_cleaned.csv`
    - `player_pass_2025_cleaned.csv`
    
    **Optional file (for complete analysis):**
    - `player_shot_2025_cleaned.csv`
    
    Make sure the file paths in the `load_data()` function match your directory structure.
    """)

# Footer
st.markdown("---")
st.markdown("⚽ **EPL 2024/25 Player Similarity Analysis** | Built with Streamlit")