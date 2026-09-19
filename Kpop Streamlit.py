import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from transformers import pipeline

import re

@st.cache_data
def load_lyric_data():
    """Loads the scraped database. Cached so it only reads the CSV once."""
    # Update this path if your CSV name is different
    return pd.read_csv("./data/raw_lyrics/kpop_corporate_lyrics_20260919.csv")

def detect_language(text):
    """Uses Regex to detect Hangul vs English characters."""
    has_hangul = bool(re.search(r'[\uAC00-\uD7A3]', text))
    has_english = bool(re.search(r'[a-zA-Z]', text))
    
    if has_hangul and has_english:
        return "Mixed (Konglish/Dual)"
    elif has_hangul:
        return "Hangul"
    else:
        return "English"



# --- 1. NLP PIPELINE ---

@st.cache_resource
def initialize_sentiment_engine():
    """
    Loads the Hugging Face Multilingual Transformer.
    @st.cache_resource ensures the heavy 660MB model only loads once when the app starts,
    preventing the dashboard from freezing every time the user interacts with it.
    """
    print("Loading Hugging Face Multilingual Transformer...")
    nlp_engine = pipeline(
        "sentiment-analysis", 
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    return nlp_engine

def map_stars_to_polarity(label):
    """Converts Hugging Face 'X stars' text label into a -1.0 to +1.0 float scale."""
    star_mapping = {
        '1 star':  -1.0,
        '2 stars': -0.5,
        '3 stars':  0.0,
        '4 stars':  0.5,
        '5 stars':  1.0
    }
    return star_mapping.get(label, 0.0)

def score_kpop_lyrics(lyrics_list, language_tags):
    """Processes raw lyrics strings and returns a dashboard-ready DataFrame."""
    nlp = initialize_sentiment_engine()
    
    raw_predictions = nlp(lyrics_list)
    structured_data = []
    
    for idx, pred in enumerate(raw_predictions):
        star_label = pred['label']      
        polarity_score = map_stars_to_polarity(star_label)
        
        structured_data.append({
            "line_number": idx + 1,
            "lyric_text": lyrics_list[idx],
            "language_type": language_tags[idx],
            "sentiment_score": polarity_score
        })
        
    return pd.DataFrame(structured_data)

# --- 2. VISUALIZATION ---

def plot_kpop_sentiment_arc(df):
    """Renders the Plotly scatter timeline on the Streamlit dashboard."""
    st.markdown("### 🎤 K-pop Narrative Arc: Sentiment vs. Language")
    
    color_map = {
        'Hangul': '#00ffff',             # Neon Cyan
        'English': '#fe019a',            # Neon Pink
        'Mixed (Konglish/Dual)': '#9b5de5' # Vibrant Purple
    }
    
    df['color'] = df['language_type'].map(color_map).fillna('#ffffff')
    
    fig = go.Figure()

    # Add the base timeline
    fig.add_trace(go.Scatter(
        x=df['line_number'],
        y=df['sentiment_score'],
        mode='lines',
        line=dict(color='#444444', width=2, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Add scatter traces for each language
    for lang in df['language_type'].unique():
        lang_df = df[df['language_type'] == lang]
        fig.add_trace(go.Scatter(
            x=lang_df['line_number'],
            y=lang_df['sentiment_score'],
            mode='markers',
            name=lang,
            marker=dict(
                color=lang_df['color'].iloc[0], 
                size=12, 
                line=dict(width=1.5, color='white')
            ),
            text=lang_df['lyric_text'],
            hovertemplate='<b>Line %{x}</b><br>Score: %{y:.2f}<br><i>"%{text}"</i><extra></extra>'
        ))

    # Add neutral baseline
    fig.add_hline(y=0, line_width=1, line_color="white", opacity=0.3)

    # Style layout
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis_title="Song Timeline (Line Number)",
        yaxis_title="Sentiment Polarity (-1.0 to +1.0)",
        yaxis=dict(range=[-1.1, 1.1]),
        legend=dict(
            title="Lyric Language",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- 3. STREAMLIT APP EXECUTION ---

st.title("Multilingual Sentiment Analytics")
st.write("Evaluating cross-lingual polarity transitions in corporate K-pop tracks.")

# 1. Load the real database
df = load_lyric_data()

# 2. Build the interactive UI filters
col1, col2 = st.columns(2)

with col1:
    artist_list = df['artist'].unique()
    selected_artist = st.selectbox("Select Artist", artist_list)
    
with col2:
    # Filter the song list based on the chosen artist
    artist_songs = df[df['artist'] == selected_artist]['song_title'].unique()
    selected_song = st.selectbox("Select Track", artist_songs)

# 3. Add an execution button to prevent the model from running constantly
if st.button("Analyze Track Narrative"):
    
    # Extract the raw lyrics for the selected song
    song_data = df[(df['artist'] == selected_artist) & (df['song_title'] == selected_song)].iloc[0]
    raw_text = str(song_data['raw_lyrics'])
    
    # Split the text by line breaks and remove empty lines
    lyric_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    # Generate the language tags dynamically using Regex
    dynamic_tags = [detect_language(line) for line in lyric_lines]

        # Run the transformers pipeline and plot
    with st.spinner(f"Analyzing {len(lyric_lines)} lines through the vector space..."):
        df_scored = score_kpop_lyrics(lyric_lines, dynamic_tags)
        
        # --- NEW METRICS SECTION ---
        st.markdown("---") # Add a visual divider
        
        # Calculate summary statistics
        avg_score = df_scored['sentiment_score'].mean()
        dominant_lang = df_scored['language_type'].mode()[0]
        
        # Create 3 columns for a balanced dashboard layout
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric(label="Overall Sentiment (Avg)", value=f"{avg_score:.2f}")
        with m2:
            st.metric(label="Total Analyzed Lines", value=len(df_scored))
        with m3:
            st.metric(label="Dominant Language", value=dominant_lang)
            
        st.markdown("---")
        # ---------------------------
        
        # Render the chart below the metrics
        plot_kpop_sentiment_arc(df_scored)