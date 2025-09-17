import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="MovieLens Data Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('data/movie_ratings.csv')
    return df

# Load the data
try:
    df = load_data()
except FileNotFoundError:
    st.error("File not found. Please make sure 'data/movie_ratings.csv' exists.")
    st.stop()

# Title and description
st.title("🎬 MovieLens Data Analysis Dashboard")
st.markdown("Explore movie ratings from the MovieLens 200k dataset")

# Sidebar filters
st.sidebar.header("Filters")
min_ratings = st.sidebar.slider("Minimum number of ratings for analysis", 1, 200, 50)
selected_genres = st.sidebar.multiselect(
    "Select genres to analyze", 
    options=sorted(df['genres'].str.split('|').explode().unique()),
    default=['Drama', 'Comedy', 'Action', 'Romance']
)

# Process data for visualizations
# Explode genres for genre-based analysis
df_exploded = df.assign(genres=df['genres'].str.split('|')).explode('genres')

# Filter by selected genres if any are selected
if selected_genres:
    df_exploded = df_exploded[df_exploded['genres'].isin(selected_genres)]

# Question 1: Genre breakdown
st.header("1. Genre Breakdown of Rated Movies")
genre_counts = df_exploded['genres'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']

fig1 = px.bar(genre_counts, x='Genre', y='Count', 
              title="Number of Ratings by Genre")
st.plotly_chart(fig1, use_container_width=True)

# Question 2: Highest rated genres
st.header("2. Highest Rated Genres")
genre_ratings = df_exploded.groupby('genres').agg(
    avg_rating=('rating', 'mean'),
    count=('rating', 'count')
).reset_index()
genre_ratings = genre_ratings[genre_ratings['count'] >= min_ratings].sort_values('avg_rating', ascending=False)

fig2 = px.bar(genre_ratings, x='genres', y='avg_rating', 
              title=f"Average Rating by Genre (min {min_ratings} ratings)",
              labels={'genres': 'Genre', 'avg_rating': 'Average Rating'})
st.plotly_chart(fig2, use_container_width=True)

# Question 3: Mean rating by release year
st.header("3. Mean Rating by Movie Release Year")
year_ratings = df.groupby('year').agg(
    avg_rating=('rating', 'mean'),
    count=('rating', 'count')
).reset_index()
year_ratings = year_ratings[year_ratings['count'] >= min_ratings]

fig3 = px.line(year_ratings, x='year', y='avg_rating',
               title=f"Average Rating by Movie Release Year (min {min_ratings} ratings)")
st.plotly_chart(fig3, use_container_width=True)

# Question 4: Best-rated movies with minimum ratings
st.header("4. Best-Rated Movies")

# Calculate movie ratings and counts
movie_stats = df.groupby('title').agg(
    avg_rating=('rating', 'mean'),
    count=('rating', 'count')
).reset_index()

# For at least 50 ratings
min_50 = movie_stats[movie_stats['count'] >= 50].nlargest(5, 'avg_rating')
# For at least 150 ratings
min_150 = movie_stats[movie_stats['count'] >= 150].nlargest(5, 'avg_rating')

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Top 5 Movies with ≥50 Ratings")
    st.dataframe(min_50[['title', 'avg_rating', 'count']].reset_index(drop=True))

with col2:
    st.subheader(f"Top 5 Movies with ≥150 Ratings")
    st.dataframe(min_150[['title', 'avg_rating', 'count']].reset_index(drop=True))

# Extra Credit: Rating by age for selected genres
st.header("5. Rating by Viewer Age for Selected Genres")

if selected_genres:
    age_bins = [0, 18, 25, 35, 45, 55, 100]
    age_labels = ['<18', '18-25', '26-35', '36-45', '46-55', '55+']
    df_exploded['age_group'] = pd.cut(df_exploded['age'], bins=age_bins, labels=age_labels)
    
    age_genre_ratings = df_exploded.groupby(['age_group', 'genres']).agg(
        avg_rating=('rating', 'mean'),
        count=('rating', 'count')
    ).reset_index()
    age_genre_ratings = age_genre_ratings[age_genre_ratings['count'] >= 50]
    
    fig5 = px.line(age_genre_ratings, x='age_group', y='avg_rating', color='genres',
                  title="Average Rating by Age Group and Genre",
                  labels={'age_group': 'Age Group', 'avg_rating': 'Average Rating'})
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Select genres in the sidebar to see rating trends by age.")

# Extra Credit: Ratings count vs mean rating per genre
st.header("6. Ratings Count vs Mean Rating per Genre")

fig6 = px.scatter(genre_ratings, x='count', y='avg_rating', text='genres',
                 title="Ratings Count vs Mean Rating per Genre",
                 labels={'count': 'Number of Ratings', 'avg_rating': 'Average Rating'})
fig6.update_traces(textposition='top center')
st.plotly_chart(fig6, use_container_width=True)

# Calculate correlation
correlation = np.corrcoef(genre_ratings['count'], genre_ratings['avg_rating'])[0, 1]
st.write(f"Correlation between number of ratings and average rating: {correlation:.3f}")

# Add some summary statistics
st.sidebar.header("Dataset Summary")
st.sidebar.write(f"Total ratings: {df.shape[0]:,}")
st.sidebar.write(f"Unique users: {df['user_id'].nunique():,}")
st.sidebar.write(f"Unique movies: {df['movie_id'].nunique():,}")
st.sidebar.write(f"Time span: {df['year'].min()} - {df['year'].max()}")

# Add download button for the processed data
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

csv = convert_df_to_csv(df)
st.sidebar.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='movie_ratings_processed.csv',
    mime='text/csv',
)