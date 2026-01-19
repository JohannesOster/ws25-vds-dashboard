import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="OECD Primary Care Dashboard")

# --- DATA PREP ---


@st.cache_data
def load_data():
    """Load and prepare the raw data."""
    try:
        df_raw = pd.read_csv('data.csv', low_memory=False)
    except FileNotFoundError:
        return None

    # Clean numerical column
    df_raw['OBS_VALUE'] = pd.to_numeric(df_raw['OBS_VALUE'], errors='coerce')

    # Filter Main Data
    df = df_raw[
        (df_raw['TIME_PERIOD'] >= 2016) &
        (df_raw['TIME_PERIOD'] <= 2023) &
        (df_raw['MEASURE'].isin(['ADMRASTH', 'ADMRCOPD', 'ADMRDBUC']))
    ].copy()

    # Map friendly names
    measure_map = {'ADMRASTH': 'Asthma', 'ADMRCOPD': 'COPD', 'ADMRDBUC': 'Diabetes'}
    df['Condition'] = df['MEASURE'].map(measure_map)

    return df


@st.cache_data
def compute_clusters(df, year_range, sex_code):
    """Compute clusters based on filtered data."""
    # Filter data for clustering
    df_filtered = df[
        (df['TIME_PERIOD'].between(year_range[0], year_range[1])) &
        (df['SEX'] == sex_code)
    ]

    # Create pivot table for clustering
    df_model_input = df_filtered.pivot_table(
        index=['REF_AREA', 'Reference area'],
        columns='Condition',
        values='OBS_VALUE',
        aggfunc='mean'
    )

    # Impute missing values with column mean (Critical for clustering)
    df_model_input = df_model_input.fillna(df_model_input.mean())

    # Need at least 3 countries for 3 clusters
    if len(df_model_input) < 3:
        return None

    # K-Means Clustering
    scaler = StandardScaler()
    X = scaler.fit_transform(df_model_input)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_model_input['Cluster'] = kmeans.fit_predict(X).astype(str)

    # PCA for Visualization
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    df_model_input['PC1'] = comps[:, 0]
    df_model_input['PC2'] = comps[:, 1]

    return df_model_input.reset_index()


# Load Data
df_main = load_data()

if df_main is None:
    st.error("Error: 'data.csv' not found. Please upload it to the repository.")
    st.stop()

# --- DASHBOARD HEADER ---
st.title("🏥 OECD Primary Care Quality Explorer")
st.markdown("""
**Objective:** Visualize the effectiveness of primary care systems in avoiding hospital admissions (2016-2023).
**Instructions:** Click on a country in the **Cluster Analysis** chart (left) to filter the detailed views.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Global Settings")
    year_range = st.slider("Time Period", 2016, 2023, (2016, 2023))

    # Sex filter (Mapping readable names to your specific data codes)
    sex_option = st.selectbox("Gender View", ["Total", "Male", "Female"])
    sex_map = {"Total": "_T", "Male": "M", "Female": "F"}
    selected_sex_code = sex_map[sex_option]

# Compute clusters based on current filters (this will update when filters change)
df_clusters = compute_clusters(df_main, year_range, selected_sex_code)

if df_clusters is None:
    st.warning("Not enough data for the selected filters. Please adjust the time period or gender.")
    st.stop()

# Filter the detailed dataset based on sidebar
df_filtered = df_main[
    (df_main['TIME_PERIOD'].between(year_range[0], year_range[1])) &
    (df_main['SEX'] == selected_sex_code)
]

# --- BRUSHING & LINKING LOGIC ---
# This 'brush' enables the interaction. Clicking a point selects that 'Reference area'.
# empty='all' ensures all data is shown when nothing is selected
brush = alt.selection_point(fields=['Reference area'], empty=True)

# Color Palette: Colorblind Safe (Vermillion, Blue, Yellow)
cluster_colors = alt.Scale(domain=['0', '1', '2'], range=['#D55E00', '#0072B2', '#F0E442'])

# CHART 1: PCA MODEL (The "Controller")
scatter = alt.Chart(df_clusters).mark_circle(size=150, opacity=0.8, stroke='black', strokeWidth=1).encode(
    x=alt.X('PC1', title='PC1: General Admission Volume'),
    y=alt.Y('PC2', title='PC2: Condition Specificity'),
    color=alt.Color('Cluster', scale=cluster_colors, legend=alt.Legend(title="Cluster")),
    tooltip=['Reference area', 'Asthma', 'COPD', 'Diabetes'],
    # Gray out points when not selected
    opacity=alt.condition(brush, alt.value(0.9), alt.value(0.1))
).add_params(
    brush
).properties(
    title=f'Cluster Analysis: {sex_option}, {year_range[0]}-{year_range[1]} (Click to Filter)',
    height=350
)

# CHART 2: BAR CHART (Listens to Brush)
bar = alt.Chart(df_filtered).mark_bar().encode(
    x=alt.X('mean(OBS_VALUE)', title='Avg Admissions per 100k'),
    y=alt.Y('Reference area', sort='-x', title=None),
    color=alt.Color('Condition', legend=alt.Legend(orient='bottom')),
    tooltip=['Reference area', 'Condition', 'mean(OBS_VALUE)']
).transform_filter(
    brush
).properties(
    title=f'Admission Rates by Country ({sex_option})',
    height=350
)

# CHART 3: LINE CHART (Listens to Brush)
line = alt.Chart(df_filtered).mark_line(point=True).encode(
    x='TIME_PERIOD:O',
    y=alt.Y('mean(OBS_VALUE)', title='Admissions per 100k'),
    color='Condition',
    tooltip=['TIME_PERIOD', 'Condition', 'mean(OBS_VALUE)']
).transform_filter(
    brush
).properties(
    title='Trends Over Time',
    height=300
)

# CHART 4: HEATMAP (Listens to Brush)
heatmap = alt.Chart(df_filtered).mark_rect().encode(
    x='TIME_PERIOD:O',
    y='Reference area',
    color=alt.Color('mean(OBS_VALUE)', scale=alt.Scale(scheme='blues'), title='Rate'),
    tooltip=['Reference area', 'TIME_PERIOD', 'mean(OBS_VALUE)']
).transform_filter(
    brush
).properties(
    title='Intensity Heatmap',
    height=300
)

# --- LAYOUT ---
left_column = alt.vconcat(scatter, heatmap).resolve_scale(color='independent')
right_column = alt.vconcat(bar, line).resolve_scale(color='independent')
combined_chart = alt.hconcat(left_column, right_column).resolve_scale(color='independent')

st.altair_chart(combined_chart, use_container_width=True)
