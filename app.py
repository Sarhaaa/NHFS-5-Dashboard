import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="NFHS India Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = "All India National Family Health Survey.xlsx - in.csv"
    
    # The file structure is complex. We need to find the header row.
    # Based on analysis, the row containing "Population and Household Profile" descriptions is likely the header.
    # We'll read the file without a header first to locate it.
    try:
        df_raw = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found. Please ensure the CSV file is in the same directory.")
        return None

    # Find the row index where the 6th column (index 5) starts with "Population and Household Profile"
    header_row_idx = None
    for i, row in df_raw.iterrows():
        # Check if the column exists and is a string
        if isinstance(row[5], str) and row[5].startswith("Population and Household Profile"):
            header_row_idx = i
            break
            
    if header_row_idx is None:
        # Fallback: Try looking for "India/States/UTs" in the first column
        for i, row in df_raw.iterrows():
            if str(row[0]).strip() == "India/States/UTs":
                header_row_idx = i
                break
    
    if header_row_idx is None:
        st.error("Could not determine the header row structure.")
        return None

    # Reload data with the correct header
    df = pd.read_csv(file_path, header=header_row_idx)
    
    # Rename the first three columns standardly if they aren't already
    df.columns.values[0] = "State"
    df.columns.values[1] = "Survey"
    df.columns.values[2] = "Area"

    # Filter out metadata rows (rows where Survey is not NFHS-3 or NFHS-4)
    df = df[df['Survey'].isin(['NFHS-3', 'NFHS-4'])]

    # Drop columns that are unnamed or have numbers as headers (artifacts of the file)
    # We keep State, Survey, Area and columns with " - " in the name (Indicators)
    cols_to_keep = ['State', 'Survey', 'Area']
    indicator_cols = [c for c in df.columns if isinstance(c, str) and " - " in c]
    cols_to_keep.extend(indicator_cols)
    
    df = df[cols_to_keep]

    # Convert numeric columns, handling 'NA', '*', '()'
    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

data = load_data()

if data is not None:
    # --- Title and Intro ---
    st.title("🇮🇳 National Family Health Survey (NFHS) Dashboard")
    st.markdown("Explore data from NFHS-3 and NFHS-4 across Indian States and Union Territories.")

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # 1. State Filter
    all_states = sorted(data['State'].unique())
    selected_states = st.sidebar.multiselect("Select State(s)", all_states, default=all_states[:3])
    if not selected_states:
        selected_states = all_states # Select all if none selected to avoid empty charts

    # 2. Survey Filter
    selected_survey = st.sidebar.multiselect("Select Survey", data['Survey'].unique(), default=['NFHS-4'])
    
    # 3. Area Filter
    selected_area = st.sidebar.multiselect("Select Area", data['Area'].unique(), default=['Total'])

    # --- Indicator Selection Logic ---
    # Extract categories from column headers (text before the first " - ")
    indicator_cols = [c for c in data.columns if c not in ['State', 'Survey', 'Area']]
    categories = sorted(list(set([col.split(" - ")[0] for col in indicator_cols])))
    
    selected_category = st.sidebar.selectbox("Select Indicator Category", categories)
    
    # Filter indicators belonging to selected category
    category_indicators = [col for col in indicator_cols if col.startswith(selected_category)]
    # Remove the category prefix for cleaner display in dropdown
    display_indicators = {col.split(" - ", 1)[1]: col for col in category_indicators}
    
    selected_metric_name = st.sidebar.selectbox("Select Indicator", list(display_indicators.keys()))
    selected_metric_col = display_indicators[selected_metric_name]

    # --- Data Filtering ---
    filtered_df = data[
        (data['State'].isin(selected_states)) &
        (data['Survey'].isin(selected_survey)) &
        (data['Area'].isin(selected_area))
    ]

    # --- Main Dashboard Area ---
    
    # 1. Comparison Bar Chart
    st.subheader(f"Comparison: {selected_metric_name}")
    
    if not filtered_df.empty:
        # Sort values for better visualization
        filtered_df_sorted = filtered_df.sort_values(by=selected_metric_col, ascending=False)
        
        # Dynamic color based on Survey or Area if multiple selected
        color_col = 'Survey' if len(selected_survey) > 1 else 'State'
        if len(selected_area) > 1: color_col = 'Area'

        fig_bar = px.bar(
            filtered_df_sorted,
            x='State',
            y=selected_metric_col,
            color=color_col,
            barmode='group',
            hover_data=['Survey', 'Area'],
            text_auto='.1f',
            title=f"{selected_metric_name} by State"
        )
        fig_bar.update_layout(xaxis_title="State", yaxis_title="Value (%) or Rate")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data available for the current filter selection.")

    # 2. Metric Scorecards (Averages for selected selection)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        avg_val = filtered_df[selected_metric_col].mean()
        st.metric(label=f"Average {selected_metric_name}", value=f"{avg_val:.2f}")
    
    with c2:
        max_row = filtered_df.loc[filtered_df[selected_metric_col].idxmax()] if not filtered_df.empty and filtered_df[selected_metric_col].notna().any() else None
        if max_row is not None:
            st.metric(label="Highest Value", value=f"{max_row[selected_metric_col]:.1f}", delta=max_row['State'])
    
    with c3:
        min_row = filtered_df.loc[filtered_df[selected_metric_col].idxmin()] if not filtered_df.empty and filtered_df[selected_metric_col].notna().any() else None
        if min_row is not None:
            st.metric(label="Lowest Value", value=f"{min_row[selected_metric_col]:.1f}", delta=min_row['State'], delta_color="inverse")

    # 3. Survey Trends Comparison (Scatter/Dot Plot)
    # Only useful if we have multiple surveys or areas for the same state
    st.subheader("State-wise Variance Analysis")
    
    # Pivot for clearer comparison view if possible
    if len(selected_survey) > 1 or len(selected_area) > 1:
        fig_dot = px.scatter(
            filtered_df,
            x=selected_metric_col,
            y="State",
            color="Survey",
            symbol="Area",
            size_max=10,
            title=f"Distribution of {selected_metric_name}",
            height=max(400, len(selected_states) * 30)
        )
        fig_dot.update_traces(marker_size=12)
        fig_dot.update_layout(xaxis_title=selected_metric_name, yaxis_title="State")
        st.plotly_chart(fig_dot, use_container_width=True)
    else:
        st.caption("Select multiple Surveys or Areas to see variance charts.")

    # 4. Data Table
    st.markdown("---")
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df[['State', 'Survey', 'Area', selected_metric_col]].style.highlight_max(axis=0))