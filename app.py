import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="NFHS India Dashboard", layout="wide")

@st.cache_data
def load_data():
    # 1. robust file path handling
    # This looks for the file in the same directory as the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "All India National Family Health Survey.csv")
    
    try:
        # Read the file without header first to locate structure
        df_raw = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        st.error(f"⚠️ File not found! Please upload 'All India National Family Health Survey.xlsx - in.csv' to your GitHub repository in the same folder as app.py.")
        return None

    # 2. Extract Data Rows (The states are at the top)
    # We filter for rows that have 'NFHS-3' or 'NFHS-4' in the second column (index 1)
    data_rows = df_raw[df_raw[1].isin(['NFHS-3', 'NFHS-4'])].copy()

    # 3. Extract Header Row (The headers are at the bottom)
    # We look for the specific row starting with "India/States/UTs"
    header_row = None
    for i, row in df_raw.iterrows():
        if str(row[0]).strip() == "India/States/UTs":
            header_row = row
            break
            
    if header_row is None:
        st.error("Could not locate the header row (starting with 'India/States/UTs'). Check file format.")
        return None

    # 4. Clean and Assign Headers
    clean_headers = []
    for val in header_row:
        val_str = str(val).strip()
        # Rename the first few fixed columns
        if val_str == "India/States/UTs": clean_headers.append("State")
        elif val_str == "Survey": clean_headers.append("Survey")
        elif val_str == "Area": clean_headers.append("Area")
        else: clean_headers.append(val_str)

    # Trim headers or data to match lengths
    if len(clean_headers) > len(data_rows.columns):
        clean_headers = clean_headers[:len(data_rows.columns)]
    else:
        data_rows = data_rows.iloc[:, :len(clean_headers)]

    data_rows.columns = clean_headers

    # 5. Convert Numeric Columns
    # Identify indicator columns (usually long strings or containing " - ")
    # We exclude the first 3 identifier columns
    cols_to_keep = ['State', 'Survey', 'Area']
    indicator_cols = [c for c in data_rows.columns if c not in cols_to_keep and isinstance(c, str)]
    
    final_df = data_rows[cols_to_keep + indicator_cols].copy()

    # Force numeric conversion, turning "NA", "*", "()" into NaN
    for col in indicator_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    return final_df

# Load Data
data = load_data()

if data is not None:
    # --- Dashboard Title ---
    st.title("🇮🇳 NFHS-3 & NFHS-4 Data Dashboard")

    # --- Sidebar ---
    st.sidebar.header("Filter Options")

    # State Selector
    available_states = sorted(data['State'].astype(str).unique())
    selected_states = st.sidebar.multiselect("Select State", available_states, default=available_states[:3])
    if not selected_states: selected_states = available_states

    # Survey Selector
    selected_survey = st.sidebar.multiselect("Select Survey", data['Survey'].unique(), default=['NFHS-4'])
    if not selected_survey: selected_survey = ['NFHS-4']

    # Area Selector
    valid_areas = [x for x in data['Area'].unique() if pd.notna(x)]
    selected_area = st.sidebar.multiselect("Select Area", valid_areas, default=['Total'])
    if not selected_area: selected_area = valid_areas

    # --- Metric Selection Logic ---
    # Group indicators by category (text before " - ")
    indicator_cols = [c for c in data.columns if c not in ['State', 'Survey', 'Area']]
    
    categories = sorted(list(set([c.split(" - ")[0] for c in indicator_cols if " - " in c])))
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Filter indicators for that category
    cat_indicators = [c for c in indicator_cols if c.startswith(selected_category)]
    
    # Create a clean display map (remove the category prefix)
    display_map = {c.split(" - ", 1)[1]: c for c in cat_indicators}
    selected_metric_name = st.sidebar.selectbox("Select Indicator", sorted(list(display_map.keys())))
    selected_metric_col = display_map[selected_metric_name]

    # --- Filtering Data ---
    filtered_df = data[
        (data['State'].isin(selected_states)) &
        (data['Survey'].isin(selected_survey)) &
        (data['Area'].isin(selected_area))
    ]

    # --- Visualizations ---
    st.subheader(f"Analysis: {selected_metric_name}")

    if not filtered_df.empty:
        # Check if column has valid data
        if filtered_df[selected_metric_col].isna().all():
            st.warning("No data available for this indicator with selected filters.")
        else:
            # 1. Bar Chart
            filtered_df = filtered_df.sort_values(selected_metric_col, ascending=False)
            
            fig = px.bar(
                filtered_df,
                x="State",
                y=selected_metric_col,
                color="Survey", 
                barmode="group",
                text_auto='.1f',
                title=f"{selected_metric_name} by State"
            )
            fig.update_layout(xaxis_title="", yaxis_title="Value (%)")
            st.plotly_chart(fig, use_container_width=True)

            # 2. Data Table
            with st.expander("View Raw Data"):
                st.dataframe(filtered_df)
    else:
        st.info("Please select filters to generate the chart.")
