import streamlit as st
import pandas as pd
import time
from datetime import datetime
from utils.database import populate_real_data
from utils.csv_importer import populate_from_csv

def create_sidebar(debris_data):
    """Create the sidebar with filtering and search options."""

    st.sidebar.markdown("<h2 class='sidebar-header'>Data Controls</h2>", unsafe_allow_html=True)

    # Data refresh button with detailed feedback
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.sidebar.button("🔄 Fetch Real Data"):
            try:
                with st.spinner("Fetching latest space debris data..."):
                    # Show progress information
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    # Initialize data fetch
                    progress_text.text("Initializing data generation...")
                    progress_bar.progress(20)
                    time.sleep(0.5)

                    # Generate and fetch data
                    progress_text.text("Generating orbital parameters...")
                    progress_bar.progress(40)
                    success = populate_real_data()

                    if success:
                        # Update progress
                        progress_text.text("Processing collision risks...")
                        progress_bar.progress(70)
                        time.sleep(0.5)

                        progress_text.text("Updating visualization...")
                        progress_bar.progress(90)
                        time.sleep(0.5)

                        progress_text.text("Complete!")
                        progress_bar.progress(100)
                        st.success("✅ Successfully updated with new NASA orbital data!")
                        time.sleep(1)  # Give user time to see success message
                        st.session_state.last_update = time.time()
                        st.rerun()
                    else:
                        st.error("❌ Failed to fetch new data. Using existing data.")

            except Exception as e:
                st.error(f"Error during data update: {str(e)}")

    with col2:
        # Show time until next auto-refresh
        if 'last_update' in st.session_state:
            time_since_update = time.time() - st.session_state.last_update
            time_until_refresh = max(0, 180 - time_since_update)  # 3 minutes (180 seconds)
            minutes = int(time_until_refresh // 60)
            seconds = int(time_until_refresh % 60)
            st.caption(f"Auto-refresh in: {minutes}m {seconds}s")


    # Show data timestamp
    if 'last_update' in st.session_state:
        st.sidebar.text(f"Last updated: {datetime.fromtimestamp(st.session_state.last_update).strftime('%H:%M:%S')}")

    # Add CSV import section
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='sidebar-header'>Import CSV Data</h3>", unsafe_allow_html=True)
    
    # File upload widgets for CSV files
    uploaded_files = st.sidebar.file_uploader("Upload CSV file(s)", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        # Save uploaded files temporarily
        temp_csv_paths = []
        for file in uploaded_files:
            file_path = f"temp_{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            temp_csv_paths.append(file_path)
        
        # Display options for what to do with the CSV data
        csv_action = st.sidebar.radio(
            "Action for CSV data:",
            ["Import to database", "Train models", "Both"]
        )
            
        if st.sidebar.button("Process CSV Data"):
            try:
                with st.spinner("Processing CSV data..."):
                    # Show progress
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    progress_text.text("Reading CSV files...")
                    progress_bar.progress(30)
                    
                    # Store paths for training if needed
                    if csv_action in ["Train models", "Both"]:
                        # Store paths for model training (will be used in main.py)
                        st.session_state.csv_files_for_training = temp_csv_paths.copy()
                    
                    # Import to database if selected
                    if csv_action in ["Import to database", "Both"]:
                        progress_text.text("Importing to database...")
                        progress_bar.progress(50)
                        success = populate_from_csv(temp_csv_paths)
                        
                        if success:
                            progress_text.text("Processing data...")
                            progress_bar.progress(70)
                            time.sleep(0.5)
                            
                            progress_text.text("Updating visualization...")
                            progress_bar.progress(90)
                            time.sleep(0.5)
                            
                            progress_text.text("Complete!")
                            progress_bar.progress(100)
                            st.success("✅ Successfully processed CSV data!")
                            time.sleep(1)
                            st.session_state.last_update = time.time()
                            
                            if csv_action == "Both":
                                st.info("CSV data imported to database. You can now train models in the Model Status panel.")
                            
                            st.rerun()
                        else:
                            st.error("❌ Failed to import CSV data to database.")
                    elif csv_action == "Train models":
                        progress_text.text("Preparing data for model training...")
                        progress_bar.progress(100)
                        st.success("✅ CSV files prepared for model training!")
                        st.info("Go to the Model Status panel to start training.")
                        time.sleep(1)
                        st.rerun()
                        
                # Don't cleanup files immediately if they're needed for training
                if csv_action not in ["Train models", "Both"]:
                    # Cleanup temporary files
                    for file_path in temp_csv_paths:
                        try:
                            import os
                            os.remove(file_path)
                        except:
                            pass
                        
            except Exception as e:
                st.error(f"Error processing CSV data: {str(e)}")
                
    st.sidebar.markdown("---")
    
    # Filtering options
    st.sidebar.markdown("<h3 class='sidebar-header'>Filter Options</h3>", unsafe_allow_html=True)

    # Search by ID
    search_id = st.sidebar.text_input("Search by ID")

    # Altitude range filter with orbit type indicators
    alt_range = st.sidebar.slider(
        "Altitude Range (km)",
        min_value=0,
        max_value=36000,  # Extended to include GEO satellites
        value=(0, 36000)
    )

    # Add orbit type indicators
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.markdown("LEO: 0-2000km")
    with col2:
        st.markdown("MEO: 2000-35500km")
    with col3:
        st.markdown("GEO: 35500km+")

    # Risk level filter
    risk_level = st.sidebar.multiselect(
        "Risk Level",
        options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )

    # Size filter
    size_range = st.sidebar.slider(
        "Size Range (m)",
        min_value=0.0,
        max_value=10.0,
        value=(0.0, 10.0)
    )

    # Apply filters to create filtered dataframe
    df = pd.DataFrame(debris_data)

    if search_id:
        df = df[df['id'].str.contains(search_id, case=False)]

    df = df[
        (df['altitude'] >= alt_range[0]) &
        (df['altitude'] <= alt_range[1]) &
        (df['size'] >= size_range[0]) &
        (df['size'] <= size_range[1])
    ]

    risk_mapping = {
        "Low": (0, 0.3),
        "Medium": (0.3, 0.7),
        "High": (0.7, 1.0)
    }

    mask = pd.Series(False, index=df.index)
    for level in risk_level:
        low, high = risk_mapping[level]
        mask |= (df['risk_score'] >= low) & (df['risk_score'] <= high)

    df = df[mask]

    # Display filtered results
    st.sidebar.markdown("<h3 class='sidebar-subheader'>Filtered Results</h3>", unsafe_allow_html=True)
    st.sidebar.dataframe(
        df[['id', 'altitude', 'risk_score']].style.format({
            'altitude': '{:.1f}',
            'risk_score': '{:.2f}'
        }),
        hide_index=True
    )