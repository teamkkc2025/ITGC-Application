import streamlit as st
import pandas as pd
from io import BytesIO
import io
import datetime
import random

st.set_page_config(page_title="ITGC Application", layout="wide")
st.title("📊 ITGC Application")

# User selection
module = st.radio("Select Module", ["User Access Management", "Incident Management", "Change Management"])

# -------------------------
# 🔁 CHANGE MANAGEMENT FLOW
# -------------------------
if module == "Change Management":
    uploaded_file = st.file_uploader("Upload Change Management File (CSV or Excel)", type=["csv", "xlsx"])

    if "df_checked" not in st.session_state:
        st.session_state.df_checked = None

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Select Relevant Columns")
        columns = df.columns.tolist()
        col_request_id = st.selectbox("Request ID Column", columns)
        columns_with_none = ["None"] + columns
        col_raised_date = st.selectbox("Raised Date Column", columns_with_none)
        col_resolved_date = st.selectbox("Resolved Date Column", columns_with_none)

        if st.button("Run Check"):
            df_checked = df.copy()
            df_checked.rename(columns={col_request_id: "request_id"}, inplace=True)

            if col_raised_date != "None":
                df_checked.rename(columns={col_raised_date: "raised_date"}, inplace=True)
                df_checked["raised_date"] = pd.to_datetime(df_checked["raised_date"], errors='coerce')
                df_checked["missing_raised"] = df_checked["raised_date"].isna()
            else:
                df_checked["raised_date"] = pd.NaT
                df_checked["missing_raised"] = False

            if col_resolved_date != "None":
                df_checked.rename(columns={col_resolved_date: "resolved_date"}, inplace=True)
                df_checked["resolved_date"] = pd.to_datetime(df_checked["resolved_date"], errors='coerce')
                df_checked["missing_resolved"] = df_checked["resolved_date"].isna()
            else:
                df_checked["resolved_date"] = pd.NaT
                df_checked["missing_resolved"] = False

            if col_raised_date != "None" and col_resolved_date != "None":
                df_checked["resolved_before_raised"] = df_checked["resolved_date"] < df_checked["raised_date"]
                df_checked["days_to_resolve"] = (df_checked["resolved_date"] - df_checked["raised_date"]).dt.days
            else:
                df_checked["resolved_before_raised"] = False
                df_checked["days_to_resolve"] = None

            st.session_state.df_checked = df_checked

            st.subheader("📊 Summary of Findings")
            st.write(f"Missing Raised Dates: {df_checked['missing_raised'].sum()}")
            st.write(f"Missing Resolved Dates: {df_checked['missing_resolved'].sum()}")
            st.write(f"Resolved Before Raised: {df_checked['resolved_before_raised'].sum()}")

            output = BytesIO()
            df_checked.to_excel(output, index=False)
            output.seek(0)
            st.download_button("📥 Download Full Data with Checks", data=output,
                               file_name="checked_change_management.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.session_state.df_checked is not None:
        st.subheader("📄 Full Data with Calculated Fields")
        st.dataframe(st.session_state.df_checked)

        st.subheader("🎯 Sampling Section")

        # Initialize column count
        if "num_columns" not in st.session_state:
            st.session_state.num_columns = 1

        # Button to add more unique columns
        if st.button("➕ Add Another Unique Column"):
            st.session_state.num_columns += 1

        # Dynamic selectboxes to define uniqueness
        selected_cols = []
        for i in range(st.session_state.num_columns):
            col = st.selectbox(
                f"Select Unique Column {i+1}",
                st.session_state.df_checked.columns.tolist(),
                key=f"sampling_col_{i}"
            )
            if col:
                selected_cols.append(col)

        # Remove duplicates in selection
        sampling_columns = list(set(selected_cols))

        if sampling_columns:
            # Step 1: Get unique combinations
            unique_combinations = st.session_state.df_checked.drop_duplicates(subset=sampling_columns)
            st.info(f"✅ Found {len(unique_combinations)} unique combinations based on {sampling_columns}")

            # Step 2: Sampling parameters
            sample_size = st.number_input(
                "Number of Unique Combinations to Sample",
                min_value=1,
                max_value=len(unique_combinations),
                value=min(5, len(unique_combinations)),
                step=1
            )

            method = st.selectbox("Sampling Method", ["Top N (Longest)", "Bottom N (Quickest)", "Random"])

            # Step 3: Perform sampling
            if method in ["Top N (Longest)", "Bottom N (Quickest)"]:
                if "days_to_resolve" in unique_combinations.columns:
                    ascending = method == "Bottom N (Quickest)"
                    sampled_keys = unique_combinations.sort_values(by="days_to_resolve", ascending=ascending).head(sample_size)
                else:
                    st.error("❌ 'days_to_resolve' column not available for sorting.")
                    sampled_keys = pd.DataFrame()
            else:
                sampled_keys = unique_combinations.sample(n=sample_size, random_state=42)

            if not sampled_keys.empty:
                # Step 4: Merge sampled keys with full data
                sampled_df = st.session_state.df_checked.merge(sampled_keys, on=sampling_columns, how="inner")

                st.success(f"✅ Sampled {len(sampled_df)} records from {sample_size} unique combinations")
                st.dataframe(sampled_df, use_container_width=True)

                # Step 5: Download
                sample_output = BytesIO()
                sampled_df.to_excel(sample_output, index=False)
                sample_output.seek(0)

                st.download_button(
                    label="📥 Download Sampled Records",
                    data=sample_output,
                    file_name="sampled_requests.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("⚠️ Please select at least one unique column to proceed with sampling.")

# -------------------------
# 🧯 INCIDENT MANAGEMENT FLOW
# -------------------------
elif module == "Incident Management":
    uploaded_file = st.file_uploader("Upload Incident Management File (Excel or CSV)", type=["csv", "xlsx"])

    def load_data(uploaded_file):
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
        return None

    def calculate_date_differences(df, start_col, end_col, resolved_col):
        if start_col != "None":
            df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
        if resolved_col != "None":
            df[resolved_col] = pd.to_datetime(df[resolved_col], errors='coerce')
        if end_col != "None":
            df[end_col] = pd.to_datetime(df[end_col], errors='coerce')

        if start_col != "None" and resolved_col != "None":
            df['Start-Resolved'] = (df[resolved_col] - df[start_col]).dt.days
        else:
            df['Start-Resolved'] = None

        if resolved_col != "None" and end_col != "None":
            df['Resolved-Close'] = (df[end_col] - df[resolved_col]).dt.days
        else:
            df['Resolved-Close'] = None

        return df

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.subheader("📋 Incident Management Columns")
            st.write("Data preview:", df.head())

            columns_with_none = ["None"] + df.columns.tolist()
            start_col = st.selectbox("Select Start Date Column", columns_with_none)
            resolved_col = st.selectbox("Select Resolved Date Column", columns_with_none)
            end_col = st.selectbox("Select Close/End Date Column", columns_with_none)

            df = calculate_date_differences(df, start_col, end_col, resolved_col)

            st.write("✅ Updated Data with Date Differences:")
            st.dataframe(df,height=200, use_container_width=True)

            st.download_button("📥 Download Updated File", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="updated_incidents.csv", mime="text/csv")

            # 🔁 Random Sampling
            # 🔁 Enhanced Unique-Based Sampling
            st.subheader("🎯 Random Sampling Based on Unique Combinations")

            # Step 1: Initialize state for dynamic unique columns
            if "incident_num_columns" not in st.session_state:
                st.session_state.incident_num_columns = 1

            # Step 2: Add unique column button
            if st.button("➕ Add Another Unique Column"):
                st.session_state.incident_num_columns += 1

            # Step 3: Generate selectboxes dynamically
            selected_cols = []
            for i in range(st.session_state.incident_num_columns):
                col = st.selectbox(
                    f"Select Unique Column {i+1}",
                    df.columns.tolist(),
                    key=f"incident_sampling_col_{i}"
                )
                if col:
                    selected_cols.append(col)

            sampling_columns = list(set(selected_cols))  # remove duplicates

            if sampling_columns:
                unique_combinations = df.drop_duplicates(subset=sampling_columns)
                st.info(f"✅ Found {len(unique_combinations)} unique combinations based on {sampling_columns}")

                sample_size = st.number_input(
                    "Number of Unique Combinations to Sample",
                    min_value=1,
                    max_value=len(unique_combinations),
                    value=min(5, len(unique_combinations)),
                    step=1
                )

                method = st.selectbox("Sampling Method", ["Top N (Start-Resolved)", "Bottom N (Start-Resolved)", "Random"])

                # Step 4: Perform sampling
                if method in ["Top N (Start-Resolved)", "Bottom N (Start-Resolved)"]:
                    if "Start-Resolved" in unique_combinations.columns:
                        ascending = method == "Bottom N (Start-Resolved)"
                        sampled_keys = unique_combinations.sort_values(by="Start-Resolved", ascending=ascending).head(sample_size)
                    else:
                        st.error("❌ 'Start-Resolved' column not available for sorting.")
                        sampled_keys = pd.DataFrame()
                else:
                    sampled_keys = unique_combinations.sample(n=sample_size, random_state=42)

                # Step 5: Merge sampled keys with original data
                if not sampled_keys.empty:
                    sampled_df = df.merge(sampled_keys, on=sampling_columns, how="inner")

                    st.success(f"✅ Sampled {len(sampled_df)} records from {sample_size} unique combinations.")
                    st.dataframe(sampled_df, height=300, use_container_width=True)

                    sample_buffer = BytesIO()
                    sampled_df.to_csv(sample_buffer, index=False)
                    st.download_button(
                        "📥 Download Sample Records",
                        data=sample_buffer.getvalue(),
                        file_name="incident_sample.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ Please select at least one unique column to proceed with sampling.")


            if risk_col:
                # Extract last word (risk level) regardless of delimiter or format
                df["Parsed_Risk_Level"] = df[risk_col].astype(str).str.extract(r'([Cc]ritical|[Hh]igh|[Mm]edium|[Ll]ow)', expand=False).str.capitalize()

                st.markdown("Define SLA thresholds (in days) for each risk level:")

                # Start-Resolved thresholds
                crit_threshold = st.number_input("Critical Risk Threshold (Start-Resolved)", min_value=0, value=1)
                high_threshold = st.number_input("High Risk Threshold (Start-Resolved)", min_value=0, value=2)
                
                med_threshold = st.number_input("Medium Risk Threshold (Start-Resolved)", min_value=0, value=4)
                low_threshold = st.number_input("Low Risk Threshold (Start-Resolved)", min_value=0, value=6)

                # Resolved-Close thresholds
                crit_close_threshold = st.number_input("Critical Risk Threshold (Resolved-Close)", min_value=0, value=1)
                high_close_threshold = st.number_input("High Risk Threshold (Resolved-Close)", min_value=0, value=1)
                med_close_threshold = st.number_input("Medium Risk Threshold (Resolved-Close)", min_value=0, value=2)
                low_close_threshold = st.number_input("Low Risk Threshold (Resolved-Close)", min_value=0, value=3)

                # Apply filters
                def exceeds_threshold(row):
                    risk = row["Parsed_Risk_Level"]
                    if risk == "Critical":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > crit_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > crit_close_threshold)
                        )
                    elif risk == "High":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > high_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > high_close_threshold)
                        )
                    elif risk == "Medium":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > med_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > med_close_threshold)
                        )
                    elif risk == "Low":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > low_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > low_close_threshold)
                        )
                    return False

                df["Exceeds_Threshold"] = df.apply(exceeds_threshold, axis=1)
                observations_df = df[df["Exceeds_Threshold"] == True]

                if not observations_df.empty:
                    st.warning(f"{len(observations_df)} record(s) exceeded the threshold limits.")
                    st.dataframe(observations_df, height=200, use_container_width=True)

                    obs_buffer = BytesIO()
                    observations_df.to_csv(obs_buffer, index=False)
                    st.download_button("📥 Download Observations File", data=obs_buffer.getvalue(),
                                    file_name="incident_observations.csv", mime="text/csv")
                else:
                    st.success("✅ All records are within threshold limits.")
                
                # ✅ Download full dataset with flags
                st.subheader("📥 Download Full Data with SLA Checks")
                full_buffer = BytesIO()
                with pd.ExcelWriter(full_buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Full_Data")
                st.download_button("Download Full Incident Data", data=full_buffer.getvalue(),
                                file_name="incident_full_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# 🔍 USER ACCESS FLOW
# -------------------------
elif module == "User Access Management":
    def read_file(uploaded_file, label):
        try:
            if uploaded_file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                    except Exception:
                        df = pd.read_csv(uploaded_file, encoding='windows-1252')
                if df.empty or df.columns.size == 0:
                    raise ValueError("No columns found in the CSV file.")
                st.write(f"### {label} (CSV Preview)")
                st.dataframe(df,height=200, use_container_width=True)
                return df
            else:
                sheets = pd.ExcelFile(uploaded_file).sheet_names
                selected_sheet = st.selectbox(f"Select sheet from {label}", sheets, key=label)
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                if df.empty or df.columns.size == 0:
                    raise ValueError("No columns found in the Excel sheet.")
                st.write(f"### {label} ({selected_sheet})")
                st.dataframe(df.head())
                return df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None

    # --- Mapping Step ---
    st.subheader("📂 Upload Files for Mapping")
    with st.expander("ℹ️ Click here for instructions"):
     st.markdown("""
    ### 📝 Steps to Perform Before Downloading:
    
    1. **Upload Required Files**  
       - First, upload the **HR file**, **User Access list**, and optionally the **AD data**.
    
    2. **Sheet Selection**  
       - For Excel uploads, select the sheet in which your relevant data is stored.
    
    3. **Key Column Matching**  
       - Choose the **common key columns** (e.g., *Employee Code*, *Email ID*) to map the datasets together correctly.
    
    4. **Column Mapping**  
       - Select additional fields you want to join from HR or AD into the user access data.
    
    5. **Download Options**  
       - Choose between `Full Dataset` or `Selected Columns Only`.
       - If `Selected Columns Only`, select the specific columns you wish to include in the export.
    
    6. **Export**  
       - Click the download button to export the processed and merged data.
    """)
    
    uploaded_hr_file = st.file_uploader("Upload HR Data File", type=["xlsx", "csv"], key="hr")
    uploaded_access_file = st.file_uploader("Upload User Access File", type=["xlsx", "csv"], key="access")
    uploaded_ad_file = st.file_uploader("Upload AD Data File (optional)", type=["xlsx", "csv"], key="ad")

    if uploaded_hr_file and uploaded_access_file:
        hr_df = read_file(uploaded_hr_file, "HR Data Preview")
        access_df = read_file(uploaded_access_file, "User Access Data Preview")
        def safe_merge(left, right, l_key, r_key, keep_suffix):
    # Normalize keys to string, strip spaces, and upper case
            for df, k in [(left, l_key), (right, r_key)]:
                df[k] = (df[k].astype(str)
                        .str.strip()
                        .str.upper()
                        .replace({'': pd.NA, 'NAN': pd.NA}))
                merged = pd.merge(
                left, right,
                left_on=l_key, right_on=r_key,
                how='left', indicator=True,
                suffixes=('', keep_suffix)
                )
            st.write("🔍 Merge Status:", merged['_merge'].value_counts())
            return merged.drop(columns=['_merge', r_key] if r_key != l_key else ['_merge'])

        if hr_df is not None and access_df is not None:
            # HR join selection
            hr_key = st.selectbox("Select Key Column in HR Data", hr_df.columns)
            access_hr_key = st.selectbox("Select Matching Column in User Access File (for HR)", access_df.columns)
            hr_columns = st.multiselect("Select HR Columns to Join", hr_df.columns.tolist(), default=[hr_key])
            if hr_key not in hr_columns:
                hr_columns.append(hr_key)
            hr_filtered = hr_df[hr_columns]
            access_df[access_hr_key] = access_df[access_hr_key].astype(str)
            hr_filtered[hr_key] = hr_filtered[hr_key].astype(str)
            matched_data = safe_merge(access_df, hr_filtered, access_hr_key, hr_key, '_HR')

            if hr_key in matched_data.columns and hr_key != access_hr_key:
                matched_data.drop(columns=[hr_key], inplace=True)

            # Optional AD Join
            if uploaded_ad_file:
                ad_df = read_file(uploaded_ad_file, "AD Data Preview")
                if ad_df is not None:
                    ad_key = st.selectbox("Select Key Column in AD Data", ad_df.columns)
                    access_ad_key = st.selectbox("Select Matching Column in User Access File (for AD)", matched_data.columns)
                    ad_columns = st.multiselect("Select AD Columns to Join", ad_df.columns.tolist(), default=[ad_key])
                    if ad_key not in ad_columns:
                        ad_columns.append(ad_key)
                    ad_filtered = ad_df[ad_columns]
                    if ad_key in ad_filtered.columns and access_ad_key in matched_data.columns:
                        matched_data[access_ad_key] = matched_data[access_ad_key].astype(str)
                        ad_filtered[ad_key] = ad_filtered[ad_key].astype(str)
                        matched_data = safe_merge(matched_data, ad_filtered, access_ad_key, ad_key, '_AD')

                        if ad_key in matched_data.columns and ad_key != access_ad_key:
                            matched_data.drop(columns=[ad_key], inplace=True)
                    else:
                        st.error("❌ Selected join keys not found in respective datasets.")

            st.markdown("---")
            st.subheader("✅ Final Merged Dataset")
            st.dataframe(matched_data.head())

            export_format = st.radio("Choose export format", ["Excel", "CSV"])
            if export_format == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    matched_data.to_excel(writer, index=False, sheet_name="MappedData")
                st.download_button(
                    label="📥 Download Mapped File as Excel",
                    data=buffer.getvalue(),
                    file_name=f"Mapped_User_Access_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                csv_data = matched_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Mapped File as CSV",
                    data=csv_data,
                    file_name=f"Mapped_User_Access_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            # --- Dormancy ---
            import pandas as pd
            import datetime
            from io import BytesIO
            import matplotlib.pyplot as plt
            import streamlit as st

            st.markdown("---")
            st.subheader("📅 Dormancy & GAP Analysis")

            date_col = st.selectbox("Select the Last Logon Date Column for GAP Calculation", matched_data.columns)

            if date_col:
                try:
                    matched_data[date_col] = pd.to_datetime(matched_data[date_col], errors="coerce")

                    # Allow user to input custom date
                    use_custom_date = st.checkbox("Use custom last login threshold date")

                    if use_custom_date:
                        custom_date = st.date_input("Select custom 'As Of' date for dormancy calculation", value=datetime.date.today())
                        threshold_base_date = pd.to_datetime(custom_date)
                    else:
                        threshold_base_date = matched_data[date_col].max()
                        st.info(f"Latest date in selected column: **{threshold_base_date.date()}**")

                    formatted_date = threshold_base_date.strftime("%d-%m-%Y")
                    gap_col_name = f"GAP_{formatted_date}"
                    matched_data[gap_col_name] = (threshold_base_date - matched_data[date_col]).dt.days

                    st.success("✅ Last Login GAP column calculated and added.")

                    # --- Status Column Selection ---
                    st.markdown("### 🔍 Status Configuration")
                    status_col = st.selectbox("Select the Status Column", 
                                            ["None"] + list(matched_data.columns),
                                            help="Select the column that contains account status information")

                    # Initialize variables
                    status_filter_applied = False
                    active_statuses = []

                    if status_col != "None":
                        # Show unique status values
                        unique_statuses = matched_data[status_col].dropna().unique().tolist()
                        st.info(f"Found status values: {', '.join(map(str, unique_statuses))}")
                        
                        # Let user select which statuses represent "active" accounts
                        active_statuses = st.multiselect(
                            "Select which status values represent ACTIVE accounts",
                            options=unique_statuses,
                            help="Only accounts with these status values will be checked for dormancy"
                        )
                        
                        if active_statuses:
                            status_filter_applied = True
                            active_count = matched_data[matched_data[status_col].isin(active_statuses)].shape[0]
                            st.success(f"✅ Will analyze {active_count} accounts with active status: {', '.join(map(str, active_statuses))}")

                    # Threshold input
                    threshold = st.number_input("Enter dormancy threshold (in days)", min_value=1, value=30)

                    # Filter records exceeding threshold
                    if gap_col_name in matched_data.columns:
                        # Apply status filter if configured
                        if status_filter_applied:
                            # Only check dormancy for accounts with active status
                            analysis_df = matched_data[matched_data[status_col].isin(active_statuses)].copy()
                            filter_description = f"active accounts (status: {', '.join(map(str, active_statuses))})"
                        else:
                            # Check all accounts
                            analysis_df = matched_data.copy()
                            filter_description = "all accounts"
                        
                        # Find dormant records
                        dormant_condition = analysis_df[gap_col_name] > threshold
                        dormant_df = analysis_df[dormant_condition].copy()
                        
                        # Add "Days Since Threshold" column
                        days_since_threshold_col = f"Days_Since_{threshold}d_Threshold"
                        dormant_df[days_since_threshold_col] = dormant_df[gap_col_name] - threshold
                        
                        dormant_count = len(dormant_df)
                        analysis_total = len(analysis_df)
                        within_count = analysis_total - dormant_count

                        if dormant_count > 0:
                            # --- OBSERVATION SUMMARY ---
                            min_days_over = dormant_df[days_since_threshold_col].min()
                            max_days_over = dormant_df[days_since_threshold_col].max()
                            
                            st.markdown("### 🔍 **Observation Summary**")
                            if status_filter_applied:
                                observation_text = (
                                    f"During dormancy analysis of our system, **{dormant_count} unique active users** "
                                    f"have their dormancy exceed the **{threshold}-day threshold** by a range of "
                                    f"**{min_days_over} to {max_days_over} days**. These accounts are marked as active "
                                    f"({', '.join(map(str, active_statuses))}) but have not logged in within the specified timeframe."
                                )
                            else:
                                observation_text = (
                                    f"During dormancy analysis of our system, **{dormant_count} unique users** "
                                    f"have their dormancy exceed the **{threshold}-day threshold** by a range of "
                                    f"**{min_days_over} to {max_days_over} days**."
                                )
                            
                            st.info(observation_text)
                            st.markdown("---")
                            
                            if status_filter_applied:
                                st.warning(f"⚠️ {dormant_count} ACTIVE account(s) exceeded the dormancy threshold of {threshold} days.")
                                st.info(f"📊 Analysis scope: {analysis_total} {filter_description}")
                            else:
                                st.warning(f"⚠️ {dormant_count} account(s) exceeded the dormancy threshold of {threshold} days.")
                            
                            # Reorder columns to show key information first
                            key_columns = []
                            if status_col != "None":
                                key_columns.append(status_col)
                            key_columns.extend([date_col, gap_col_name, days_since_threshold_col])
                            
                            # Add remaining columns
                            other_columns = [col for col in dormant_df.columns if col not in key_columns]
                            display_columns = key_columns + other_columns
                            
                            st.dataframe(dormant_df[display_columns], height=200, use_container_width=True)

                            # 📊 Pie chart of threshold results
                            st.subheader("📈 Records Exceeding vs Within Threshold")
                            st.caption(f"Analysis of {filter_description}")

                            # Pie chart data
                            sizes = [dormant_count, within_count]
                            labels = ["Exceeded", "Within"]
                            colors = ["#ff6666", "#66b3ff"]

                            # Create a larger, clearer figure with white background
                            fig, ax = plt.subplots(figsize=(3.75, 3.75), dpi=100, facecolor='white')

                            # Plot the pie
                            wedges, texts, autotexts = ax.pie(
                                sizes,
                                labels=None,
                                autopct='%.1f%%',
                                startangle=90,
                                colors=colors,
                                textprops={'fontsize': 10, 'color': 'black'}
                            )
                            ax.axis('equal')

                            # Add readable legend below chart
                            ax.legend(
                                wedges,
                                [f"{labels[i]} ({sizes[i]})" for i in range(len(labels))],
                                loc='lower center',
                                fontsize=10,
                                frameon=False,
                                bbox_to_anchor=(0.5, -0.15),
                                ncol=2
                            )

                            # Save to buffer
                            buf = BytesIO()
                            fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white')
                            buf.seek(0)

                            # Show image in Streamlit with suitable width
                            st.image(buf, width=375)

                            # Summary statistics
                            st.markdown("### 📊 Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Analyzed", analysis_total)
                            with col2:
                                st.metric("Exceeding Threshold", dormant_count)
                            with col3:
                                avg_days_over = dormant_df[days_since_threshold_col].mean()
                                st.metric("Avg Days Over Threshold", f"{avg_days_over:.1f}")

                            # ⬇️ Download Options
                            st.markdown("### ⬇️ Download Dormancy Observations")
                            option = st.radio("Select download type", ["Full Dataset", "Selected Columns Only"], key="dormancy_dl")

                            if option == "Selected Columns Only":
                                sel_cols = st.multiselect("Select columns to include", dormant_df.columns.tolist(), key="dormancy_cols")
                                if sel_cols:
                                    export_df = dormant_df[sel_cols]
                                else:
                                    export_df = dormant_df
                            else:
                                export_df = dormant_df

                            # Create Excel file with observation summary at the top
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                # Create a summary sheet with observation text
                                summary_data = {
                                    'Dormancy Analysis Summary': [
                                        f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                        f"Threshold: {threshold} days",
                                        f"Total Accounts Analyzed: {analysis_total}",
                                        f"Accounts Exceeding Threshold: {dormant_count}",
                                        f"Range of Days Over Threshold: {min_days_over} to {max_days_over} days",
                                        "",
                                        # Clean version of observation text (remove markdown formatting)
                                        observation_text.replace("**", "").replace("*", "")
                                    ]
                                }
                                
                                summary_df = pd.DataFrame(summary_data)
                                summary_df.to_excel(writer, sheet_name='Summary', index=False, header=False)
                                
                                # Add the actual data to a separate sheet
                                export_df.to_excel(writer, sheet_name='Dormant_Accounts', index=False)

                            buffer.seek(0)

                            file_suffix = "active_dormant" if status_filter_applied else "all_dormant"
                            st.download_button(
                                label="📥 Download Dormancy Observations",
                                data=buffer.getvalue(),
                                file_name=f"dormancy_observations_{file_suffix}_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        else:
                            if status_filter_applied:
                                st.success(f"✅ No active accounts exceeded the dormancy threshold of {threshold} days.")
                                st.info(f"📊 Analyzed {analysis_total} {filter_description}")
                            else:
                                st.success("✅ No records exceeded the dormancy threshold.")

                except Exception as e:
                    st.error(f"Error processing GAP calculation: {e}")

            # --- Random Sampling ---
            st.markdown("---")
            st.subheader("🎯 Random Sampling")

            # Add column selectors similar to multiple roles check
            sampling_user_col = st.selectbox("Select the User Identifier Column for Sampling", matched_data.columns, key="sampling_user_col")
            sampling_role_col = st.selectbox("Select the Role Column for Sampling", matched_data.columns, key="sampling_role_col")

            if sampling_user_col and sampling_role_col:
                try:
                    # Get unique combinations of user and role
                    unique_combinations = matched_data[[sampling_user_col, sampling_role_col]].drop_duplicates()
                    max_samples = len(unique_combinations)
                    
                    st.info(f"📊 Found {max_samples} unique user-role combinations available for sampling")
                    
                    sample_size = st.number_input(
                        "Enter number of unique user-role combinations to sample", 
                        min_value=1, 
                        max_value=max_samples, 
                        value=min(5, max_samples)
                    )

                    if st.button("Generate Sample"):
                        # Sample unique combinations first
                        sampled_combinations = unique_combinations.sample(n=sample_size, random_state=42)
                        
                        # Create a list to store all matching records
                        sample_records = []
                        
                        # For each sampled combination, get all matching records from original data
                        for _, row in sampled_combinations.iterrows():
                            user_id = row[sampling_user_col]
                            role = row[sampling_role_col]
                            
                            # Find all records matching this user-role combination
                            matching_records = matched_data[
                                (matched_data[sampling_user_col] == user_id) & 
                                (matched_data[sampling_role_col] == role)
                            ]
                            sample_records.append(matching_records)
                        
                        # Combine all sample records
                        if sample_records:
                            sample_df = pd.concat(sample_records, ignore_index=True)
                            
                            st.success(f"✅ Generated sample with {len(sample_df)} records from {sample_size} unique user-role combinations")
                            st.dataframe(sample_df, height=400, use_container_width=True)
                            
                            # Show summary of sampled combinations
                            st.subheader("📋 Sampled User-Role Combinations Summary")
                            summary_df = sample_df.groupby([sampling_user_col, sampling_role_col]).size().reset_index(name='Record_Count')
                            st.dataframe(summary_df, use_container_width=True)

                            # Download functionality
                            sample_buffer = io.BytesIO()
                            with pd.ExcelWriter(sample_buffer, engine="xlsxwriter") as writer:
                                sample_df.to_excel(writer, index=False, sheet_name="UserRoleSample")
                                summary_df.to_excel(writer, index=False, sheet_name="SampleSummary")

                            st.download_button(
                                label="📥 Download User-Role Sample",
                                data=sample_buffer.getvalue(),
                                file_name=f"UserRole_Sample_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.error("❌ No records found for the sampled combinations")
                            
                except Exception as e:
                    st.error(f"Error during user-role sampling: {e}")
            else:
                st.warning("⚠️ Please select both User Identifier and Role columns to enable sampling")

            # --- Multiple Roles Check ---
            st.markdown("---")
            st.subheader("🧍‍♂️ Multiple Roles Per User Check")

            user_col = st.selectbox("Select the User Identifier Column", matched_data.columns, key="user_col")
            role_col = st.selectbox("Select the Role Column", matched_data.columns, key="role_col")
            min_roles = st.number_input("Minimum Number of Roles Allowed Per User", min_value=1, value=1, step=1)

            if user_col and role_col:
                try:
                    # Count roles per user
                    role_counts = matched_data.groupby(user_col)[role_col].nunique().reset_index()
                    role_counts = role_counts.rename(columns={role_col: "Role_Count"})

                    # Filter users exceeding threshold
                    flagged_users = role_counts[role_counts["Role_Count"] > min_roles]

                    if not flagged_users.empty:
                        st.warning(f"⚠️ {len(flagged_users)} user(s) found with more than {min_roles} unique roles.")

                        # Merge back to get original records
                        flagged_df = matched_data[matched_data[user_col].isin(flagged_users[user_col])]
                        st.dataframe(flagged_df, height=400, use_container_width=True)

                        from io import BytesIO
                        flagged_buffer = BytesIO()
                        flagged_df.to_excel(flagged_buffer, index=False)
                        flagged_buffer.seek(0)

                        st.download_button(
                            label="📥 Download Users with Multiple Roles",
                            data=flagged_buffer.getvalue(),
                            file_name=f"multiple_roles_flagged_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.success("✅ No users have more roles than the defined threshold.")
                except Exception as e:
                    st.error(f"Error during multiple roles check: {e}")


            # -------------------------
            # 📉 DATA COMPLETENESS CHECK
            # -------------------------
            elif module == "Data Completeness Check":
                st.subheader("📉 Data Completeness Check")
                uploaded_file = st.file_uploader("Upload File to Check for Nulls (CSV or Excel)", type=["csv", "xlsx"])

                if uploaded_file:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    st.write("📋 Preview of Uploaded Data:")
                    st.dataframe(df.head(), use_container_width=True)

                    # Count nulls and blanks
                    st.subheader("📊 Null or Empty Cell Summary")
                    null_summary = df.isnull().sum()
                    empty_summary = (df.applymap(lambda x: str(x).strip() == '')).sum()

                    summary_df = pd.DataFrame({
                        "Total Rows": len(df),
                        "Null Cells": null_summary,
                        "Empty Strings": empty_summary,
                        "Combined Missing": null_summary + empty_summary
                    })

                    import ace_tools as tools
                    tools.display_dataframe_to_user("Missing Data Summary", summary_df)

                    # 📥 Download
                    buffer = BytesIO()
                    summary_df.to_excel(buffer, index=True)
                    buffer.seek(0)

                    st.download_button(
                        label="📥 Download Missing Value Summary",
                        data=buffer,
                        file_name="missing_data_summary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )






            # --- Role Consistency Check for IT vs Non-IT ---
            st.markdown("---")
            st.subheader("🔐 IT vs Non-IT Access Validation")

            dept_col = st.selectbox("Select Department Column", matched_data.columns, key="dept_check")
            role_col = st.selectbox("Select Role/Access Column", matched_data.columns, key="role_check")

            if dept_col and role_col:
                try:
                    it_roles = set(matched_data[matched_data[dept_col].str.lower().str.contains("it|information technology|i\.t\.", case=False, na=False)][role_col].dropna().unique())
                    non_it_roles = set(matched_data[~matched_data[dept_col].str.lower().str.contains("it|information technology|i\.t\.", case=False, na=False)][role_col].dropna().unique())
                    common_roles = it_roles & non_it_roles

                    if common_roles:
                        flagged_df = matched_data[matched_data[role_col].isin(common_roles)]
                        st.warning("⚠️ Common roles found between IT and non-IT users:")
                        st.dataframe(flagged_df)

                        flagged_buffer = io.BytesIO()
                        with pd.ExcelWriter(flagged_buffer, engine="xlsxwriter") as writer:
                            flagged_df.to_excel(writer, index=False, sheet_name="IT_NonIT_Conflict")

                        st.download_button(
                            label="📥 Download Flagged Records",
                            data=flagged_buffer.getvalue(),
                            file_name=f"IT_NonIT_Conflicts_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.success("✅ No common roles between IT and non-IT users.")
                except Exception as e:
                    st.error(f"Error during IT vs Non-IT access comparison: {e}")
