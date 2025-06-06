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
        sampling_column = st.selectbox("Select Column for Sampling", st.session_state.df_checked.columns.tolist())
        sample_size = st.number_input("Number of Samples", min_value=1, max_value=len(st.session_state.df_checked), value=5, step=1)
        method = st.selectbox("Sampling Method", ["Top N (Longest)", "Bottom N (Quickest)", "Random"])

        if method == "Top N (Longest)":
            sample_df = st.session_state.df_checked.sort_values(by=sampling_column, ascending=False).head(sample_size)
        elif method == "Bottom N (Quickest)":
            sample_df = st.session_state.df_checked.sort_values(by=sampling_column, ascending=True).head(sample_size)
        else:
            sample_df = st.session_state.df_checked.sample(n=sample_size, random_state=1)

        st.write("📊 Sampled Records")
        st.dataframe(sample_df)

        sample_output = BytesIO()
        sample_df.to_excel(sample_output, index=False)
        sample_output.seek(0)
        st.download_button("📥 Download Sample Records", data=sample_output,
                           file_name="sampled_requests.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
            st.subheader("🎯 Random Sampling")
            sample_size = st.number_input("Number of Random Samples", min_value=1, max_value=len(df), value=5)
            if st.button("Generate Incident Sample"):
                sample_df = df.sample(n=sample_size, random_state=42)
                st.dataframe(sample_df,height=300, use_container_width=True)

                sample_buffer = BytesIO()
                sample_df.to_csv(sample_buffer, index=False)
                st.download_button("📥 Download Sample Records", data=sample_buffer.getvalue(),
                                   file_name="incident_sample.csv", mime="text/csv")
                
            st.subheader("⚠️ Risk Category Threshold Check")
            risk_col = st.selectbox("Select Risk Level Column", df.columns)

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
    uploaded_hr_file = st.file_uploader("Upload HR Data File", type=["xlsx", "csv"], key="hr")
    uploaded_access_file = st.file_uploader("Upload User Access File", type=["xlsx", "csv"], key="access")
    uploaded_ad_file = st.file_uploader("Upload AD Data File (optional)", type=["xlsx", "csv"], key="ad")

    if uploaded_hr_file and uploaded_access_file:
        hr_df = read_file(uploaded_hr_file, "HR Data Preview")
        access_df = read_file(uploaded_access_file, "User Access Data Preview")

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
            matched_data = pd.merge(access_df, hr_filtered, left_on=access_hr_key, right_on=hr_key, how="left")
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
                        matched_data = pd.merge(matched_data, ad_filtered, left_on=access_ad_key, right_on=ad_key, how='left')
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
            st.markdown("---")
            st.subheader("📅 Dormancy & GAP Analysis")

            date_col = st.selectbox("Select the Last Logon Date Column for GAP Calculation", matched_data.columns)
            if date_col:
                try:
                    matched_data[date_col] = pd.to_datetime(matched_data[date_col], errors="coerce")
                    max_date = matched_data[date_col].max()
                    st.info(f"Latest date in selected column: **{max_date.date()}**")
                    formatted_date = max_date.strftime("%d-%m-%Y")
                    gap_col_name = f"GAP_{formatted_date}"
                    matched_data[gap_col_name] = (max_date - matched_data[date_col]).dt.days
                    # matched_data["GAP"] = (max_date - matched_data[date_col]).dt.days
                    st.success("✅ Last Login GAP column calculated and added.")

                    # Ask user for threshold
                    threshold = st.number_input("Enter dormancy threshold (in days)", min_value=1, value=30)

                    # Apply threshold to the GAP column
                    if gap_col_name in matched_data.columns:
                        dormant_df = matched_data[matched_data[gap_col_name] > threshold]

                        if not dormant_df.empty:
                            st.warning(f"⚠️ {len(dormant_df)} record(s) exceeded the dormancy threshold of {threshold} days.")
                            st.dataframe(dormant_df, height=200, use_container_width=True)

                            # Create downloadable Excel file
                            from io import BytesIO
                            buffer = BytesIO()
                            dormant_df.to_excel(buffer, index=False)
                            buffer.seek(0)

                            st.download_button(
                                label="📥 Download Dormancy Observations",
                                data=buffer.getvalue(),
                                file_name=f"dormancy_observations_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.success("✅ No records exceeded the dormancy threshold.")

                except Exception as e:
                    st.error(f"Error processing GAP calculation: {e}")
                
            # --- AD-HR Join Date Difference ---
            # --- Custom Join Date Differences ---
            st.markdown("---")
            st.subheader("➕ Add Custom Date Difference Columns")

            # Initialize dynamic config store
            if "custom_diffs" not in st.session_state:
                st.session_state.custom_diffs = []

            # Form to add new difference column
            with st.form("date_diff_form", clear_on_submit=True):
                diff_col_name = st.text_input("Enter name for the new difference column (e.g., AD-HR)")
                date_col1 = st.selectbox("Select first (recent) date column", matched_data.columns, key="col1")
                date_col2 = st.selectbox("Select second (earlier) date column", matched_data.columns, key="col2")
                add_btn = st.form_submit_button("➕ Add Date Difference Column")

                if add_btn:
                    if diff_col_name and date_col1 and date_col2:
                        st.session_state.custom_diffs.append({
                            "name": diff_col_name,
                            "col1": date_col1,
                            "col2": date_col2
                        })
                    else:
                        st.warning("⚠️ Please enter a name and select both columns.")

            # Calculate and add each custom date difference column
            for entry in st.session_state.custom_diffs:
                try:
                    matched_data[entry["col1"]] = pd.to_datetime(matched_data[entry["col1"]], errors="coerce")
                    matched_data[entry["col2"]] = pd.to_datetime(matched_data[entry["col2"]], errors="coerce")
                    matched_data[entry["name"]] = (matched_data[entry["col1"]] - matched_data[entry["col2"]]).dt.days
                    st.success(f"✅ '{entry['name']}' column calculated successfully.")
                except Exception as e:
                    st.error(f"Error calculating {entry['name']}: {e}")

            # --- Final Output ---
            st.subheader("📊 Final Data with GAP and Date Difference Columns")
            st.dataframe(matched_data.head(), use_container_width=True)

            output_buffer = io.BytesIO()
            with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
                matched_data.to_excel(writer, index=False, sheet_name="User Access Review")

            st.download_button(
                label="📥 Download Final File with GAP & Date Differences",
                data=output_buffer.getvalue(),
                file_name=f"User_Access_Reviewed_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # --- Random Sampling ---
            st.markdown("---")
            st.subheader("🎯 Random Sampling")
            sample_size = st.number_input("Enter number of random samples to extract", min_value=1, max_value=len(matched_data), value=5)

            if st.button("Generate Sample"):
                sample_df = matched_data.sample(n=sample_size, random_state=42)
                st.dataframe(sample_df)

                sample_buffer = io.BytesIO()
                with pd.ExcelWriter(sample_buffer, engine="xlsxwriter") as writer:
                    sample_df.to_excel(writer, index=False, sheet_name="RandomSample")

                st.download_button(
                    label="📥 Download Random Sample",
                    data=sample_buffer.getvalue(),
                    file_name=f"Random_Sample_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

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