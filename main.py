import streamlit as st
import pandas as pd
from io import BytesIO
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from typing import Dict, List, Optional
import os

# Azure AI SDK imports with fallback
try:
    from azure.ai.anomalydetector import AnomalyDetectorClient
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
    from azure.core.exceptions import HttpResponseError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    st.error("Azure AI SDKs not installed. Install with: pip install azure-ai-anomalydetector azure-ai-textanalytics azure-identity")

class AzureAIService:
    """Azure AI integration optimized for Azure deployment"""
    
    def __init__(self):
        self.anomaly_client = None
        self.text_client = None
        self.is_azure_hosted = self._check_azure_environment()
        
        if AZURE_AVAILABLE:
            self._initialize_clients()
    
    def _check_azure_environment(self) -> bool:
        """Check if running on Azure App Service"""
        return os.getenv('WEBSITE_SITE_NAME') is not None
    
    def _get_credential(self):
        """Get appropriate credential based on environment"""
        if self.is_azure_hosted:
            return DefaultAzureCredential()
        else:
            # Fallback to environment variables for local development
            return None
    
    def _initialize_clients(self):
        """Initialize Azure AI clients with proper authentication"""
        anomaly_endpoint = os.getenv('ANOMALY_DETECTOR_ENDPOINT')
        text_endpoint = os.getenv('TEXT_ANALYTICS_ENDPOINT')
        
        if self.is_azure_hosted:
            # Use Managed Identity on Azure
            credential = self._get_credential()
            auth_method = "Managed Identity"
            
            if anomaly_endpoint:
                try:
                    self.anomaly_client = AnomalyDetectorClient(
                        endpoint=anomaly_endpoint,
                        credential=credential
                    )
                except Exception as e:
                    st.error(f"Anomaly Detector initialization failed: {e}")
            
            if text_endpoint:
                try:
                    self.text_client = TextAnalyticsClient(
                        endpoint=text_endpoint,
                        credential=credential
                    )
                except Exception as e:
                    st.error(f"Text Analytics initialization failed: {e}")
        
        else:
            # Use API keys for local development
            auth_method = "API Keys"
            anomaly_key = os.getenv('ANOMALY_DETECTOR_KEY')
            text_key = os.getenv('TEXT_ANALYTICS_KEY')
            
            if anomaly_endpoint and anomaly_key:
                try:
                    self.anomaly_client = AnomalyDetectorClient(
                        endpoint=anomaly_endpoint,
                        credential=AzureKeyCredential(anomaly_key)
                    )
                except Exception as e:
                    st.error(f"Anomaly Detector initialization failed: {e}")
            
            if text_endpoint and text_key:
                try:
                    self.text_client = TextAnalyticsClient(
                        endpoint=text_endpoint,
                        credential=AzureKeyCredential(text_key)
                    )
                except Exception as e:
                    st.error(f"Text Analytics initialization failed: {e}")
        
        # Display connection status
        self._display_connection_status(auth_method)
    
    def _display_connection_status(self, auth_method: str):
        """Display Azure AI connection status"""
        anomaly_status = "Connected" if self.anomaly_client else "Not Connected"
        text_status = "Connected" if self.text_client else "Not Connected"
        
        st.sidebar.subheader("Azure AI Status")
        st.sidebar.write(f"Authentication: {auth_method}")
        st.sidebar.write(f"Anomaly Detector: {anomaly_status}")
        st.sidebar.write(f"Text Analytics: {text_status}")
        
        if self.anomaly_client and self.text_client:
            st.sidebar.success("All Azure AI services connected")
        elif self.anomaly_client or self.text_client:
            st.sidebar.warning("Partial Azure AI connection")
        else:
            st.sidebar.error("Azure AI services not connected")
    
    def detect_anomalies(self, df: pd.DataFrame, timestamp_col: str, 
                        value_col: str, granularity: str = "daily") -> Dict:
        """Detect anomalies using Azure Anomaly Detector"""
        if not self.anomaly_client:
            return {"error": "Anomaly Detector client not initialized"}
        
        try:
            # Prepare data
            df_clean = df[[timestamp_col, value_col]].copy()
            df_clean = df_clean.dropna()
            df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col])
            df_clean = df_clean.sort_values(timestamp_col)
            
            if len(df_clean) < 12:
                return {"error": "Need at least 12 data points for anomaly detection"}
            
            # Create time series data for API
            series_data = []
            for _, row in df_clean.iterrows():
                series_data.append({
                    "timestamp": row[timestamp_col].isoformat(),
                    "value": float(row[value_col])
                })
            
            # API request
            request = {
                "series": series_data,
                "granularity": granularity,
                "sensitivity": 95,
                "maxAnomalyRatio": 0.25
            }
            
            response = self.anomaly_client.detect_entire_series(request)
            
            # Process results
            anomalies = []
            for i, is_anomaly in enumerate(response.is_anomaly):
                if is_anomaly:
                    anomalies.append({
                        "index": i,
                        "timestamp": series_data[i]["timestamp"],
                        "value": series_data[i]["value"],
                        "expected_value": response.expected_values[i] if response.expected_values else None
                    })
            
            return {
                "anomalies": anomalies,
                "total_points": len(series_data),
                "anomaly_count": len(anomalies),
                "anomaly_percentage": (len(anomalies) / len(series_data)) * 100
            }
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    def analyze_text(self, texts: List[str]) -> Dict:
        """Analyze text using Azure Text Analytics"""
        if not self.text_client:
            return {"error": "Text Analytics client not initialized"}
        
        try:
            # Filter and limit texts
            valid_texts = [str(text).strip() for text in texts if text and str(text).strip()][:10]
            
            if not valid_texts:
                return {"error": "No valid texts to analyze"}
            
            # Perform analyses
            sentiment_results = self.text_client.analyze_sentiment(valid_texts)
            entity_results = self.text_client.recognize_entities(valid_texts)
            key_phrase_results = self.text_client.extract_key_phrases(valid_texts)
            
            # Process results
            results = []
            for i, text in enumerate(valid_texts):
                result = {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": {
                        "overall": sentiment_results[i].sentiment,
                        "positive_score": sentiment_results[i].confidence_scores.positive,
                        "neutral_score": sentiment_results[i].confidence_scores.neutral,
                        "negative_score": sentiment_results[i].confidence_scores.negative
                    },
                    "entities": [
                        {
                            "text": entity.text,
                            "category": entity.category,
                            "confidence": entity.confidence_score
                        }
                        for entity in entity_results[i].entities
                    ],
                    "key_phrases": key_phrase_results[i].key_phrases
                }
                results.append(result)
            
            return {"results": results}
            
        except Exception as e:
            return {"error": f"Text analysis failed: {str(e)}"}
    
    def generate_summary(self, anomaly_results: Dict, context: str) -> str:
        """Generate analysis summary"""
        if "error" in anomaly_results:
            return f"**{context} Analysis**\n\nError: {anomaly_results['error']}"
        
        anomalies = anomaly_results.get("anomalies", [])
        total_points = anomaly_results.get("total_points", 0)
        anomaly_percentage = anomaly_results.get("anomaly_percentage", 0)
        
        summary = f"""**{context} Analysis Results**

**Data Overview:**
- Total data points: {total_points}
- Anomalies detected: {len(anomalies)}
- Anomaly rate: {anomaly_percentage:.2f}%

**Risk Assessment:**"""
        
        if anomaly_percentage > 10:
            summary += "\n🔴 HIGH RISK: Significant anomalies detected (>10%)"
        elif anomaly_percentage > 5:
            summary += "\n🟡 MEDIUM RISK: Moderate anomalies detected (5-10%)"
        else:
            summary += "\n🟢 LOW RISK: Few anomalies detected (<5%)"
        
        if anomalies:
            summary += f"\n\n**Top Anomalies:**"
            for i, anomaly in enumerate(anomalies[:3]):
                date_str = anomaly['timestamp'][:10]
                expected_str = f" (Expected: {anomaly['expected_value']:.1f})" if anomaly['expected_value'] else ""
                summary += f"\n- {date_str}: {anomaly['value']:.1f}{expected_str}"
        
        return summary

# Utility functions
def robust_parse_date(date_val):
    try:
        return parser.parse(str(date_val), dayfirst=False, yearfirst=False)
    except Exception:
        try:
            return parser.parse(str(date_val), dayfirst=True, yearfirst=False)
        except Exception:
            return pd.NaT

def flexible_priority_sampling(df, priority_col, sort_col, sample_size_per_category=3):
    """Flexible sampling function for priority-based analysis"""
    df_copy = df.copy()
    df_copy[priority_col] = df_copy[priority_col].fillna('Unknown').astype(str).str.strip()
    df_copy = df_copy[df_copy[priority_col] != '']

    if len(df_copy) == 0:
        return pd.DataFrame(), {}

    unique_priorities = sorted(df_copy[priority_col].unique())
    st.info(f"Found priority levels: {', '.join(unique_priorities)}")

    sample_results = []
    sampling_summary = {}

    for priority in unique_priorities:
        priority_subset = df_copy[df_copy[priority_col] == priority].copy()
        if len(priority_subset) == 0:
            continue
            
        try:
            if priority_subset[sort_col].dtype in ['int64', 'float64']:
                sorted_subset = priority_subset.sort_values(by=sort_col, ascending=True, na_last=True)
            else:
                try:
                    priority_subset[sort_col + '_numeric'] = pd.to_numeric(priority_subset[sort_col], errors='coerce')
                    sorted_subset = priority_subset.sort_values(by=sort_col + '_numeric', ascending=True, na_last=True)
                    sorted_subset = sorted_subset.drop(columns=[sort_col + '_numeric'])
                except:
                    sorted_subset = priority_subset.sort_values(by=sort_col, ascending=True, na_last=True)
        except Exception:
            sorted_subset = priority_subset.copy()

        total_count = len(sorted_subset)
        if total_count <= sample_size_per_category * 2:
            sample = sorted_subset.copy()
            sample['Sample_Type'] = f'{priority}_All_{total_count}_records'
            sample_results.append(sample)
            sampling_summary[priority] = {
                'total_records': total_count,
                'samples_taken': total_count,
                'sampling_method': 'All records'
            }
        else:
            bottom_sample = sorted_subset.head(sample_size_per_category).copy()
            top_sample = sorted_subset.tail(sample_size_per_category).copy()
            bottom_sample['Sample_Type'] = f'{priority}_Bottom_{sample_size_per_category}'
            top_sample['Sample_Type'] = f'{priority}_Top_{sample_size_per_category}'
            sample_results.append(bottom_sample)
            sample_results.append(top_sample)
            sampling_summary[priority] = {
                'total_records': total_count,
                'samples_taken': sample_size_per_category * 2,
                'sampling_method': f'Top {sample_size_per_category} and Bottom {sample_size_per_category}'
            }
    
    if sample_results:
        final_sample = pd.concat(sample_results, ignore_index=True)
        return final_sample, sampling_summary
    else:
        return pd.DataFrame(), {}

# Streamlit App Configuration
st.set_page_config(
    page_title="Azure AI Enhanced ITGC Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Azure AI Enhanced ITGC Application")

# Initialize Azure AI Service
azure_ai = AzureAIService()

# Module Selection
module = st.radio("Select Module", [
    "Change Management",
    "Incident Management", 
    "User Access Management",
    "AI Analytics Dashboard"
])

# AI Analytics Dashboard
if module == "AI Analytics Dashboard":
    st.subheader("AI-Powered Analytics Dashboard")
    
    uploaded_file = st.file_uploader("Upload Data for AI Analysis", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Load data
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Column selection
            timestamp_cols = [col for col in df.columns if any(keyword in col.lower() 
                            for keyword in ['date', 'time', 'created', 'raised', 'resolved'])]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Time Series Anomaly Detection")
                if timestamp_cols and numeric_cols:
                    timestamp_col = st.selectbox("Select Timestamp Column", timestamp_cols)
                    value_col = st.selectbox("Select Value Column", numeric_cols)
                    granularity = st.selectbox("Select Granularity", ["daily", "hourly", "monthly"])
                    
                    if st.button("Detect Anomalies"):
                        with st.spinner("Analyzing data for anomalies..."):
                            results = azure_ai.detect_anomalies(df, timestamp_col, value_col, granularity)
                        
                        if "error" not in results:
                            st.success(f"Found {results['anomaly_count']} anomalies")
                            summary = azure_ai.generate_summary(results, f"{value_col} Analysis")
                            st.markdown(summary)
                            
                            # Visualization
                            if results['anomalies']:
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                # Plot data
                                df_plot = df.copy()
                                df_plot[timestamp_col] = pd.to_datetime(df_plot[timestamp_col], errors='coerce')
                                df_plot = df_plot.dropna(subset=[timestamp_col])
                                ax.plot(df_plot[timestamp_col], df_plot[value_col], 
                                       label='Data', alpha=0.7)
                                
                                # Highlight anomalies
                                anomaly_dates = [pd.to_datetime(a['timestamp']) for a in results['anomalies']]
                                anomaly_values = [a['value'] for a in results['anomalies']]
                                ax.scatter(anomaly_dates, anomaly_values, color='red', 
                                         s=100, label='Anomalies', zorder=5)
                                
                                ax.set_title(f'Anomaly Detection Results - {value_col}')
                                ax.set_xlabel('Date')
                                ax.set_ylabel(value_col)
                                ax.legend()
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                        else:
                            st.error(results['error'])
                else:
                    st.info("Upload data with timestamp and numeric columns")
            
            with col2:
                st.subheader("Text Analytics")
                if text_cols:
                    text_col = st.selectbox("Select Text Column", text_cols)
                    max_texts = st.number_input("Max texts to analyze", min_value=1, max_value=20, value=5)
                    
                    if st.button("Analyze Text"):
                        with st.spinner("Analyzing text..."):
                            sample_texts = df[text_col].dropna().astype(str).head(max_texts).tolist()
                            text_results = azure_ai.analyze_text(sample_texts)
                        
                        if "error" not in text_results:
                            st.success("Text analysis complete")
                            
                            # Sentiment distribution
                            if "results" in text_results:
                                sentiments = [r['sentiment']['overall'] for r in text_results['results']]
                                sentiment_counts = pd.Series(sentiments).value_counts()
                                
                                if not sentiment_counts.empty:
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    colors = ['green' if x == 'positive' else 'red' if x == 'negative' else 'gray' 
                                            for x in sentiment_counts.index]
                                    sentiment_counts.plot(kind='bar', ax=ax, color=colors)
                                    ax.set_title('Sentiment Distribution')
                                    ax.set_xlabel('Sentiment')
                                    ax.set_ylabel('Count')
                                    plt.xticks(rotation=0)
                                    st.pyplot(fig)
                                    plt.close()
                                
                                # Detailed results
                                st.subheader("Analysis Details")
                                for i, result in enumerate(text_results['results'][:3]):
                                    with st.expander(f"Text {i+1}: {result['text']}"):
                                        st.write(f"**Sentiment:** {result['sentiment']['overall']}")
                                        
                                        if result['entities']:
                                            st.write("**Entities:**")
                                            for entity in result['entities'][:5]:
                                                st.write(f"- {entity['text']} ({entity['category']})")
                                        
                                        if result['key_phrases']:
                                            st.write("**Key Phrases:**")
                                            st.write(", ".join(result['key_phrases'][:5]))
                        else:
                            st.error(text_results['error'])
                else:
                    st.info("Upload data with text columns")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Change Management Module
elif module == "Change Management":
    st.subheader("AI-Enhanced Change Management")
    
    uploaded_file = st.file_uploader("Upload Change Management File", type=["csv", "xlsx"])
    
    if "df_checked" not in st.session_state:
        st.session_state.df_checked = None
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Column selection
            columns = df.columns.tolist()
            col_request_id = st.selectbox("Request ID Column", columns)
            columns_with_none = ["None"] + columns
            col_raised_date = st.selectbox("Raised Date Column", columns_with_none)
            col_resolved_date = st.selectbox("Resolved Date Column", columns_with_none)
            col_description = st.selectbox("Description Column (for AI analysis)", columns_with_none)
            
            if st.button("Run Analysis"):
                # Basic processing
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
                
                # Display basic metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Missing Raised Dates", df_checked['missing_raised'].sum())
                with col2:
                    st.metric("Missing Resolved Dates", df_checked['missing_resolved'].sum())
                with col3:
                    st.metric("Resolved Before Raised", df_checked['resolved_before_raised'].sum())
                
                # AI Analysis
                st.subheader("AI Analysis Results")
                
                # Anomaly detection
                if ('days_to_resolve' in df_checked.columns and 
                    df_checked['days_to_resolve'].notna().sum() > 12 and
                    col_raised_date != "None"):
                    
                    st.write("**Resolution Time Anomaly Detection**")
                    time_series_df = df_checked[
                        df_checked['days_to_resolve'].notna() & 
                        df_checked['raised_date'].notna()
                    ].copy()
                    
                    if len(time_series_df) >= 12:
                        with st.spinner("Detecting anomalies..."):
                            anomaly_results = azure_ai.detect_anomalies(
                                time_series_df, 'raised_date', 'days_to_resolve'
                            )
                        
                        if "error" not in anomaly_results:
                            summary = azure_ai.generate_summary(
                                anomaly_results, "Change Resolution Times"
                            )
                            st.markdown(summary)
                        else:
                            st.warning(f"Anomaly detection: {anomaly_results['error']}")
                
                # Text analysis
                if col_description != "None":
                    st.write("**Description Analysis**")
                    descriptions = df_checked[col_description].dropna().astype(str).head(5).tolist()
                    
                    if descriptions:
                        with st.spinner("Analyzing descriptions..."):
                            text_results = azure_ai.analyze_text(descriptions)
                        
                        if "error" not in text_results and "results" in text_results:
                            sentiments = [r['sentiment']['overall'] for r in text_results['results']]
                            sentiment_counts = pd.Series(sentiments).value_counts()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Sentiment Distribution:**")
                                for sentiment, count in sentiment_counts.items():
                                    st.write(f"- {sentiment.title()}: {count}")
                            
                            with col2:
                                all_entities = []
                                for result in text_results['results']:
                                    all_entities.extend([e['text'] for e in result['entities']])
                                
                                if all_entities:
                                    entity_counts = pd.Series(all_entities).value_counts().head(5)
                                    st.write("**Common Systems/Entities:**")
                                    for entity, count in entity_counts.items():
                                        st.write(f"- {entity}: {count}")
                        else:
                            st.warning(f"Text analysis: {text_results.get('error', 'Unknown error')}")
            
            # Download functionality
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            st.download_button(
                "Download Updated Incident Data",
                data=output.getvalue(),
                file_name=f"incident_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# User Access Management Module
elif module == "User Access Management":
    st.subheader("AI-Enhanced User Access Management")
    
    def read_file(uploaded_file, label):
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                sheets = pd.ExcelFile(uploaded_file).sheet_names
                if len(sheets) > 1:
                    selected_sheet = st.selectbox(f"Select sheet from {label}", sheets, key=label)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.write(f"### {label}")
            st.dataframe(df.head())
            return df
            
        except Exception as e:
            st.error(f"Error reading {label}: {str(e)}")
            return None
    
    with st.expander("Instructions"):
        st.markdown("""
        ### Steps to Follow:
        1. Upload HR data file and User Access file
        2. Select matching columns for data joining
        3. Choose additional columns to include
        4. Run AI analysis if desired
        5. Download the merged results
        """)
    
    uploaded_hr_file = st.file_uploader("Upload HR Data File", type=["xlsx", "csv"], key="hr")
    uploaded_access_file = st.file_uploader("Upload User Access File", type=["xlsx", "csv"], key="access")
    
    if uploaded_hr_file and uploaded_access_file:
        hr_df = read_file(uploaded_hr_file, "HR Data")
        access_df = read_file(uploaded_access_file, "User Access Data")
        
        if hr_df is not None and access_df is not None:
            # Column mapping
            hr_key = st.selectbox("HR Key Column", hr_df.columns)
            access_key = st.selectbox("Access Key Column (to match HR)", access_df.columns)
            hr_columns = st.multiselect("Additional HR Columns to Include", 
                                      [col for col in hr_df.columns if col != hr_key])
            
            if st.button("Merge Data"):
                try:
                    # Prepare data for merge
                    hr_subset = hr_df[[hr_key] + hr_columns].copy()
                    
                    # Normalize keys
                    hr_subset[hr_key] = hr_subset[hr_key].astype(str).str.strip().str.upper()
                    access_df[access_key] = access_df[access_key].astype(str).str.strip().str.upper()
                    
                    # Merge data
                    merged_data = pd.merge(
                        access_df, hr_subset,
                        left_on=access_key, right_on=hr_key,
                        how='left', indicator=True
                    )
                    
                    # Show merge statistics
                    merge_stats = merged_data['_merge'].value_counts()
                    st.write("Merge Results:", merge_stats)
                    
                    # Clean up
                    if hr_key != access_key:
                        merged_data.drop(columns=[hr_key], inplace=True)
                    merged_data.drop(columns=['_merge'], inplace=True)
                    
                    st.subheader("Merged Dataset")
                    st.dataframe(merged_data.head())
                    
                    # AI Analysis option
                    if st.button("Run AI Analysis on Access Data"):
                        st.subheader("AI Analysis Results")
                        
                        # Text analysis on role/access descriptions
                        text_columns = [col for col in merged_data.columns if 
                                      any(keyword in col.lower() for keyword in 
                                          ['description', 'role', 'access', 'permission', 'title'])]
                        
                        if text_columns:
                            text_col = st.selectbox("Select column for AI analysis:", text_columns)
                            descriptions = merged_data[text_col].dropna().astype(str).head(5).tolist()
                            
                            if descriptions:
                                with st.spinner("Analyzing access descriptions..."):
                                    text_results = azure_ai.analyze_text(descriptions)
                                
                                if "error" not in text_results and "results" in text_results:
                                    st.write("**Access Description Analysis:**")
                                    
                                    # Extract entities (systems, roles, etc.)
                                    all_entities = []
                                    for result in text_results['results']:
                                        all_entities.extend([e['text'] for e in result['entities']])
                                    
                                    if all_entities:
                                        entity_counts = pd.Series(all_entities).value_counts().head(10)
                                        st.write("**Common Systems/Roles Mentioned:**")
                                        for entity, count in entity_counts.items():
                                            st.write(f"- {entity}: {count}")
                                    
                                    # Show key phrases
                                    all_phrases = []
                                    for result in text_results['results']:
                                        all_phrases.extend(result['key_phrases'])
                                    
                                    if all_phrases:
                                        phrase_counts = pd.Series(all_phrases).value_counts().head(10)
                                        st.write("**Common Key Phrases:**")
                                        for phrase, count in phrase_counts.items():
                                            st.write(f"- {phrase}: {count}")
                                else:
                                    st.warning(f"Text analysis failed: {text_results.get('error', 'Unknown error')}")
                        else:
                            st.info("No suitable text columns found for AI analysis")
                    
                    # Dormancy Analysis
                    st.subheader("Dormancy Analysis")
                    date_columns = [col for col in merged_data.columns if 
                                  any(keyword in col.lower() for keyword in 
                                      ['date', 'login', 'logon', 'access', 'last'])]
                    
                    if date_columns:
                        date_col = st.selectbox("Select Last Login Date Column", date_columns)
                        threshold_days = st.number_input("Dormancy Threshold (days)", min_value=1, value=90)
                        
                        if st.button("Analyze Dormancy"):
                            try:
                                merged_data[date_col] = pd.to_datetime(merged_data[date_col], errors='coerce')
                                current_date = pd.Timestamp.now()
                                merged_data['Days_Since_Login'] = (current_date - merged_data[date_col]).dt.days
                                
                                dormant_users = merged_data[merged_data['Days_Since_Login'] > threshold_days]
                                
                                if len(dormant_users) > 0:
                                    st.warning(f"Found {len(dormant_users)} dormant users (>{threshold_days} days)")
                                    st.dataframe(dormant_users[['Days_Since_Login'] + 
                                                              [col for col in merged_data.columns 
                                                               if col != 'Days_Since_Login']].head())
                                else:
                                    st.success("No dormant users found")
                            
                            except Exception as e:
                                st.error(f"Error in dormancy analysis: {str(e)}")
                    
                    # Download merged data
                    output_format = st.radio("Export Format", ["Excel", "CSV"])
                    
                    if output_format == "Excel":
                        output = BytesIO()
                        merged_data.to_excel(output, index=False)
                        output.seek(0)
                        st.download_button(
                            "Download Merged Data (Excel)",
                            data=output,
                            file_name=f"merged_user_access_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        csv_data = merged_data.to_csv(index=False)
                        st.download_button(
                            "Download Merged Data (CSV)",
                            data=csv_data,
                            file_name=f"merged_user_access_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error merging data: {str(e)}")

# Generate AI Test Summary
if st.sidebar.button("Generate AI Test Summary"):
    st.subheader("AI-Powered Test Summary")
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    summary_content = f"""
# Azure AI Enhanced ITGC Testing Summary

**Generated:** {current_time}  
**Module:** {module}  
**Environment:** {"Azure Cloud" if azure_ai.is_azure_hosted else "Local Development"}

## Executive Summary
This report presents results from AI-enhanced ITGC testing using Azure Cognitive Services for advanced pattern detection and analysis.

## AI Capabilities Deployed
- **Anomaly Detection**: Time series analysis for unusual patterns
- **Text Analytics**: Sentiment analysis and entity extraction
- **Automated Risk Assessment**: AI-driven classification and prioritization

## Key Findings
- Automated detection of process timing anomalies
- Systematic identification of data quality issues
- Evidence-based risk classification using AI

## Technology Stack
- **Platform**: {"Azure App Service with Managed Identity" if azure_ai.is_azure_hosted else "Local Development Environment"}
- **AI Services**: Azure Anomaly Detector, Azure Text Analytics
- **Data Processing**: Python, Pandas, Streamlit

## Recommendations
1. Implement continuous AI-powered monitoring
2. Address identified data quality gaps
3. Enhance controls based on AI-detected patterns
4. Regular automated analysis cycles

## Risk Assessment
AI analysis provides objective, data-driven risk assessment replacing manual review processes with automated intelligence.

---
*Report generated by Azure AI Enhanced ITGC Application*
"""
    
    st.markdown(summary_content)
    
    st.download_button(
        "Download Test Summary",
        data=summary_content.encode('utf-8'),
        file_name=f"ai_test_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

# Footer
st.markdown("---")
st.markdown("**Azure AI Enhanced ITGC Application** | Powered by Azure Cognitive Services")_counts().head(5)
                                    st.write("**Common Entities:**")
                                    for entity, count in entity_counts.items():
                                        st.write(f"- {entity}: {count}")
                        else:
                            st.warning(f"Text analysis: {text_results.get('error', 'Unknown error')}")
                
                # Download results
                output = BytesIO()
                df_checked.to_excel(output, index=False)
                output.seek(0)
                st.download_button(
                    "Download Analysis Results",
                    data=output,
                    file_name=f"change_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Incident Management Module  
elif module == "Incident Management":
    st.subheader("AI-Enhanced Incident Management")
    
    uploaded_file = st.file_uploader("Upload Incident File", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            columns_with_none = ["None"] + df.columns.tolist()
            start_col = st.selectbox("Start Date Column", columns_with_none)
            resolved_col = st.selectbox("Resolved Date Column", columns_with_none)
            end_col = st.selectbox("Close Date Column", columns_with_none)
            description_col = st.selectbox("Description Column", columns_with_none)
            
            # Calculate date differences
            if start_col != "None":
                df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
            if resolved_col != "None":
                df[resolved_col] = pd.to_datetime(df[resolved_col], errors='coerce')
            if end_col != "None":
                df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
            
            if start_col != "None" and resolved_col != "None":
                df['Start-Resolved'] = (df[resolved_col] - df[start_col]).dt.days
            
            if resolved_col != "None" and end_col != "None":
                df['Resolved-Close'] = (df[end_col] - df[resolved_col]).dt.days
            
            st.write("Updated Data:")
            st.dataframe(df.head())
            
            # AI Analysis
            if st.button("Run AI Analysis"):
                st.subheader("AI Analysis Results")
                
                # Anomaly detection on resolution times
                if ('Start-Resolved' in df.columns and 
                    df['Start-Resolved'].notna().sum() > 12 and
                    start_col != "None"):
                    
                    st.write("**Resolution Time Anomaly Detection**")
                    time_series_df = df[
                        df['Start-Resolved'].notna() & 
                        df[start_col].notna()
                    ].copy()
                    
                    if len(time_series_df) >= 12:
                        with st.spinner("Detecting anomalies..."):
                            anomaly_results = azure_ai.detect_anomalies(
                                time_series_df, start_col, 'Start-Resolved'
                            )
                        
                        if "error" not in anomaly_results:
                            summary = azure_ai.generate_summary(
                                anomaly_results, "Incident Resolution Times"
                            )
                            st.markdown(summary)
                        else:
                            st.warning(f"Anomaly detection: {anomaly_results['error']}")
                
                # Text analysis
                if description_col != "None":
                    st.write("**Incident Description Analysis**")
                    descriptions = df[description_col].dropna().astype(str).head(5).tolist()
                    
                    if descriptions:
                        with st.spinner("Analyzing descriptions..."):
                            text_results = azure_ai.analyze_text(descriptions)
                        
                        if "error" not in text_results and "results" in text_results:
                            sentiments = [r['sentiment']['overall'] for r in text_results['results']]
                            sentiment_counts = pd.Series(sentiments).value_counts()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Sentiment Distribution:**")
                                for sentiment, count in sentiment_counts.items():
                                    st.write(f"- {sentiment.title()}: {count}")
                            
                            with col2:
                                all_entities = []
                                for result in text_results['results']:
                                    all_entities.extend([e['text'] for e in result['entities']])
                                
                                if all_entities:
                                    entity_counts = pd.Series(all_entities).value
