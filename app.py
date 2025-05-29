import streamlit as st
import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
import tempfile
import io
import sys
import re
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Chat with Your Data",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_ready' not in st.session_state:
    st.session_state.agent_ready = False
if 'df' not in st.session_state:
    st.session_state.df = None

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the CSV data"""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        
        # Convert date column to datetime if exists
        if 'date_joined' in df.columns:
            df['date_joined'] = pd.to_datetime(df['date_joined'])
        
        # Create age groups if age column exists
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def setup_database_and_agent(df, api_key):
    """Setup SQLite database and create SQL agent"""
    try:
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = temp_db.name
        temp_db.close()
        
        # Create SQLite connection and load data
        conn = sqlite3.connect(db_path)
        df.to_sql('customers', conn, if_exists='replace', index=False)
        conn.close()
        
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        # Create SQL database connection
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Create toolkit and agent
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor, db_path
    except Exception as e:
        st.error(f"Error setting up database and agent: {str(e)}")
        return None, None

def parse_and_format_response(response):
    """Parse AI response and format it properly with tables and visualizations"""
    
    # Check if response contains structured data patterns
    patterns = {
        'gender_age_job': r'(Male|Female),\s*([\w-]+),\s*([\w\s]+):\s*([\d.]+)',
        'gender_balance': r'(Male|Female).*?(\d+\.?\d*)',
        'age_group': r'([\d-]+|\d+\+).*?(\d+\.?\d*)',
        'region_data': r'([A-Za-z\s]+).*?(\d+\.?\d*)',
        'job_classification': r'([A-Za-z\s]+Collar|Other).*?(\d+\.?\d*)'
    }
    
    formatted_response = {"text": response, "table": None, "chart": None}
    
    # Try to extract structured data
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        if matches and len(matches) > 3:  # Only process if we have meaningful data
            if pattern_name == 'gender_age_job':
                # Parse gender, age group, job classification data
                data = []
                for match in matches:
                    data.append({
                        'Gender': match[0],
                        'Age Group': match[1],
                        'Job Classification': match[2].strip(),
                        'Average Balance': float(match[3])
                    })
                
                df = pd.DataFrame(data)
                formatted_response["table"] = df
                
                # Create pivot table for better visualization
                if len(df) > 0:
                    pivot_df = df.pivot_table(
                        values='Average Balance', 
                        index=['Gender', 'Age Group'], 
                        columns='Job Classification',
                        aggfunc='mean'
                    ).round(2)
                    formatted_response["pivot_table"] = pivot_df
                
                # Generate summary text
                avg_by_gender = df.groupby('Gender')['Average Balance'].mean()
                summary = f"\n\n**Summary:**\n"
                summary += f"• Overall, {'Female' if avg_by_gender.get('Female', 0) > avg_by_gender.get('Male', 0) else 'Male'} customers have a higher average balance\n"
                summary += f"• Female average: ${avg_by_gender.get('Female', 0):,.2f}\n"
                summary += f"• Male average: ${avg_by_gender.get('Male', 0):,.2f}\n"
                
                highest_combo = df.loc[df['Average Balance'].idxmax()]
                summary += f"• Highest balance combination: {highest_combo['Gender']} {highest_combo['Age Group']} {highest_combo['Job Classification']} (${highest_combo['Average Balance']:,.2f})"
                
                formatted_response["text"] = summary
                break
                
            elif pattern_name in ['gender_balance', 'age_group', 'region_data', 'job_classification']:
                # Parse simpler two-column data
                data = []
                for match in matches:
                    data.append({
                        'Category': match[0].strip(),
                        'Value': float(match[1])
                    })
                
                df = pd.DataFrame(data)
                if len(df) > 0:
                    formatted_response["table"] = df
                    
                    # Generate summary
                    total = df['Value'].sum()
                    highest = df.loc[df['Value'].idxmax()]
                    lowest = df.loc[df['Value'].idxmin()]
                    
                    summary = f"\n\n**Summary:**\n"
                    summary += f"• Total: {total:,.2f}\n"
                    summary += f"• Highest: {highest['Category']} ({highest['Value']:,.2f})\n"
                    summary += f"• Lowest: {lowest['Category']} ({lowest['Value']:,.2f})\n"
                    summary += f"• Average: {df['Value'].mean():,.2f}"
                    
                    formatted_response["text"] = summary
                break
    
    return formatted_response

def ask_question(agent, question):
    """Ask a question to the AI agent"""
    try:
        # Capture stdout to get the agent's reasoning
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Run the agent
        response = agent.run(question)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        return response
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {str(e)}"

def display_formatted_response(formatted_response):
    """Display the formatted response with tables and charts"""
    
    # Display summary text
    if formatted_response["text"]:
        st.markdown(formatted_response["text"])
    
    # Display table if available
    if formatted_response["table"] is not None:
        st.subheader("📊 Detailed Data")
        
        # Display regular table
        st.dataframe(
            formatted_response["table"], 
            use_container_width=True,
            hide_index=True
        )
        
        # Display pivot table if available
        if "pivot_table" in formatted_response and formatted_response["pivot_table"] is not None:
            st.subheader("📋 Pivot Summary")
            st.dataframe(formatted_response["pivot_table"], use_container_width=True)
        
        # Create visualizations
        df = formatted_response["table"]
        
        if len(df.columns) >= 2:
            # Create appropriate chart based on data structure
            if 'Gender' in df.columns and 'Age Group' in df.columns and 'Average Balance' in df.columns:
                # Multi-dimensional data - create grouped bar chart
                fig = px.bar(
                    df, 
                    x='Age Group', 
                    y='Average Balance',
                    color='Gender',
                    facet_col='Job Classification' if 'Job Classification' in df.columns else None,
                    title='Average Balance by Demographics',
                    labels={'Average Balance': 'Average Balance ($)'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif len(df.columns) == 2:
                # Simple two-column data
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig_bar = px.bar(
                        df, 
                        x='Category', 
                        y='Value',
                        title='Bar Chart View'
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Pie chart
                    if df['Value'].min() >= 0:  # Only for non-negative values
                        fig_pie = px.pie(
                            df, 
                            values='Value', 
                            names='Category',
                            title='Distribution View'
                        )
                        fig_pie.update_layout(height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">💬 Chat with Your Data</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API key from Streamlit secrets
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except KeyError:
            st.error("❌ Google API Key not found in Streamlit secrets!")
            st.stop()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file to chat with your data"
        )
        
        # Data preview toggle
        if st.session_state.df is not None:
            show_data = st.checkbox("Show Data Preview", value=False)
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - What is the average balance by gender?
        - How many customers are in each age group?
        - Which region has the highest average balance?
        - Show me the top 10 customers by balance
        - What's the distribution of job classifications?
        """)
    
    # Get API key from secrets
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("❌ Google API Key not found in Streamlit secrets. Please add GOOGLE_API_KEY to your secrets.toml file.")
        st.info("Create a `.streamlit/secrets.toml` file with: GOOGLE_API_KEY = 'your_api_key_here'")
        return
    
    if uploaded_file is None:
        st.info("📁 Please upload a CSV file to start chatting with your data.")
        return
    
    # Load and process data
    if st.session_state.df is None or uploaded_file.name not in str(st.session_state.get('uploaded_file_name', '')):
        with st.spinner("🔄 Loading and processing data..."):
            st.session_state.df = load_and_process_data(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            if st.session_state.df is not None:
                st.success(f"✅ Data loaded successfully! Shape: {st.session_state.df.shape}")
    
    if st.session_state.df is None:
        return
    
    # Show data preview if requested
    if 'show_data' in locals() and show_data:
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(st.session_state.df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Info:**")
                st.write(f"- Rows: {st.session_state.df.shape[0]:,}")
                st.write(f"- Columns: {st.session_state.df.shape[1]}")
            
            with col2:
                st.write("**Column Names:**")
                st.write(list(st.session_state.df.columns))
    
    # Setup agent if not ready
    if not st.session_state.agent_ready:
        with st.spinner("🤖 Setting up AI agent..."):
            agent, db_path = setup_database_and_agent(st.session_state.df, api_key)
            if agent:
                st.session_state.agent = agent
                st.session_state.db_path = db_path
                st.session_state.agent_ready = True
                st.success("🎉 AI agent is ready! You can now ask questions about your data.")
            else:
                st.error("❌ Failed to setup AI agent. Please check your API key and try again.")
                return
    
    # Chat interface
    st.markdown("---")
    st.subheader("💬 Chat with Your Data")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="chat-message bot-message"><strong>AI:</strong></div>', 
                           unsafe_allow_html=True)
                # Handle both old string responses and new formatted responses
                if isinstance(message["content"], dict):
                    display_formatted_response(message["content"])
                else:
                    st.markdown(message["content"])
    
    # Chat input
    question = st.chat_input("Ask a question about your data...")
    
    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message immediately
        with st.container():
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {question}</div>', 
                       unsafe_allow_html=True)
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = ask_question(st.session_state.agent, question)
        
        # Format the response
        formatted_response = parse_and_format_response(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        
        # Display AI response with proper formatting
        with st.container():
            st.markdown('<div class="chat-message bot-message"><strong>AI:</strong></div>', 
                       unsafe_allow_html=True)
            display_formatted_response(formatted_response)
        
        # Rerun to update the display
        st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()