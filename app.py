import streamlit as st
import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
import tempfile
import io
import sys

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
                st.markdown(f'<div class="chat-message bot-message"><strong>AI:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
    
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
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display AI response
        with st.container():
            st.markdown(f'<div class="chat-message bot-message"><strong>AI:</strong> {response}</div>', 
                       unsafe_allow_html=True)
        
        # Rerun to update the display
        st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()