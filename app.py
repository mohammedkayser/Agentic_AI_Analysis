import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import warnings
from dotenv import load_dotenv
import time
import traceback

# LangChain imports with proper error handling
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase
    from langchain.agents.agent_types import AgentType
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.stop()

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Agentic Data Analysis Chat", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e8f4f8;
        border-left-color: #1f77b4;
    }
    .assistant-message {
        background-color: #f0f2f6;
        border-left-color: #ff7f0e;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with proper defaults
def initialize_session_state():
    defaults = {
        'messages': [],
        'db_path': None,
        'df': None,
        'agent_executor': None,
        'data_uploaded': False,
        'processing': False,
        'last_error': None,
        'analysis_cache': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

class DataAnalysisApp:
    def __init__(self):
        self.llm = None
        self.max_retries = 3
        self.timeout_seconds = 30
        load_dotenv()
        self.setup_api()

    def setup_api(self):
        """Setup Gemini API with error handling"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
                st.info("💡 You can get your API key from: https://makersuite.google.com/app/apikey")
                st.stop()
                
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Using more stable model
                google_api_key=api_key,
                temperature=0.1,  # Slightly higher for better responses
                max_tokens=2048,  # Limit token count
                timeout=self.timeout_seconds
            )
            
            # Test API connection
            test_response = self.llm.invoke("Hello")
            if not test_response:
                raise Exception("API test failed")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            st.info("Please check your API key and internet connection")
            st.stop()

    def preprocess_data(self, df):
        """Clean and preprocess the uploaded data"""
        try:
            # Clean column names
            df.columns = [str(col).strip().replace(" ", "_").replace("-", "_").lower() for col in df.columns]
            
            # Remove special characters from column names
            df.columns = [''.join(c for c in col if c.isalnum() or c == '_') for col in df.columns]
            
            # Handle date columns
            date_keywords = ['date', 'time', 'created', 'updated', 'joined']
            for col in df.columns:
                if any(keyword in col.lower() for keyword in date_keywords):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except:
                        continue
            
            # Create age groups if age column exists
            if 'age' in df.columns and df['age'].dtype in ['int64', 'float64']:
                try:
                    df['age_group'] = pd.cut(
                        df['age'], 
                        bins=[0, 25, 35, 50, 65, 100], 
                        labels=['18-25', '26-35', '36-50', '51-65', '65+'],
                        include_lowest=True
                    )
                except:
                    pass
            
            # Handle missing values
            df = df.fillna('NULL')  # Replace NaN with 'NULL' for SQL compatibility
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error preprocessing data: {str(e)}")
            return df

    def create_database(self, df):
        """Create SQLite database with error handling"""
        try:
            # Create temporary database
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            db_path = temp_db.name
            temp_db.close()
            
            # Connect and create table
            conn = sqlite3.connect(db_path, timeout=10)
            
            # Ensure proper data types for SQLite
            df_copy = df.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].astype(str)
            
            df_copy.to_sql("data_table", conn, if_exists="replace", index=False, method='multi')
            conn.commit()
            conn.close()
            
            return db_path
            
        except Exception as e:
            st.error(f"❌ Error creating database: {str(e)}")
            return None

    def create_agent(self, db_path):
        """Create SQL agent with optimized settings"""
        try:
            # Create database connection
            db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            
            # Test database connection
            try:
                schema_info = db.get_table_info()
                if not schema_info:
                    raise Exception("Empty schema")
            except Exception as e:
                st.error(f"❌ Database connection failed: {str(e)}")
                return None, None
            
            # Create optimized system prompt
            system_prompt = f"""
You are an expert data analyst. Follow these rules strictly:

1. **BE CONCISE**: Keep responses under 500 words
2. **USE LIMITS**: Always add LIMIT 20 to SELECT queries unless user asks for more
3. **FORMAT TABLES**: Use markdown table format with | separators
4. **HANDLE ERRORS**: If a query fails, try a simpler version
5. **BE SPECIFIC**: Focus on the exact question asked

Database Schema:
{schema_info}

Available table: data_table

Example response format:
| Column | Value | Count |
|--------|-------|-------|
| Item1  | 100   | 25%   |

Always provide brief insights after showing data.
"""
            
            # Create toolkit
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            
            # Create agent with conservative settings
            agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,  # Reduce noise
                handle_parsing_errors=True,
                max_iterations=3,  # Reduced iterations
                max_execution_time=25,  # Shorter timeout
                early_stopping_method="generate",
                prefix=system_prompt
            )
            
            return agent, db
            
        except Exception as e:
            st.error(f"❌ Error creating agent: {str(e)}")
            return None, None

    def direct_analysis(self, question, df):
        """Provide direct pandas analysis for common questions"""
        try:
            question_lower = question.lower()
            
            # Pattern recognition for common questions
            if any(word in question_lower for word in ['pattern', 'insight', 'interesting', 'summary', 'overview']):
                return self.generate_data_overview(df)
            
            elif any(word in question_lower for word in ['column', 'field', 'variable']):
                return self.get_column_info(df)
                
            elif any(word in question_lower for word in ['statistic', 'stat', 'describe']):
                return self.get_statistics(df)
                
            elif any(word in question_lower for word in ['missing', 'null', 'empty']):
                return self.get_missing_data_info(df)
                
            return None
            
        except Exception as e:
            return f"❌ Error in direct analysis: {str(e)}"

    def generate_data_overview(self, df):
        """Generate comprehensive data overview"""
        try:
            num_rows, num_cols = df.shape
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            overview = f"""## 📊 Dataset Overview

**Basic Information:**
- **Total Records:** {num_rows:,}
- **Total Columns:** {num_cols}
- **Numeric Columns:** {len(numeric_cols)}
- **Categorical Columns:** {len(categorical_cols)}

"""
            
            # Top patterns in categorical data
            if categorical_cols:
                overview += "## 🔍 Key Patterns\n\n"
                for col in categorical_cols[:3]:  # Limit to first 3 columns
                    try:
                        value_counts = df[col].value_counts().head(5)
                        if len(value_counts) > 0:
                            overview += f"### {col.replace('_', ' ').title()}\n"
                            overview += "| Category | Count | Percentage |\n"
                            overview += "|----------|-------|------------|\n"
                            total = len(df)
                            for val, count in value_counts.items():
                                pct = (count/total)*100
                                overview += f"| {str(val)[:20]} | {count:,} | {pct:.1f}% |\n"
                            overview += "\n"
                    except:
                        continue
            
            # Numeric insights
            if numeric_cols:
                overview += "## 📈 Numeric Summary\n\n"
                try:
                    desc = df[numeric_cols].describe()
                    overview += "| Statistic | " + " | ".join([col[:15] for col in numeric_cols[:4]]) + " |\n"
                    overview += "|-----------|" + "|".join(["-" * 15] * min(len(numeric_cols), 4)) + "|\n"
                    
                    for stat in ['mean', 'std', 'min', 'max']:
                        if stat in desc.index:
                            values = []
                            for col in numeric_cols[:4]:
                                try:
                                    val = desc.loc[stat, col]
                                    values.append(f"{val:.2f}" if pd.notnull(val) else "N/A")
                                except:
                                    values.append("N/A")
                            overview += f"| {stat.title()} | " + " | ".join(values) + " |\n"
                except Exception as e:
                    overview += f"Error generating numeric summary: {str(e)}\n"
            
            overview += "\n## 💡 **Key Insights:**\n"
            overview += f"- Dataset has **{num_rows:,} records** across **{num_cols} variables**\n"
            overview += f"- **{len(categorical_cols)} categorical** and **{len(numeric_cols)} numeric** fields\n"
            
            if categorical_cols:
                top_cat = df[categorical_cols[0]].value_counts().index[0] if len(df[categorical_cols[0]].value_counts()) > 0 else "N/A"
                overview += f"- Most common category in '{categorical_cols[0]}': **{top_cat}**\n"
            
            return overview
            
        except Exception as e:
            return f"❌ Error generating overview: {str(e)}"

    def get_column_info(self, df):
        """Get detailed column information"""
        try:
            info = "## 📋 Column Information\n\n"
            info += "| Column | Type | Non-Null Count | Sample Values |\n"
            info += "|--------|------|----------------|---------------|\n"
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                
                # Get sample values (first few unique values)
                try:
                    sample_vals = df[col].dropna().unique()[:3]
                    sample_str = ", ".join([str(val)[:20] for val in sample_vals])
                except:
                    sample_str = "N/A"
                
                info += f"| {col} | {dtype} | {non_null:,} | {sample_str} |\n"
            
            return info
            
        except Exception as e:
            return f"❌ Error getting column info: {str(e)}"

    def get_statistics(self, df):
        """Get statistical summary"""
        try:
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                return "No numeric columns found for statistical analysis."
            
            stats = "## 📈 Statistical Summary\n\n"
            desc = numeric_df.describe()
            
            # Convert to markdown table
            stats += desc.round(2).to_markdown()
            
            return stats
            
        except Exception as e:
            return f"❌ Error generating statistics: {str(e)}"

    def get_missing_data_info(self, df):
        """Get missing data information"""
        try:
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            
            info = "## 🔍 Missing Data Analysis\n\n"
            info += "| Column | Missing Count | Missing % |\n"
            info += "|--------|---------------|----------|\n"
            
            for col in df.columns:
                miss_count = missing[col]
                miss_pct = missing_pct[col]
                info += f"| {col} | {miss_count} | {miss_pct:.1f}% |\n"
            
            total_missing = missing.sum()
            info += f"\n**Total missing values:** {total_missing:,}\n"
            
            return info
            
        except Exception as e:
            return f"❌ Error analyzing missing data: {str(e)}"

    def ask_question_with_retry(self, question, agent_executor, df):
        """Ask question with retry logic and fallbacks"""
        
        # Check cache first
        cache_key = hash(question)
        if cache_key in st.session_state.analysis_cache:
            return st.session_state.analysis_cache[cache_key]
        
        # Try direct analysis first for common questions
        direct_result = self.direct_analysis(question, df)
        if direct_result:
            st.session_state.analysis_cache[cache_key] = direct_result
            return direct_result
        
        # Try agent with retries
        for attempt in range(self.max_retries):
            try:
                with st.spinner(f"🤖 Analyzing... (Attempt {attempt + 1}/{self.max_retries})"):
                    
                    # Enhanced question with explicit instructions
                    enhanced_question = f"""
{question}

IMPORTANT: 
- Use LIMIT 20 in SQL queries
- Format results as markdown tables
- Keep response under 300 words
- If query fails, try a simpler version
                    """
                    
                    start_time = time.time()
                    response = agent_executor.run(enhanced_question)
                    elapsed_time = time.time() - start_time
                    
                    if response and len(response.strip()) > 0:
                        # Add timing info for slow queries
                        if elapsed_time > 15:
                            response += f"\n\n*Query took {elapsed_time:.1f} seconds*"
                        
                        # Cache successful result
                        st.session_state.analysis_cache[cache_key] = response
                        return response
                    
            except Exception as e:
                error_msg = str(e)
                st.warning(f"⚠️ Attempt {attempt + 1} failed: {error_msg[:100]}...")
                
                if attempt == self.max_retries - 1:
                    # Final fallback to pandas analysis
                    fallback_result = self.fallback_analysis(question, df)
                    st.session_state.analysis_cache[cache_key] = fallback_result
                    return fallback_result
                
                time.sleep(2)  # Brief pause between retries
        
        return "❌ Unable to process the question after multiple attempts. Please try a simpler query."

    def fallback_analysis(self, question, df):
        """Fallback pandas analysis when agent fails"""
        try:
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['top', 'highest', 'maximum']):
                # Find numeric columns and show top values
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    top_values = df.nlargest(10, col)
                    result = f"## Top 10 Records by {col}\n\n"
                    result += top_values.head(10).to_markdown(index=False)
                    return result
            
            elif any(word in question_lower for word in ['count', 'frequency']):
                # Show value counts for categorical columns
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    col = cat_cols[0]
                    counts = df[col].value_counts().head(10)
                    result = f"## Value Counts for {col}\n\n"
                    result += "| Value | Count |\n|-------|-------|\n"
                    for val, count in counts.items():
                        result += f"| {val} | {count} |\n"
                    return result
            
            # Default fallback
            return self.generate_data_overview(df)
            
        except Exception as e:
            return f"❌ Fallback analysis failed: {str(e)}"

def main():
    # Initialize session state
    initialize_session_state()
    
    # Create app instance
    try:
        app = DataAnalysisApp()
    except Exception as e:
        st.error(f"❌ Failed to initialize app: {str(e)}")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic Data Analysis Chat</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📊 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to start analyzing your data"
        )
        
        if uploaded_file is not None:
            try:
                # Load data with error handling
                try:
                    df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
                    st.stop()
                
                if df.empty:
                    st.error("❌ The uploaded file is empty")
                    st.stop()
                
                # Preprocess data
                with st.spinner("Processing data..."):
                    df = app.preprocess_data(df)
                    db_path = app.create_database(df)
                    
                    if db_path:
                        agent_executor, db = app.create_agent(db_path)
                        
                        if agent_executor:
                            # Update session state
                            st.session_state.update({
                                'df': df,
                                'db_path': db_path,
                                'agent_executor': agent_executor,
                                'data_uploaded': True,
                                'last_error': None
                            })
                            
                            st.success("✅ Data uploaded and processed successfully!")
                        else:
                            st.error("❌ Failed to create analysis agent")
                    else:
                        st.error("❌ Failed to create database")
                
                # Show data info
                if st.session_state.data_uploaded:
                    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
                    st.write("**Dataset Info:**")
                    st.write(f"- **Rows:** {df.shape[0]:,}")
                    st.write(f"- **Columns:** {df.shape[1]}")
                    st.write(f"- **Size:** {df.memory_usage().sum() / 1024:.1f} KB")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Expandable sections
                    with st.expander("📋 View Columns"):
                        st.write(list(df.columns))
                    
                    with st.expander("👀 Sample Data"):
                        st.dataframe(df.head())
                    
                    with st.expander("📈 Quick Stats"):
                        st.write(df.describe())
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state.data_uploaded = False
        
        # Sample questions
        if st.session_state.data_uploaded:
            st.markdown("---")
            st.header("💡 Quick Analysis")
            
            sample_questions = [
                "Show me data overview and patterns",
                "What columns do I have?",
                "Give me basic statistics",
                "Show missing data information",
                "What are the top categories?",
                "Display summary statistics table"
            ]
            
            for question in sample_questions:
                if st.button(question, key=f"sample_{question}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    # Process immediately
                    with st.spinner("Processing..."):
                        response = app.ask_question_with_retry(
                            question, 
                            st.session_state.agent_executor, 
                            st.session_state.df
                        )
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

    # Main chat interface
    if not st.session_state.data_uploaded:
        st.info("👆 Please upload a CSV file in the sidebar to start analyzing")
        
        # Instructions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📤 Step 1: Upload
            - Click "Choose a CSV file" in sidebar
            - Wait for processing to complete
            - Check for any error messages
            """)
        
        with col2:
            st.markdown("""
            ### 💬 Step 2: Ask Questions
            - Use natural language queries
            - Try the sample questions
            - Ask about patterns, statistics, insights
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Step 3: Analyze
            - Get AI-powered insights
            - View formatted tables and charts
            - Ask follow-up questions
            """)
    else:
        # Display conversation
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong></div>',
                    unsafe_allow_html=True
                )
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your data..."):
            if not st.session_state.processing:
                st.session_state.processing = True
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get response
                try:
                    response = app.ask_question_with_retry(
                        prompt, 
                        st.session_state.agent_executor, 
                        st.session_state.df
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_response = f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
                
                st.session_state.processing = False
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.analysis_cache = {}  # Clear cache too
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9em;">'
        'Built with ❤️ using Streamlit & LangChain | Powered by Google Gemini'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()