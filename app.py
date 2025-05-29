import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import warnings
from dotenv import load_dotenv
import time
import traceback
import threading

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
        'analysis_cache': {},
        'db_connection': None,
        'sql_database': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

class DatabaseManager:
    """Separate class to handle database connections and operations"""
    
    def __init__(self):
        self.db_lock = threading.Lock()
    
    def create_database(self, df):
        """Create SQLite database with improved error handling and connection management"""
        try:
            # Create temporary database file
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            db_path = temp_db.name
            temp_db.close()
            
            # Create connection with optimized settings
            conn = sqlite3.connect(
                db_path, 
                timeout=30,  # Increased timeout
                check_same_thread=False,  # Allow multi-threading
                isolation_level=None  # Autocommit mode
            )
            
            # Set pragmas for better performance and reliability
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance between performance and safety
            conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
            conn.execute("PRAGMA cache_size=10000")  # Increase cache size
            
            # Prepare dataframe for SQLite
            df_copy = df.copy()
            
            # Clean and standardize data types
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    # Convert object columns to string and handle nulls
                    df_copy[col] = df_copy[col].astype(str).replace('nan', 'NULL')
                elif df_copy[col].dtype in ['float64', 'int64']:
                    # Handle numeric nulls
                    df_copy[col] = df_copy[col].fillna(0)
            
            # Insert data in chunks for better memory management
            chunk_size = 1000
            try:
                df_copy.to_sql(
                    "data_table", 
                    conn, 
                    if_exists="replace", 
                    index=False, 
                    method='multi',
                    chunksize=chunk_size
                )
                
                # Verify table creation
                cursor = conn.execute("SELECT COUNT(*) FROM data_table")
                row_count = cursor.fetchone()[0]
                
                if row_count != len(df):
                    raise Exception(f"Row count mismatch: expected {len(df)}, got {row_count}")
                
                # Create indexes for better query performance
                self._create_indexes(conn, df_copy.columns)
                
                conn.close()
                st.success(f"✅ Database created successfully with {row_count:,} records")
                return db_path
                
            except Exception as e:
                conn.close()
                os.unlink(db_path)  # Clean up on failure
                raise e
                
        except Exception as e:
            st.error(f"❌ Error creating database: {str(e)}")
            return None
    
    def _create_indexes(self, conn, columns):
        """Create indexes on commonly queried columns"""
        try:
            # Create indexes on likely key columns
            index_keywords = ['id', 'name', 'date', 'time', 'gender', 'age', 'job', 'classification']
            
            for col in columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in index_keywords):
                    try:
                        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON data_table({col})")
                    except:
                        pass  # Skip if index creation fails
        except Exception as e:
            st.warning(f"⚠️ Warning: Could not create indexes: {str(e)}")

    def test_connection(self, db_path):
        """Test database connection and return connection info"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(
                    db_path, 
                    timeout=10,
                    check_same_thread=False
                )
                
                # Test basic operations
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                if not tables:
                    raise Exception("No tables found in database")
                
                # Get table info
                cursor = conn.execute("PRAGMA table_info(data_table)")
                columns = cursor.fetchall()
                
                # Test sample query
                cursor = conn.execute("SELECT COUNT(*) FROM data_table LIMIT 1")
                count = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'status': 'success',
                    'tables': len(tables),
                    'columns': len(columns),
                    'rows': count
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

class DataAnalysisApp:
    def __init__(self):
        self.llm = None
        self.max_retries = 3
        self.timeout_seconds = 30
        self.db_manager = DatabaseManager()
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
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1,
                max_tokens=2048,
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
        """Clean and preprocess the uploaded data with fixed categorical handling"""
        try:
            # Clean column names
            original_columns = df.columns.tolist()
            df.columns = [str(col).strip().replace(" ", "_").replace("-", "_").lower() for col in df.columns]
            
            # Remove special characters from column names
            df.columns = [''.join(c for c in col if c.isalnum() or c == '_') for col in df.columns]
            
            # Handle duplicate column names
            seen_cols = set()
            new_cols = []
            for col in df.columns:
                if col in seen_cols:
                    counter = 1
                    new_col = f"{col}_{counter}"
                    while new_col in seen_cols:
                        counter += 1
                        new_col = f"{col}_{counter}"
                    new_cols.append(new_col)
                    seen_cols.add(new_col)
                else:
                    new_cols.append(col)
                    seen_cols.add(col)
            
            df.columns = new_cols
            
            # Handle missing values BEFORE creating categorical columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('NULL')
                elif df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(0)
            
            # Handle date columns
            date_keywords = ['date', 'time', 'created', 'updated', 'joined']
            for col in df.columns:
                if any(keyword in col.lower() for keyword in date_keywords):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except:
                        continue
            
            # Create age groups if age column exists (AFTER handling missing values)
            age_col = None
            for col in df.columns:
                if 'age' in col.lower():
                    age_col = col
                    break
            
            if age_col and df[age_col].dtype in ['int64', 'float64']:
                try:
                    # Ensure no missing values in age column before creating categories
                    df[age_col] = df[age_col].fillna(0)  # Fill any remaining NaN values
                    
                    # Create age groups with proper handling of edge cases
                    df['age_group'] = pd.cut(
                        df[age_col], 
                        bins=[0, 25, 35, 50, 65, 100], 
                        labels=['18-25', '26-35', '36-50', '51-65', '65+'],
                        include_lowest=True,
                        right=True  # Include right boundary
                    )
                    
                    # Handle any remaining NaN in age_group (for ages outside bins)
                    df['age_group'] = df['age_group'].cat.add_categories(['Unknown'])
                    df['age_group'] = df['age_group'].fillna('Unknown')
                    
                except Exception as e:
                    st.warning(f"Could not create age groups: {str(e)}")
                    pass
            
            # Final pass to handle any remaining categorical columns with missing values
            for col in df.columns:
                if pd.api.types.is_categorical_dtype(df[col]):
                    # For categorical columns, add 'NULL' as a category if needed
                    if df[col].isnull().any():
                        if 'NULL' not in df[col].cat.categories:
                            df[col] = df[col].cat.add_categories(['NULL'])
                        df[col] = df[col].fillna('NULL')
            
            # Log column mapping for debugging
            if len(original_columns) != len(df.columns):
                st.info(f"Column names were cleaned. Original: {len(original_columns)}, New: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error preprocessing data: {str(e)}")
            # Return original dataframe with minimal processing if preprocessing fails
            try:
                # Just handle missing values without complex transformations
                df_simple = df.copy()
                for col in df_simple.columns:
                    if df_simple[col].dtype == 'object':
                        df_simple[col] = df_simple[col].fillna('NULL')
                    else:
                        df_simple[col] = df_simple[col].fillna(0)
                return df_simple
            except:
                return df

    def create_agent(self, db_path):
        """Create SQL agent with improved connection handling"""
        try:
            # Test database connection first
            connection_test = self.db_manager.test_connection(db_path)
            if connection_test['status'] != 'success':
                st.error(f"❌ Database connection test failed: {connection_test.get('error', 'Unknown error')}")
                return None, None
            
            st.success(f"✅ Database connection verified: {connection_test['rows']:,} rows, {connection_test['columns']} columns")
            
            # Create database connection with retry logic
            max_connection_retries = 3
            db = None
            
            for attempt in range(max_connection_retries):
                try:
                    # Create SQLDatabase with proper URI
                    db_uri = f"sqlite:///{db_path}"
                    db = SQLDatabase.from_uri(
                        db_uri,
                        sample_rows_in_table_info=3,  # Limit sample rows for performance
                        include_tables=['data_table']  # Explicitly specify table
                    )
                    
                    # Test the connection
                    schema_info = db.get_table_info()
                    if not schema_info:
                        raise Exception("Empty schema information")
                    
                    # Test a simple query
                    test_query = "SELECT COUNT(*) FROM data_table LIMIT 1"
                    result = db.run(test_query)
                    
                    st.success(f"✅ SQLDatabase connection established successfully")
                    break
                    
                except Exception as e:
                    st.warning(f"⚠️ Database connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_connection_retries - 1:
                        st.error("❌ Failed to establish database connection after multiple attempts")
                        return None, None
                    time.sleep(1)
            
            # Create enhanced system prompt
            schema_info = db.get_table_info()
            system_prompt = f"""
You are an expert data analyst. Follow these rules strictly:

1. **BE CONCISE**: Keep responses under 500 words
2. **USE LIMITS**: Always add LIMIT 20 to SELECT queries unless user asks for more
3. **FORMAT TABLES**: Use markdown table format with | separators
4. **HANDLE ERRORS**: If a query fails, try a simpler version
5. **BE SPECIFIC**: Focus on the exact question asked
6. **GROUP BY RULES**: When using GROUP BY, include all non-aggregate columns in GROUP BY clause

Database Schema:
{schema_info}

Available table: data_table

Example queries:
- Simple count: SELECT column, COUNT(*) FROM data_table GROUP BY column LIMIT 10
- Average by group: SELECT group_col, AVG(numeric_col) FROM data_table GROUP BY group_col LIMIT 10
- Multiple groups: SELECT col1, col2, AVG(value) FROM data_table GROUP BY col1, col2 LIMIT 10

Always provide brief insights after showing data.
"""
            
            # Create toolkit with error handling
            try:
                toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            except Exception as e:
                st.error(f"❌ Error creating SQL toolkit: {str(e)}")
                return None, None
            
            # Create agent with conservative settings
            try:
                agent = create_sql_agent(
                    llm=self.llm,
                    toolkit=toolkit,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                    max_iterations=5,  # Slightly increased for complex queries
                    max_execution_time=30,
                    early_stopping_method="generate",
                    prefix=system_prompt,
                    return_intermediate_steps=False  # Reduce noise
                )
                
                st.success("✅ SQL Agent created successfully")
                return agent, db
                
            except Exception as e:
                st.error(f"❌ Error creating SQL agent: {str(e)}")
                return None, None
                
        except Exception as e:
            st.error(f"❌ Error in create_agent: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
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
        """Ask question with retry logic and improved error handling"""
        
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

IMPORTANT INSTRUCTIONS:
- Use LIMIT 20 in SQL queries unless specifically asked for more
- Format results as clean markdown tables
- Keep response under 400 words
- If a query fails, try a simpler version
- When using GROUP BY, include ALL non-aggregate columns in the GROUP BY clause
- Use proper SQL syntax for SQLite
                    """
                    
                    start_time = time.time()
                    
                    # Execute with timeout protection
                    try:
                        response = agent_executor.run(enhanced_question)
                    except Exception as query_error:
                        # Try fallback for specific SQL errors
                        if "database" in str(query_error).lower() or "connection" in str(query_error).lower():
                            st.warning(f"⚠️ Database connection issue on attempt {attempt + 1}: {str(query_error)[:100]}...")
                            if attempt < self.max_retries - 1:
                                time.sleep(2)
                                continue
                        raise query_error
                    
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
                
                # Log detailed error for debugging
                if "database" in error_msg.lower():
                    st.error(f"Database error details: {error_msg}")
                
                if attempt == self.max_retries - 1:
                    # Final fallback to pandas analysis
                    fallback_result = self.fallback_analysis(question, df)
                    st.session_state.analysis_cache[cache_key] = fallback_result
                    return fallback_result
                
                time.sleep(2)  # Brief pause between retries
        
        return "❌ Unable to process the question after multiple attempts. Please try a simpler query or check the database connection."

    def fallback_analysis(self, question, df):
        """Fallback pandas analysis when agent fails"""
        try:
            question_lower = question.lower()
            
            # Handle balance/average questions
            if 'balance' in question_lower and 'average' in question_lower:
                balance_cols = [col for col in df.columns if 'balance' in col.lower()]
                if balance_cols:
                    result = "## Average Balance Analysis\n\n"
                    
                    # Group by available categorical columns
                    group_cols = []
                    for potential_col in ['gender', 'age_group', 'job', 'classification']:
                        matching_cols = [col for col in df.columns if potential_col in col.lower()]
                        if matching_cols:
                            group_cols.extend(matching_cols)
                    
                    if group_cols and balance_cols:
                        try:
                            grouped = df.groupby(group_cols[:3])[balance_cols[0]].mean().reset_index()
                            result += grouped.head(15).to_markdown(index=False, floatfmt=".2f")
                            return result
                        except:
                            pass
            
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

    # Main header
    st.markdown('<div class="main-header">🤖 Agentic Data Analysis Chat</div>', unsafe_allow_html=True)
    
    # Sidebar for file upload and information
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file to start analyzing"
        )
        
        # Process uploaded file
        if uploaded_file is not None and not st.session_state.data_uploaded:
            try:
                with st.spinner("📊 Processing your data..."):
                    # Read file based on extension
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Validate data
                    if df.empty:
                        st.error("❌ The uploaded file is empty")
                        st.stop()
                    
                    if len(df.columns) == 0:
                        st.error("❌ No columns found in the file")
                        st.stop()
                    
                    # Preprocess data
                    df_processed = app.preprocess_data(df)
                    
                    # Create database
                    db_path = app.db_manager.create_database(df_processed)
                    
                    if db_path:
                        # Create agent
                        agent_executor, sql_db = app.create_agent(db_path)
                        
                        if agent_executor:
                            # Store in session state
                            st.session_state.df = df_processed
                            st.session_state.db_path = db_path
                            st.session_state.agent_executor = agent_executor
                            st.session_state.sql_database = sql_db
                            st.session_state.data_uploaded = True
                            
                            st.success(f"✅ Data loaded successfully! {len(df_processed):,} rows, {len(df_processed.columns)} columns")
                            st.rerun()
                        else:
                            st.error("❌ Failed to create analysis agent")
                    else:
                        st.error("❌ Failed to create database")
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state.data_uploaded = False
        
        # Show data information if uploaded
        if st.session_state.data_uploaded and st.session_state.df is not None:
            st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
            st.markdown("### 📊 Dataset Info")
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df):,}")
                st.metric("Numeric Cols", len(df.select_dtypes(include=['number']).columns))
            
            with col2:
                st.metric("Columns", len(df.columns))
                st.metric("Text Cols", len(df.select_dtypes(include=['object']).columns))
            
            # Show column names
            with st.expander("📋 Column Names"):
                for i, col in enumerate(df.columns, 1):
                    st.write(f"{i}. {col}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset button
            if st.button("🔄 Upload New Data", type="secondary"):
                # Clean up
                if st.session_state.db_path and os.path.exists(st.session_state.db_path):
                    try:
                        os.unlink(st.session_state.db_path)
                    except:
                        pass
                
                # Reset session state
                for key in ['df', 'db_path', 'agent_executor', 'sql_database', 'data_uploaded', 'messages', 'analysis_cache']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.rerun()
        
        # Usage tips
        if not st.session_state.data_uploaded:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Upload CSV or Excel files
            - Ensure data has column headers
            - Files up to 200MB supported
            - Clean data works best
            """)
        
        else:
            st.markdown("### 💬 Sample Questions")
            sample_questions = [
                "Show me a summary of the data",
                "What are the column names and types?",
                "Show me the top 10 records",
                "What patterns do you see?",
                "Are there any missing values?",
                "Show statistics for numeric columns"
            ]
            
            for q in sample_questions:
                if st.button(q, key=f"sample_{hash(q)}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()

    # Main chat interface
    if st.session_state.data_uploaded:
        # Chat container
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>🧑‍💼 You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(message["content"])
        
        # Chat input
        st.markdown("---")
        
        # Question input
        question = st.chat_input("Ask a question about your data...")
        
        if question:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Process question
            try:
                with st.spinner("🤖 Analyzing your data..."):
                    if st.session_state.agent_executor and st.session_state.df is not None:
                        response = app.ask_question_with_retry(
                            question, 
                            st.session_state.agent_executor, 
                            st.session_state.df
                        )
                        
                        # Add assistant response
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("❌ Analysis agent not available. Please reload your data.")
                        
            except Exception as e:
                error_msg = f"❌ Error processing question: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
            
            st.rerun()
        
        # Clear chat button
        if st.session_state.messages:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.analysis_cache = {}
                    st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Agentic Data Analysis Chat!
        
        ### 🚀 Getting Started:
        1. **Upload your data** using the sidebar (CSV or Excel files)
        2. **Ask questions** in natural language about your data
        3. **Get insights** powered by AI analysis
        
        ### 🎯 What you can do:
        - Get data summaries and overviews
        - Analyze patterns and trends
        - Generate statistics and visualizations
        - Ask complex analytical questions
        - Explore relationships in your data
        
        ### 📝 Example Questions:
        - "What are the main patterns in this data?"
        - "Show me the average values by category"
        - "Which records have the highest values?"
        - "Are there any data quality issues?"
        - "What insights can you provide?"
        
        ### 🔧 Features:
        - **Natural Language Processing**: Ask questions in plain English
        - **SQL Agent**: Automated query generation and execution
        - **Smart Caching**: Faster responses for repeated questions
        - **Error Recovery**: Robust handling of complex queries
        - **Data Preprocessing**: Automatic data cleaning and optimization
        """)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧠 AI-Powered
            Advanced language models understand your questions and generate appropriate SQL queries automatically.
            """)
        
        with col2:
            st.markdown("""
            ### ⚡ Fast & Efficient
            Optimized database operations with caching for quick responses to your analytical queries.
            """)
        
        with col3:
            st.markdown("""
            ### 🛡️ Robust & Reliable
            Built-in error handling, retry logic, and fallback mechanisms ensure consistent performance.
            """)

# Cleanup function
def cleanup_resources():
    """Clean up temporary files and connections"""
    try:
        if st.session_state.get('db_path') and os.path.exists(st.session_state.db_path):
            os.unlink(st.session_state.db_path)
    except:
        pass

# Register cleanup
import atexit
atexit.register(cleanup_resources)

# Run the app
if __name__ == "__main__":
    main()