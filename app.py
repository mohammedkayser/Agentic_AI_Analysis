# Full app.py with OutputFixingParser to handle malformed markdown tables

import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import warnings
from dotenv import load_dotenv
import io

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.output_parsers import OutputFixingParser

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Agentic Data Analysis Chat", page_icon="🤖", layout="wide")

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
</style>
""", unsafe_allow_html=True)

for key in ["messages", "db_path", "df", "agent_executor", "data_uploaded", "run_sample"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else False if key in ["data_uploaded", "run_sample"] else None

class DataAnalysisApp:
    def __init__(self):
        load_dotenv()
        self.setup_api()

    def setup_api(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
            st.stop()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )

    def preprocess_data(self, df):
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
        date_columns = ['date_joined', 'date', 'created_date', 'timestamp']
        for col in df.columns:
            if any(word in col.lower() for word in date_columns):
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100],
                                     labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        return df

    def create_database(self, df):
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = temp_db.name
        df.to_sql("data_table", sqlite3.connect(db_path), if_exists="replace", index=False)
        return db_path

    def create_agent(self, db_path):
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        schema = db.get_table_info()
        prompt = f"""
        You are a helpful data analyst AI. Format structured data as markdown tables using |.
        Do NOT use triple backticks (```) around tables.
        Always respond inline with results. Do not say "see table above".
        Limit results to 30 rows max. Provide interpretation after the table.

        Database schema:
        {schema}
        """
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        return create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            output_parser=OutputFixingParser.from_llm(self.llm),
            max_iterations=20,
            max_execution_time=60,
            early_stopping_method="force",
            prefix=prompt
        ), db

    def enhance_question(self, question):
        return (
            question.strip()
            + "\n\nPlease format your answer as a markdown table (without code block)."
            + " Limit rows to 30 max. Do not refer to tables above. Provide insights inline."
        )

    def format_response(self, response):
        try:
            lines = response.strip().split("\n")
            table_lines = [l for l in lines if "|" in l and len(l.strip()) > 5]
            if len(table_lines) >= 2 and any("---" in l for l in table_lines):
                start = lines.index(next(l for l in table_lines if "---" in l)) - 1
                end = start + 1 + sum(1 for l in table_lines if l != table_lines[0])
                table_text = "\n".join(lines[start:end+1])
                df = pd.read_csv(io.StringIO(table_text), sep="|", engine="python")
                df = df.dropna(axis=1, how='all')
                if len(df) > 30:
                    df = df.head(30)
                explanation = response.replace(table_text, '').strip()
                return {"explanation": explanation + "\n\n*Only first 30 rows shown*", "table": df}
        except Exception as e:
            st.warning("⚠️ Unable to parse markdown table. Showing raw text.")
        return {"explanation": response, "table": None}

    def ask_question(self, question, agent_executor):
        with st.spinner("🤖 Analyzing your question..."):
            enhanced = self.enhance_question(question)
            response = agent_executor.run(enhanced)
            return self.format_response(response)


# The rest of the code for main() remains unchanged

def main():
    app = DataAnalysisApp()
    st.markdown('<h1 class="main-header">🤖 Agentic Data Analysis Chat</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📊 Upload CSV")
        file = st.file_uploader("Upload a CSV", type=['csv'])
        if file:
            df = pd.read_csv(file)
            df = app.preprocess_data(df)
            db_path = app.create_database(df)
            agent_executor, _ = app.create_agent(db_path)
            st.session_state.update({
                'df': df,
                'db_path': db_path,
                'agent_executor': agent_executor,
                'data_uploaded': True
            })
            st.success("✅ Data uploaded!")
            st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
            st.write("**Dataset Info:**")
            st.write(f"- Rows: {df.shape[0]:,}")
            st.write(f"- Columns: {df.shape[1]}")
            st.write(f"- Memory: {df.memory_usage().sum() / 1024:.1f} KB")
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("📋 Columns"):
                st.write(list(df.columns))
            with st.expander("👀 Sample Data"):
                st.dataframe(df.head())
            with st.expander("📈 Data Summary"):
                st.write(df.describe())

        if st.session_state.data_uploaded:
            st.header("💡 Sample Questions")
            sample_qs = [
                "What are the most interesting patterns in this dataset?",
                "Show me summary statistics in table format",
                "Which job classifications are most common?",
                "Break down average balance by age group",
                "Create a table of top regions by customer count",
                "What insights can we learn from this data?"
            ]
            for q in sample_qs:
                if st.button(q, key=q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.session_state.run_sample = True
                    st.rerun()

    if st.session_state.data_uploaded:
        if st.session_state.run_sample:
            last_q = st.session_state.messages[-1]["content"]
            response = app.ask_question(last_q, st.session_state.agent_executor)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.run_sample = False
            st.rerun()

        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='chat-message assistant-message'><strong>🤖 Assistant:</strong></div>", unsafe_allow_html=True)
                content = msg['content']
                if isinstance(content, dict):
                    if content['explanation']:
                        st.markdown(content['explanation'])
                    if content['table'] is not None:
                        st.dataframe(content['table'])
                else:
                    st.markdown(content)

        if prompt := st.chat_input("Ask me anything about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = app.ask_question(prompt, st.session_state.agent_executor)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("👆 Upload a CSV to start analyzing.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>Built with ❤️ using Streamlit + LangChain | Powered by Gemini</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
