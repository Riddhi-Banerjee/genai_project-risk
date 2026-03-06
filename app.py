import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process, LLM # Added LLM here
from crewai.tools import BaseTool
from pydantic import Field

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide")
st.title("🛡️ Agentic Risk Intelligence Engine")

# --- 2. AUTHENTICATION ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("🔑 Please provide a Google API Key.")
    st.stop()

# Force-set the keys for all underlying libraries
os.environ["GEMINI_API_KEY"] = api_key
os.environ["GOOGLE_API_KEY"] = api_key

# --- 3. MODEL CONFIGURATION (The Critical Part) ---
# We define a dedicated LLM object. 
# This bypasses the default string-to-API mapping that is causing your 404.
gemini_llm = LLM(
    model="gemini/gemini-1.5-flash", 
    api_key=api_key,
    temperature=0.7
)

# --- 4. CUSTOM DATA TOOL ---
class RiskDataTool(BaseTool):
    name: str = "Risk Data Reader"
    description: str = "Reads project, financial, or market data from local CSV files."

    def _run(self, file_name: str) -> str:
        try:
            # Files must be in the same root directory as app.py
            df = pd.read_csv(file_name)
            return df.head(25).to_string() # Head(25) to save tokens
        except Exception as e:
            return f"Error reading {file_name}: {e}"

csv_tool = RiskDataTool()

# --- 5. AGENT DEFINITIONS ---
market_agent = Agent(
    role='Market Analysis Agent',
    goal='Identify external economic risks from market_trends.csv',
    backstory='Expert economist monitoring inflation.',
    tools=[csv_tool],
    llm=gemini_llm, # Using the LLM object here
    verbose=True
)

scoring_agent = Agent(
    role='Risk Scoring Agent',
    goal='Identify financial risks and overdue payments from transaction.csv',
    backstory='Forensic auditor focusing on budget health.',
    tools=[csv_tool],
    llm=gemini_llm, # Using the LLM object here
    verbose=True
)

manager = Agent(
    role='Project Risk Manager',
    goal='Synthesize reports into a final executive strategy.',
    backstory='Chief Risk Officer.',
    llm=gemini_llm, # Using the LLM object here
    verbose=True
)

# --- 6. WORKFLOW ---
target_project = st.text_input("Enter Project ID:", "PROJ_0001")

if st.button("🚀 Run Agentic Audit"):
    t1 = Task(description=f"Analyze market context for {target_project}", agent=market_agent, expected_output="Market risk summary.")
    t2 = Task(description=f"Analyze financial history for {target_project}", agent=scoring_agent, expected_output="Financial report.")
    t3 = Task(description="Combine reports into a mitigation strategy.", agent=manager, expected_output="Final Strategy.")

    risk_crew = Crew(
        agents=[market_agent, scoring_agent, manager],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    with st.status("Agents are collaborating...", expanded=True) as status:
        try:
            result = risk_crew.kickoff()
            status.update(label="Analysis Complete", state="complete")
            st.markdown(result.raw)
        except Exception as e:
            st.error(f"Execution Error: {e}")
