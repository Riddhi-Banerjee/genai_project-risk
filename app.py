import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- 1. UI SETUP ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide")
st.title("🛡️ Agentic Risk Intelligence Engine")

# --- 2. API KEY HANDLING ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("🔑 Please provide a Google API Key in the sidebar or app secrets.")
    st.stop()

# LiteLLM/CrewAI uses this specific environment variable for Gemini
os.environ["GEMINI_API_KEY"] = api_key 

# --- 3. CUSTOM DATA TOOL ---
class RiskDataTool(BaseTool):
    name: str = "Risk Data Reader"
    description: str = "Reads project, financial, or market data from local CSV files."

    def _run(self, file_name: str) -> str:
        try:
            # Note: Ensure these CSV files are in your GitHub root folder
            df = pd.read_csv(file_name)
            return df.head(30).to_string()
        except Exception as e:
            return f"Error reading {file_name}: {e}"

csv_tool = RiskDataTool()

# --- 4. AGENT DEFINITIONS ---
# Format: "provider/model-name"
# Using 'gemini/' prefix as required by LiteLLM
gemini_model = "gemini/gemini-1.5-flash"

market_agent = Agent(
    role='Market Analysis Agent',
    goal='Identify external economic risks from market_trends.csv',
    backstory='Expert economist monitoring inflation and interest rates.',
    tools=[csv_tool],
    llm=gemini_model,
    verbose=True
)

scoring_agent = Agent(
    role='Risk Scoring Agent',
    goal='Identify financial risks and overdue payments from transaction.csv',
    backstory='Forensic auditor focusing on budget leaks.',
    tools=[csv_tool],
    llm=gemini_model,
    verbose=True
)

manager = Agent(
    role='Project Risk Manager',
    goal='Synthesize reports into a final executive mitigation strategy.',
    backstory='Chief Risk Officer.',
    llm=gemini_model,
    verbose=True
)

# --- 5. EXECUTION ---
target_project = st.text_input("Enter Project ID:", "PROJ_0001")

if st.button("🚀 Run Agentic Audit"):
    t1 = Task(description=f"Analyze market context for {target_project} in market_trends.csv", agent=market_agent, expected_output="Economic risk summary.")
    t2 = Task(description=f"Analyze financial history for {target_project} in transaction.csv", agent=scoring_agent, expected_output="Financial risk report.")
    t3 = Task(description="Combine findings into a 3-step strategy.", agent=manager, expected_output="Executive Risk Strategy.")

    risk_crew = Crew(
        agents=[market_agent, scoring_agent, manager],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    with st.status("Agents are collaborating...", expanded=True) as status:
        result = risk_crew.kickoff()
        status.update(label="Analysis Complete", state="complete")
    
    st.markdown("### 📊 Final Intelligence Report")
    st.markdown(result.raw)
