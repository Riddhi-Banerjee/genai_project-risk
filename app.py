import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- 1. UI SETUP ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide")
st.title("🛡️ Agentic Risk Intelligence Engine")

# --- 2. API KEY HANDLING ---
# This looks for 'GOOGLE_API_KEY' in your Streamlit Secrets or sidebar
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("🔑 Please provide a Google API Key in the sidebar or app secrets.")
    st.stop()

# IMPORTANT: CrewAI/LiteLLM looks specifically for 'GEMINI_API_KEY' for this model prefix
os.environ["GEMINI_API_KEY"] = api_key 

# --- 3. CUSTOM DATA TOOL ---
class RiskDataTool(BaseTool):
    name: str = "Risk Data Reader"
    description: str = "Reads and filters project, financial, or market data from local CSV files."

    def _run(self, file_name: str) -> str:
        try:
            # Load the file
            df = pd.read_csv(file_name)
            # Return first 50 rows to avoid token context limits
            return df.head(50).to_string()
        except Exception as e:
            return f"Error reading {file_name}: {e}"

# Initialize the tool
csv_tool = RiskDataTool()

# --- 4. AGENT DEFINITIONS ---
# Using the "gemini/gemini-1.5-flash" prefix for LiteLLM compatibility
model_name = "gemini/gemini-1.5-flash"

market_agent = Agent(
    role='Market Analysis Agent',
    goal='Identify external economic risks from market_trends.csv',
    backstory='You are a financial economist specializing in inflation and interest rates.',
    tools=[csv_tool],
    llm=model_name,
    verbose=True
)

scoring_agent = Agent(
    role='Risk Scoring Agent',
    goal='Identify financial risks and overdue payments from transaction.csv',
    backstory='You are a forensic auditor focusing on budget health and payment delays.',
    tools=[csv_tool],
    llm=model_name,
    verbose=True
)

status_agent = Agent(
    role='Project Status Tracking Agent',
    goal='Monitor project health from project_risk_raw_dataset.csv',
    backstory='You are a senior project auditor identifying internal delays and resource issues.',
    tools=[csv_tool],
    llm=model_name,
    verbose=True
)

manager = Agent(
    role='Project Risk Manager',
    goal='Synthesize all findings into a final executive mitigation strategy.',
    backstory='You are the Chief Risk Officer who summarizes technical reports into business strategy.',
    llm=model_name,
    verbose=True
)

# --- 5. EXECUTION UI ---
target_project = st.text_input("Enter Project ID to Analyze:", "PROJ_0001")

if st.button("🚀 Execute Agentic Audit"):
    # Define Tasks
    t1 = Task(
        description=f"Analyze market context for {target_project} using market_trends.csv", 
        agent=market_agent, 
        expected_output="A summary of external market risks."
    )
    t2 = Task(
        description=f"Analyze financial history for {target_project} using transaction.csv", 
        agent=scoring_agent, 
        expected_output="A report on financial exposure and payment status."
    )
    t3 = Task(
        description=f"Review project status and complexity for {target_project} using project_risk_raw_dataset.csv", 
        agent=status_agent, 
        expected_output="An internal project health report."
    )
    t4 = Task(
        description="Synthesize the findings from all agents into a 3-step executive mitigation plan.", 
        agent=manager, 
        expected_output="A final strategic Risk Intelligence Report."
    )

    # Form the Crew
    risk_crew = Crew(
        agents=[market_agent, scoring_agent, status_agent, manager],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential
    )

    with st.status("Agents are collaborating on analysis...", expanded=True) as status:
        result = risk_crew.kickoff()
        status.update(label="Analysis Complete", state="complete")
    
    st.markdown("### 📊 Final Executive Report")
    st.markdown(result.raw)
