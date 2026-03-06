import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- 1. APP CONFIG & UI ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide")
st.title("🛡️ Agentic Risk Intelligence Engine")

# --- 2. SECURE API KEY HANDLING ---
# Checks Streamlit Secrets first, then Sidebar input
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("🔑 Please provide a Google API Key in the sidebar or Streamlit Secrets to begin.")
    st.stop()

# Set environment variables for both LiteLLM and Google SDK
os.environ["GEMINI_API_KEY"] = api_key
os.environ["GOOGLE_API_KEY"] = api_key

# --- 3. CUSTOM DATA TOOL ---
class RiskDataTool(BaseTool):
    name: str = "Risk Data Reader"
    description: str = "Reads project, financial, or market data from local CSV files."

    def _run(self, file_name: str) -> str:
        try:
            # Files must be in the same root directory on GitHub
            df = pd.read_csv(file_name)
            # Head(30) prevents token limit errors (413/429)
            return df.head(30).to_string()
        except Exception as e:
            return f"Error reading {file_name}: {e}"

# Initialize the tool instance
csv_tool = RiskDataTool()

# --- 4. AGENT DEFINITIONS ---
# Format "gemini/gemini-1.5-flash" forces LiteLLM to use the correct API endpoint
gemini_model = "gemini/gemini-1.5-flash"

market_agent = Agent(
    role='Market Analysis Agent',
    goal='Identify external economic risks from market_trends.csv',
    backstory='You are a financial economist specializing in macro-trends and inflation.',
    tools=[csv_tool],
    llm=gemini_model,
    verbose=True,
    allow_delegation=False
)

scoring_agent = Agent(
    role='Risk Scoring Agent',
    goal='Identify financial risks and overdue payments from transaction.csv',
    backstory='You are a forensic auditor focusing on budget health and payment history.',
    tools=[csv_tool],
    llm=gemini_model,
    verbose=True,
    allow_delegation=False
)

manager = Agent(
    role='Project Risk Manager',
    goal='Synthesize all agent reports into a final executive strategy.',
    backstory='You are the Chief Risk Officer who turns raw data into actionable business plans.',
    llm=gemini_model,
    verbose=True
)

# --- 5. EXECUTION ENGINE ---
target_project = st.text_input("Enter Project ID (e.g., PROJ_0001):", "PROJ_0001")

if st.button("🚀 Execute Agentic Audit"):
    # Define the workflow tasks
    t1 = Task(
        description=f"Analyze market context for {target_project} using data from market_trends.csv", 
        agent=market_agent, 
        expected_output="A summary of external market risks."
    )
    t2 = Task(
        description=f"Analyze financial history for {target_project} using data from transaction.csv", 
        agent=scoring_agent, 
        expected_output="A report on financial exposure and overdue payments."
    )
    t3 = Task(
        description="Synthesize findings from the Market and Scoring agents into a 3-step executive mitigation plan.", 
        agent=manager, 
        expected_output="A final strategic Risk Intelligence Report."
    )

    # Assemble the Crew
    risk_crew = Crew(
        agents=[market_agent, scoring_agent, manager],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    # Run the process and display logs
    with st.status("Agents are collaborating...", expanded=True) as status:
        try:
            result = risk_crew.kickoff()
            status.update(label="Analysis Complete!", state="complete")
            
            st.markdown("---")
            st.markdown("### 📊 Final Executive Report")
            st.markdown(result.raw)
        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")
