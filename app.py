import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai.tools import BaseTool

# --- 1. UI SETUP ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide")
st.title("🛡️ Agentic Risk Intelligence Engine")

# --- 2. API KEY HANDLING ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.warning("🔑 Please provide a Google API Key in the sidebar or app secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# --- 3. UPDATED TOOL CLASS (Fixes Pydantic Error) ---
class RiskDataTool(BaseTool):
    name: str = "Risk Data Reader"
    description: str = "Reads project, financial, or market data from local CSV files."

    def _run(self, file_name: str) -> str:
        try:
            # Basic security check for file names
            allowed_files = ['project_risk_raw_dataset.csv', 'transaction.csv', 'market_trends.csv']
            if file_name not in allowed_files:
                return f"Error: Access to {file_name} is not allowed."
            
            df = pd.read_csv(file_name)
            # Return only a summary to avoid token overflow
            return df.head(50).to_string() 
        except Exception as e:
            return f"Error reading {file_name}: {e}"

# Initialize one instance of the tool
csv_tool = RiskDataTool()

# --- 4. AGENT DEFINITIONS (Compatible with latest CrewAI) ---
market_agent = Agent(
    role='Market Analysis Agent',
    goal='Identify external economic risks from market_trends.csv',
    backstory='Expert economist monitoring inflation and interest rates.',
    tools=[csv_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

scoring_agent = Agent(
    role='Risk Scoring Agent',
    goal='Identify financial risks and overdue payments from transaction.csv',
    backstory='Forensic auditor focusing on budget leaks and unpaid invoices.',
    tools=[csv_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

status_tracker = Agent(
    role='Project Status Tracking Agent',
    goal='Monitor project health and schedule delays from project_risk_raw_dataset.csv',
    backstory='Senior PM auditor who finds complexity and resource risks.',
    tools=[csv_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

manager = Agent(
    role='Project Risk Manager',
    goal='Synthesize all agent reports into a final executive mitigation strategy.',
    backstory='Chief Risk Officer responsible for the final strategic decision.',
    llm=llm,
    verbose=True
)

# --- 5. EXECUTION ---
target_project = st.text_input("Enter Project ID:", "PROJ_0001")

if st.button("🚀 Run Agentic Audit"):
    t1 = Task(description=f"Analyze market trends relevant to {target_project} in market_trends.csv", agent=market_agent, expected_output="Economic risk summary.")
    t2 = Task(description=f"Analyze transaction history for {target_project} in transaction.csv", agent=scoring_agent, expected_output="Financial risk report.")
    t3 = Task(description=f"Review status and complexity for {target_project} in project_risk_raw_dataset.csv", agent=status_tracker, expected_output="Internal health report.")
    t4 = Task(description="Combine all findings into a 3-step mitigation strategy.", agent=manager, expected_output="Executive Risk Strategy.")

    risk_crew = Crew(
        agents=[market_agent, scoring_agent, status_tracker, manager],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential
    )

    with st.status("Agents are collaborating...", expanded=True) as status:
        result = risk_crew.kickoff()
        status.update(label="Analysis Complete", state="complete")
    
    st.markdown("### 📊 Final Intelligence Report")
    st.markdown(result.raw)
