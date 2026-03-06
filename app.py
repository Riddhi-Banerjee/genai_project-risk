import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# --- 1. UI SETUP ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide")
st.title("🛡️ Agentic Risk Intelligence Engine")

# --- 2. API KEY HANDLING ---
# Priority 1: Streamlit Secrets | Priority 2: Sidebar Input
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.warning("🔑 Please provide a Google API Key to activate the intelligence agents.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# --- 3. THE AGENT TOOLS ---
@tool("read_csv_tool")
def read_csv_tool(file_name: str):
    """Reads a CSV file and returns the data as a string. 
    Available files: 'project_risk_raw_dataset.csv', 'transaction.csv', 'market_trends.csv'"""
    try:
        df = pd.read_csv(file_name)
        return df.to_string()
    except Exception as e:
        return f"Error reading file {file_name}: {e}"

# --- 4. AGENT DEFINITIONS ---
market_agent = Agent(
    role='Market Analysis Agent',
    goal='Identify external economic risks from market_trends.csv',
    backstory='You are a financial economist who monitors inflation and market volatility.',
    tools=[read_csv_tool],
    llm=llm,
    verbose=True
)

scoring_agent = Agent(
    role='Risk Scoring Agent',
    goal='Identify financial risks and overdue payments from transaction.csv',
    backstory='You are a forensic auditor focusing on budget utilization and payment delays.',
    tools=[read_csv_tool],
    llm=llm,
    verbose=True
)

status_tracker = Agent(
    role='Project Status Tracking Agent',
    goal='Monitor project health and schedule delays from project_risk_raw_dataset.csv',
    backstory='You are a senior project auditor who finds complexity and resource risks.',
    tools=[read_csv_tool],
    llm=llm,
    verbose=True
)

manager = Agent(
    role='Project Risk Manager',
    goal='Synthesize all agent reports into a final executive mitigation strategy.',
    backstory='You are the Chief Risk Officer. You turn technical data into business strategy.',
    llm=llm,
    verbose=True
)

# --- 5. EXECUTION ---
target_project = st.text_input("Enter Project ID (e.g., PROJ_0001):", "PROJ_0001")

if st.button("🚀 Run Agentic Audit"):
    # Define Tasks
    t1 = Task(description=f"Analyze market context for {target_project}", agent=market_agent, expected_output="Economic risk summary.")
    t2 = Task(description=f"Analyze transaction history for {target_project}", agent=scoring_agent, expected_output="Financial risk report.")
    t3 = Task(description=f"Review status for {target_project}", agent=status_tracker, expected_output="Internal health report.")
    t4 = Task(description="Create a 3-step mitigation strategy using all findings.", agent=manager, expected_output="Executive Risk Strategy.")

    # Form the Crew
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
