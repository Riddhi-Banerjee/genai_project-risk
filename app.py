import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# --- 1. UI SETUP ---
st.set_page_config(page_title="Risk Intelligence Engine", layout="wide")
st.title("🛡️ Agentic Risk Intelligence Engine")

# --- 2. API KEY & ENVIRONMENT ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("🔑 Please provide a Google API Key to start.")
    st.stop()

# Force set the environment variable for LiteLLM
os.environ["GOOGLE_API_KEY"] = api_key

# --- 3. THE FIX: STABLE LLM CONFIGURATION ---
# Using 'google/gemini-1.5-flash' forces the use of the stable Google AI Studio API
# instead of the buggy Vertex v1beta path.
stable_llm = LLM(
    model="google/gemini-1.5-flash",
    api_key=api_key,
    temperature=0.2
)

# --- 4. TOOL DEFINITION ---
class RiskDataTool(BaseTool):
    name: str = "Risk Data Reader"
    description: str = "Reads data from market_trends.csv or transaction.csv."

    def _run(self, file_name: str) -> str:
        try:
            df = pd.read_csv(file_name)
            return df.head(20).to_string()
        except Exception as e:
            return f"Error: {e}"

csv_tool = RiskDataTool()

# --- 5. AGENTS ---
market_agent = Agent(
    role='Market Analyst',
    goal='Find risks in market_trends.csv',
    backstory='Expert in global macroeconomics.',
    tools=[csv_tool],
    llm=stable_llm, # Linked to the fixed LLM object
    verbose=True
)

scoring_agent = Agent(
    role='Financial Auditor',
    goal='Find risks in transaction.csv',
    backstory='Specialist in corporate finance.',
    tools=[csv_tool],
    llm=stable_llm,
    verbose=True
)

manager = Agent(
    role='Risk Manager',
    goal='Summarize findings into a plan.',
    backstory='Chief Risk Officer.',
    llm=stable_llm,
    verbose=True
)

# --- 6. EXECUTION ---
target_project = st.text_input("Project ID:", "PROJ_0001")

if st.button("🚀 Run Analysis"):
    t1 = Task(description=f"Analyze market for {target_project}", agent=market_agent, expected_output="Market Report")
    t2 = Task(description=f"Analyze finance for {target_project}", agent=scoring_agent, expected_output="Finance Report")
    t3 = Task(description="Create mitigation plan", agent=manager, expected_output="Final Strategy")

    crew = Crew(
        agents=[market_agent, scoring_agent, manager],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    with st.status("Agents working...", expanded=True) as status:
        try:
            result = crew.kickoff()
            status.update(label="Complete!", state="complete")
            st.markdown(result.raw)
        except Exception as e:
            st.error(f"Error: {e}")
