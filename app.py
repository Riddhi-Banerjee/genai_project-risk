import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# --- 1. SETUP ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide")

# Secure API Key retrieval
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    # --- 2. TOOLS (The Agent's "Hands") ---
    @tool("csv_search_tool")
    def csv_search_tool(query: str):
        """Use this tool to search for project or financial data in local CSV files."""
        # This function helps agents decide which file to read based on your query
        if "market" in query.lower():
            return pd.read_csv('market_trends.csv').tail(10).to_string()
        elif "transaction" in query.lower() or "payment" in query.lower():
            return pd.read_csv('transaction.csv').head(20).to_string()
        else:
            return pd.read_csv('project_risk_raw_dataset.csv').head(20).to_string()

    # --- 3. AGENTS (The "Brains") ---
    # Based on your requirement: Market Analyst, Risk Scorer, and Project Manager
    market_agent = Agent(
        role='Market Analyst',
        goal='Identify economic risks from market_trends.csv',
        backstory='Expert in macroeconomics and industry trends.',
        tools=[csv_search_tool],
        llm=llm
    )

    finance_agent = Agent(
        role='Risk Scorer',
        goal='Analyze transaction.csv for overdue payments.',
        backstory='Financial auditor with a focus on project budget health.',
        tools=[csv_search_tool],
        llm=llm
    )

    manager = Agent(
        role='Project Risk Manager',
        goal='Synthesize reports into a final strategy.',
        backstory='Chief Risk Officer who provides the final executive summary.',
        llm=llm
    )

    # --- 4. UI ---
    st.title("🛡️ Agentic Risk Command")
    target = st.text_input("Project ID:", "PROJ_0001")

    if st.button("🚀 Start Agentic Audit"):
        t1 = Task(description=f"Analyze market for {target}", agent=market_agent, expected_output="Economic summary.")
        t2 = Task(description=f"Analyze finances for {target}", agent=finance_agent, expected_output="Financial report.")
        t3 = Task(description="Combine all findings into one strategy.", agent=manager, expected_output="Executive Report.")

        crew = Crew(agents=[market_agent, finance_agent, manager], tasks=[t1, t2, t3], verbose=True)
        
        with st.status("Agents are collaborating...", expanded=True):
            result = crew.kickoff()
            st.markdown(result.raw)
else:
    st.warning("Please provide an API Key to continue.")
