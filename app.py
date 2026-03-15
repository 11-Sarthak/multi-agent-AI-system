import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from agent import graph

st.set_page_config(
    page_title="Multi-Agent AI Research System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Research & Report Generator")

st.markdown("""
This system uses **multiple AI agents**:

- 🔎 **Researcher** → collects information  
- 📊 **Analyst** → extracts insights  
- 📝 **Writer** → generates the final report  

The **Supervisor Agent** coordinates the workflow.
""")

# Create memory thread
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Topic input
topic = st.text_input(
    "Enter a topic to research",
    placeholder="Example: Benefits and risks of AI in healthcare"
)

generate = st.button("Generate Report")

if generate and topic:

    with st.spinner("Agents are working on your report..."):

        result = graph.invoke(
            {"messages": [HumanMessage(content=topic)]},
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

    report = result.get("final_report", "Report not generated")

    st.success("Report Generated!")

    st.subheader("📄 Final Report")
    st.text(report)

    st.download_button(
        label="Download Report",
        data=report,
        file_name="ai_report.txt",
        mime="text/plain"
    )