import os
from dotenv import load_dotenv
from typing import TypedDict,Annotated,List,Literal,Dict,Any
from langgraph.checkpoint.memory import MemorySaver
import random
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.prompts import ChatPromptTemplate

memory=MemorySaver()


# %%
class supervisorstate(MessagesState):   
    next_agent:str
    research_data:str
    analysis:str
    final_report:str
    task_complete:bool
    current_task:str
    




# %%
load_dotenv()

llm = init_chat_model("groq:qwen/qwen3-32b")

# %%
llm

# %%
def create_supervisor_chain():  
    """creates the supervisior decision chain """

    supervisor_prompt = ChatPromptTemplate.from_messages([(
        "system", """you are supervisor managing a team of agents:  

        1. Researcher - Gathers information and data
        2. Analyst - Analyzes data and provides insights
        3. Writer - Creates reports and summaries

        Based on the current state and converstaion, decide which agent should work next.
        If the task is complete, respond with 'DONE'.

        Current state:  
        - Has research data: {has_research}
        - Has analysis: {has_analysis}
        - Has report: {has_report}

        Respond with only the agent name(researcher/analyst/writer) or 'DONE'

        """
    ), 
          ("human", "Task: {task}")

        
    ])

    return supervisor_prompt | llm


# %%
def supervisor_agent(state: supervisorstate)-> Dict:
    """supervisor decides next agent using groq llm"""

    messages= state["messages"]

    task=messages[-1].content if messages else "No task"

    #check what's been completed

    has_research=bool(state.get("research_data",""))
    has_analysis=bool(state.get("analysis",""))
    has_report=bool(state.get("final_report",""))

    #get llm decision

    chain=create_supervisor_chain()
    decision = chain.invoke({
        "task": task,
        "has_research":has_research,
        "has_analysis":has_analysis,
        "has_report": has_report
    })

    #parse decision
    decision_text=decision.content.strip().lower()
    print(decision)

    if "done" in decision_text or has_report:   
        next_agent="end"
        supervisor_msg="supervisor: all tasls complete ! great work team. "
    elif "researcher" in decision_text or not has_research: 
        next_agent = "researcher"
        supervisor_msg="supervisor: lets start with research. assigning the researcher .. "
    elif "analyst" in decision_text or (has_research and not has_analysis): 
        next_agent = "analyst"
        supervisor_msg="supervisor: Research done. Time fot analysis. Assigning to Analyst..."
    elif "writer" in decision_text or (has_analysis and not has_report):    
        next_agent = "writer"
        supervisor_msg="supervisor: Analysis done.  Let's create the report . Assigning to writer..."
    else:   
        next_agent = "end"
        supervisor_msg="supervisor: Task seems complete"    
    return {
        "messages":[AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
        }    


# %%
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict
from datetime import datetime

def researcher_agent(state: supervisorstate) -> Dict:
    """Uses Groq to collect research data"""

    task = state.get("current_task", "")

    research_prompt = f"""
You are a research specialist.

Research the following topic and provide structured findings.

Topic:
{task}

Provide:
1. Background information
2. Important facts
3. Current trends
4. Relevant statistics
5. Key examples
"""

    research_response = llm.invoke([
    SystemMessage(content="You are a research specialist."),
    HumanMessage(content=research_prompt)
])
    research_data = research_response.content

    return {
        "messages": [AIMessage(content="Researcher: Research completed and stored.")],
        "research_data": research_data,
        "next_agent": "analyst"
    }

# %%
def analyst_agent(state: supervisorstate) -> Dict:
    """Analyzes the research data"""

    research_data = state.get("research_data", "")
    task = state.get("current_task", "")

    analysis_prompt = f"""
You are a data analyst.

Analyze the following research findings.

Task:
{task}

Research Data:
{research_data[:1000]}

Provide:
1. Key patterns
2. Insights
3. Opportunities
4. Risks
5. Strategic implications
"""

    analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis = analysis_response.content

    return {
        "messages": [AIMessage(content="Analyst: Analysis completed.")],
        "analysis": analysis,
        "next_agent": "writer"
    }

# %%
def writer_agent(state: supervisorstate) -> Dict:   
    """uses groq to create final report"""   

    research_data=state.get("research_data","")
    analysis = state.get("analysis","")
    task=state.get("current_task","")

    #create writing prompt
    writing_prompt= f"""
You are a professional executive report writer.

IMPORTANT:
Do NOT show reasoning.
Do NOT include <think> tags.
Return ONLY the final report.

Task: {task}

Research Findings:
{research_data[:1000]}

Analysis:
{analysis[:1000]}

Create a well structured report with:

1. Executive Summary
2. Key Findings
3. Analysis & Insights
4. Recommendations
5. Conclusion

Keep it professional and concise.
"""
    # Get report from llm

    report_response=llm.invoke([HumanMessage(content=writing_prompt)])
    report=report_response.content


    # Clean reasoning tags if present
    report = report.replace("<think>", "").replace("</think>", "").strip()

    # Create final formatted report

    final_report=f"""
Final Report
{"="*50}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Topic: {task}
{"="*50}

{report}
"""
    
    
    return{
        "messages":[AIMessage(content=f"Writer:Report complete! See below for the full complete document.")],
        "final_report": final_report,
        "next_agent":"supervisor",
        "task_complete": True

    }
    







     



# %%
from typing import Literal
from langgraph.graph import END

def router(state: supervisorstate) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    """Routes to next agent based on state"""

    next_agent = state.get("next_agent", "supervisor")

    if next_agent == "end" or state.get("task_complete", False):
        return END

    if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
        return next_agent

    return "supervisor"

# %%
workflow=StateGraph(supervisorstate)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)

workflow.set_entry_point("supervisor")

#add routing

for node in ["supervisor","researcher","analyst","writer"]: 
    workflow.add_conditional_edges(
        node,
        router,{
            "supervisor": "supervisor",
            "researcher" :"researcher",
            "analyst":"analyst",
            "writer" : "writer",
            END: END
        }
        )
graph=workflow.compile(checkpointer=memory)   

# %%
graph

# %%
result = graph.invoke({
    "messages": [HumanMessage(content="What are the benefits and risks of AI in healthcare?")]
},
  config={"configurable": {"thread_id": "research-thread-1"}}
  )



# %%
print(result["final_report"])

# %%



