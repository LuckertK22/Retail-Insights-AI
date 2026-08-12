"""
api/agent/graph.py

Grafo LangGraph que define el flujo del agente conversacional.

Flujo:
  START → agent (LLM decide) → ¿tool_calls? → tools → agent → END
                                                   ↓ no
                                                  END
"""

from typing import Literal
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from api.agent.tools import consultar_insights, predecir_ventas

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0,
)

tools = [consultar_insights, predecir_ventas]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MessagesState):
    """Nodo 'agent': el LLM decide si llamar una tool o responder directamente."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Si el último mensaje tiene tool_calls, va a 'tools'. Si no, termina."""
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END


def tools_node(state: MessagesState):
    """Nodo 'tools': ejecuta las tools seleccionadas y devuelve el resultado."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = next(t for t in tools if t.name == tool_call["name"])
        output = tool.invoke(tool_call["args"])
        result.append(
            {
                "role": "tool",
                "content": output,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": result}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.add_edge("__start__", "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

agent_app = workflow.compile()
