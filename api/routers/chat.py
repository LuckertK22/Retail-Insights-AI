"""
api/routers/chat.py

Endpoint POST /chat/ — recibe texto libre y lo pasa al agente LangGraph.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from api.agent.graph import agent_app

router = APIRouter()


class ChatInput(BaseModel):
    """Entrada del endpoint /chat/."""

    message: str = Field(..., example="¿Cuánto vendería si vendo 5 Chairs en West con 20% de descuento?")


class ChatOutput(BaseModel):
    """Salida del endpoint /chat/."""

    response: str


@router.post("/", response_model=ChatOutput)
def chat(input_data: ChatInput):
    """
    Recibe una pregunta en texto libre y devuelve la respuesta del agente.
    """
    messages = [HumanMessage(content=input_data.message)]
    result = agent_app.invoke({"messages": messages})

    last_msg = result["messages"][-1]
    raw_content = last_msg.content

    if isinstance(raw_content, list) and len(raw_content) > 0:
        response_text = raw_content[0].get("text", str(raw_content))
    elif isinstance(raw_content, str):
        response_text = raw_content
    else:
        response_text = str(raw_content)

    return ChatOutput(response=response_text)
