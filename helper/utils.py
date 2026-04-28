from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re

def call_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> dict:
    """Call the LLM and return parsed JSON from the response."""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    text = response.content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)
