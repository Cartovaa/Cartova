from langchain_core.messages import SystemMessage, HumanMessage

def call_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> dict:
    """Call the LLM and return parsed JSON from the response."""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    text = response.content.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
