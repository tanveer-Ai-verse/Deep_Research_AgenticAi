from dotenv import load_dotenv
import os
import asyncio
from google import genai
from google.genai import types

load_dotenv(override=True)

# Initialize official Google GenAI client
client = genai.Client(http_options={"api_version": "v1beta"})
MODEL_NAME = "gemini-2.5-flash"

# Custom transparent Agent & Runner framework bridge for Gemini with web search grounding
def function_tool(func):
    func.description = (func.__doc__ or "").strip()
    return func

class ModelSettings:
    def __init__(self, tool_choice=None):
        self.tool_choice = tool_choice

class Agent:
    def __init__(self, name: str, instructions: str = "", model: str = MODEL_NAME, tools: list = None, output_type=None, model_settings=None):
        self.name = name
        self.instructions = instructions
        self.model = model if isinstance(model, str) else MODEL_NAME
        self.tools = tools or []
        self.output_type = output_type
        self.model_settings = model_settings

class RunResult:
    def __init__(self, final_output, history: list):
        self.final_output = final_output
        self._history = history

    def to_input_list(self) -> list:
        return list(self._history)

class Runner:
    @staticmethod
    async def run(agent: Agent, input_data) -> RunResult:
        history = []
        if isinstance(input_data, str):
            history.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            history.extend(input_data)

        gemini_contents = []
        for msg in history:
            role = "model" if msg["role"] in ("assistant", "model") else "user"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )

        tool_map = {fn.__name__: fn for fn in agent.tools}
        
        config = types.GenerateContentConfig(
            system_instruction=agent.instructions or None,
            tools=agent.tools if agent.tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if agent.tools else None
        )

        step = 1
        while True:
            response = await client.aio.models.generate_content(
                model=agent.model,
                contents=gemini_contents,
                config=config
            )

            if response.function_calls:
                gemini_contents.append(response.candidates[0].content)
                tool_response_parts = []
                for fc in response.function_calls:
                    func = tool_map.get(fc.name)
                    output = func(**fc.args) if func else f"Error: Tool {fc.name} not found"
                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"result": output}
                        )
                    )
                gemini_contents.append(types.Content(role="user", parts=tool_response_parts))
                step += 1
            else:
                final_output = response.text or ""
                history.append({"role": "assistant", "content": str(final_output)})
                return RunResult(final_output, history)

# Gemini native web search grounding tool replacement
@function_tool
def web_search_tool(query: str) -> str:
    """Search the web for the given query using Gemini native grounding search."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"Search the web and provide detailed, factual summary regarding: {query}",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    return response.text or "No search results found."

INSTRUCTIONS = """
You are a research assistant. Given a search term, you search the web for that term and 
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
"""

settings = ModelSettings(tool_choice="required")
tools = [web_search_tool]

search_agent = Agent(name="Search Agent", instructions=INSTRUCTIONS, tools=tools, model=MODEL_NAME, model_settings=settings)
