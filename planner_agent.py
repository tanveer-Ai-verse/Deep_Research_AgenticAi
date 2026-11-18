from pydantic import BaseModel, Field
import os
import json
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

# Initialize official Google GenAI client
client = genai.Client(http_options={"api_version": "v1beta"})
MODEL_NAME = "gemini-2.5-flash"
HOW_MANY_SEARCHES = int(os.getenv("HOW_MANY_SEARCHES", 5))

# Custom transparent Agent & Runner framework bridge for Gemini
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

        config_kwargs = {
            "system_instruction": agent.instructions or None,
            "tools": agent.tools if agent.tools else None,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True) if agent.tools else None
        }

        if agent.output_type:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = agent.output_type

        config = types.GenerateContentConfig(**config_kwargs)

        response = await client.aio.models.generate_content(
            model=agent.model,
            contents=gemini_contents,
            config=config
        )

        raw_text = response.text or ""
        if agent.output_type:
            try:
                if isinstance(raw_text, str):
                    data_dict = json.loads(raw_text)
                else:
                    data_dict = raw_text
                final_output = agent.output_type(**data_dict)
            except Exception as e:
                final_output = raw_text
        else:
            final_output = raw_text

        history.append({"role": "assistant", "content": str(final_output)})
        return RunResult(final_output, history)


INSTRUCTIONS = f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
"""

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")
    
planner_agent = Agent(name="Planner Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=WebSearchPlan)