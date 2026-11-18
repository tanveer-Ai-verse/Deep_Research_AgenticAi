from pydantic import BaseModel, Field
import os
import time
import json
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

# Initialize official Google GenAI client
client = genai.Client(http_options={"api_version": "v1beta"})
MODEL_NAME = "gemini-2.5-flash"

# Custom transparent Agent & Runner framework bridge for Gemini
def function_tool(func):
    func.description = (func.__doc__ or "").strip()
    return func

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

INSTRUCTIONS = """
You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some research.
Generate a comprehensive report based on the research and the query.
The final output should be in markdown format, and it should be lengthy and detailed. Aim 
for 5-10 pages of content, at least 1000 words.
"""


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


writer_agent = Agent(name="Writer Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=ReportData)