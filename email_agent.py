from dotenv import load_dotenv
import os
import json
import asyncio
from google import genai
from google.genai import types

load_dotenv(override=True)

# Initialize official Google GenAI client
client = genai.Client(http_options={"api_version": "v1beta"})
MODEL_NAME = "gemini-2.5-flash"
USE_EMAIL = os.getenv("USE_EMAIL", "true").lower() == "true"

# Custom transparent Agent & Runner framework bridge for Gemini
def function_tool(func):
    func.description = (func.__doc__ or "").strip()
    return func

class ModelSettings:
    def __init__(self, tool_choice=None):
        self.tool_choice = tool_choice

class Agent:
    def __init__(self, name: str, instructions: str = "", model: str = MODEL_NAME, tools: list = None, model_settings=None):
        self.name = name
        self.instructions = instructions
        self.model = model if isinstance(model, str) else MODEL_NAME
        self.tools = tools or []
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

# Messenger utilities (fallback to print/push if messenger module is absent)
try:
    from messenger import send_email, push
except ImportError:
    import smtplib
    from email.message import EmailMessage
    
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
    EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
    
    def send_email(subject, text_body, html_body):
        if not (EMAIL_ADDRESS and EMAIL_SMTP_SERVER and EMAIL_APP_PASSWORD):
            print(f"[Mock Email Sent] Subject: {subject}\n{text_body}")
            return
        msg = EmailMessage()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_ADDRESS
        msg["Subject"] = subject
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")
        with smtplib.SMTP(EMAIL_SMTP_SERVER, 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.send_message(msg)

    def push(message):
        print(f"Push: {message}")

settings = ModelSettings(tool_choice="required")

@function_tool
def send_email_tool(subject: str, text_body: str, html_body: str) -> str:
    """
    Send out an email with the given subject and body
    
    Args:
        subject: The subject of the email
        text_body: The body of the email as plain text
        html_body: The HTML body of the email
    """
    if USE_EMAIL:
        send_email(subject, text_body, html_body)
    else:
        push(f"Subject: {subject}\n\n{text_body}")
    return "Email sent successfully"


INSTRUCTIONS = """
You are provided with a detailed report. Use your tool to send an email, converting the report into
a clean, well presented HTML email with an appropriate subject line.
"""

email_agent = Agent(name="Email Agent", instructions=INSTRUCTIONS, tools=[send_email_tool], model=MODEL_NAME, model_settings=settings)