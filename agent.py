import os

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain.agents import Tool
from student import StudentData

class AgentOpenAIFunctions:
  def __init__(self):
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     api_key=os.getenv("OPENAI_API_KEY"))

    student_data = StudentData()

    self.tools = [
      Tool(name = student_data.name,
           func = student_data.run,
           description = student_data.description)
    ]

    prompt = hub.pull("hwchase17/openai-functions-agent")
    self.agent = create_openai_tools_agent(llm, self.tools, prompt)
