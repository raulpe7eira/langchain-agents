import os

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, Tool
from student import AcademicProfileTool, StudentDataTool

class AgentOpenAIFunctions:
  def __init__(self):
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     api_key=os.getenv("OPENAI_API_KEY"))

    academic_profile_tool = AcademicProfileTool()
    student_data_tool = StudentDataTool()

    self.tools = [
      Tool(name = student_data_tool.name,
           func = student_data_tool.run,
           description = student_data_tool.description),
      Tool(name = academic_profile_tool.name,
           func = academic_profile_tool.run,
           description = academic_profile_tool.description)
    ]

    # OpenAI Functions
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    # self.agent = create_openai_tools_agent(llm, self.tools, prompt)

    # ReAct
    prompt = hub.pull("hwchase17/react")
    self.agent = create_react_agent(llm, self.tools, prompt)
