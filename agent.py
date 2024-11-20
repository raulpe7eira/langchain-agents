import os

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, create_react_agent, Tool
from student import AcademicProfileTool, StudentDataTool
from university import AllUniversitiesTool, UniversityDataTool

class AgentOpenAIFunctions:
  def __init__(self):
    llm = ChatOpenAI(model="gpt-4o",
                     api_key=os.getenv("OPENAI_API_KEY"))

    student_data_tool = StudentDataTool()
    academic_profile_tool = AcademicProfileTool()

    all_universities_tool = AllUniversitiesTool()
    university_data_tool = UniversityDataTool()

    self.tools = [
      # Student tools
      Tool(name = student_data_tool.name,
           func = student_data_tool.run,
           description = student_data_tool.description),
      Tool(name = academic_profile_tool.name,
           func = academic_profile_tool.run,
           description = academic_profile_tool.description),
      # University tools
      Tool(name = university_data_tool.name,
           func = university_data_tool.run,
           description = university_data_tool.description),
      Tool(name = all_universities_tool.name,
           func = all_universities_tool.run,
           description = all_universities_tool.description)
    ]

    # OpenAI Functions
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    # self.agent = create_openai_tools_agent(llm, self.tools, prompt)

    # ReAct
    prompt = hub.pull("hwchase17/react")
    self.agent = create_react_agent(llm, self.tools, prompt)
