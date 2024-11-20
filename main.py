from dotenv import load_dotenv
from agent import AgentOpenAIFunctions
from langchain.agents import AgentExecutor

load_dotenv()

question = "Quais os dados da Ana?"
question = "Quais os dados da Bianca?"
question = "Quais os dados da Ana e da Bianca?"

agent_openai = AgentOpenAIFunctions()
executor = AgentExecutor(agent=agent_openai.agent,
                         tools=agent_openai.tools,
                         verbose=True)

response = executor.invoke({"input": question})
print(response)
