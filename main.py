from dotenv import load_dotenv
from agent import AgentOpenAIFunctions
from langchain.agents import AgentExecutor

load_dotenv()

question = "Quais os dados da Ana?"
question = "Quais os dados da Bianca?"
question = "Quais os dados da Ana e da Bianca?"
question = "Crie um perfil acadêmico para a Ana!"
question = "Compare o perfil acadêmico da Ana com o da Bianca!"
question = "Tenho sentido Ana desanimada com cursos de matemática. Seria uma boa parear ela com a Bianca?"
question = "Tenho sentido Ana desanimada com cursos de matemática. Seria uma boa parear ela com o Marcos?"

agent_openai = AgentOpenAIFunctions()
executor = AgentExecutor(agent=agent_openai.agent,
                         tools=agent_openai.tools,
                         verbose=True)

response = executor.invoke({"input": question})
print(response)
