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
question = "Quais os dados da USP?"
question = "Quais os dados da uNiCAmP?"
question = "Quais os dados da uNi CAmP?"
question = "Quais os dados da uNicomP?"
question = "Dentre USP e UFRJ, qual você recomenda para a acadêmica Ana?"
question = "Dentre uni camp e USP, qual você recomenda para a Ana?"
question = "Quais as faculdades com melhores chances para a Ana entrar?"
question = "Dentre todas as faculdades existentes, quais Ana possui mais chance de entrar?"
question = "Além das faculdades favoritas da Ana existem outras faculdades. Considere elas também. Quais Ana possui mais chance de entrar?"

agent_openai = AgentOpenAIFunctions()
executor = AgentExecutor(agent=agent_openai.agent,
                         tools=agent_openai.tools,
                         verbose=True)

response = executor.invoke({"input": question})
print(response)
