import os

from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

load_dotenv()

class StudentExtractor(BaseModel):
  student:str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla.")

class StudentData(BaseTool):
  name="StudentData"
  description="Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico."

  def _run(self, input: str) -> str:
    llm = ChatOpenAI(
      model="gpt-4o",
      api_key=os.getenv("OPENAI_API_KEY")
    )

    parser = JsonOutputParser(pydantic_object=StudentExtractor)

    template = PromptTemplate(
      template="Você deve analisar a {input} e extrair o nome do usuário informado. Formato de saída: {output}",
      input_variables=["input"],
      partial_variables={"output": parser}
    )

    chain = template | llm | parser

    response = chain.invoke({"input": input})
    print(response)

    return response["student"]

question = "Quais os dados da Ana?"

response = StudentData().run(question)
print(response)
