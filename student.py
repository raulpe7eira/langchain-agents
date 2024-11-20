import json
import os
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

def find_student_data(student):
  data = pd.read_csv("documentos/estudantes.csv")
  student_data = data[data["USUARIO"] == student]

  if student_data.empty:
    return {}

  return student_data.iloc[:1].to_dict()

class StudentExtractor(BaseModel):
  student:str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla.")

class StudentData(BaseTool):
  name="StudentData"
  description="Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico."

  def _run(self, input: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     api_key=os.getenv("OPENAI_API_KEY"))

    parser = JsonOutputParser(pydantic_object=StudentExtractor)

    template = PromptTemplate(
      template="Você deve analisar a {input} e extrair o nome do usuário informado. Formato de saída: {output}",
      input_variables=["input"],
      partial_variables={"output": parser}
    )

    chain = template | llm | parser
    response = chain.invoke({"input": input})

    student = response["student"]
    student = student.lower()
    student_data = find_student_data(student)

    return json.dumps(student_data)
