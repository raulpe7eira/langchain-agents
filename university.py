import json
import os
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

def find_university_data(university:str):
  data = pd.read_csv("documentos/universidades.csv")
  data["NOME_FACULDADE"] = data["NOME_FACULDADE"].str.lower()
  university_data = data[data["NOME_FACULDADE"] == university]

  if university_data.empty:
    return {}

  return university_data.iloc[:1].to_dict()

def find_universities_data():
  universities_data = pd.read_csv("documentos/universidades.csv")
  return universities_data.to_dict()

class UniversityExtractor(BaseModel):
  university:str = Field("O nome da universidade, sempre em letras minúsculas.")

class UniversityDataTool(BaseTool):
  name = "UniversityDataTool"
  description = """Esta ferramenta extrai os dados de uma universidade.
  Passe para essa ferramenta como argumento o nome da universidade."""

  def _run(self, input:str) -> str:
    llm = ChatOpenAI(model="gpt-4o",
                     api_key=os.getenv("OPENAI_API_KEY"))

    parser = JsonOutputParser(pydantic_object=UniversityExtractor)
    template = PromptTemplate(
      template="""
      Você deve analisar a entrada a seguir e extrair o nome de universidade informada em minúsculo.
      Entrada:
      -----------------
      {input}
      -----------------
      Formato de saída:
      {output}""",
      input_variables=["input"],
      partial_variables={"output" : parser.get_format_instructions()})

    chain = template | llm | parser
    response = chain.invoke({"input" : input})

    university = response['university']
    university = university.lower().strip()
    university_data = find_university_data(university)

    return json.dumps(university_data)

class AllUniversitiesTool(BaseTool):
  name = "AllUniversitiesTool"
  description = "Carrega os dados de todas as universidades. Não é necessário nenhum parâmetro de entrada."

  def _run(self, input:str):
    universities = find_universities_data()
    return universities
