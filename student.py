import json
import os
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from typing import List

def find_student_data(student):
  data = pd.read_csv("documentos/estudantes.csv")
  student_data = data[data["USUARIO"] == student]

  if student_data.empty:
    return {}

  return student_data.iloc[:1].to_dict()

class StudentExtractor(BaseModel):
  student:str = Field("""Nome do estudante informado, sempre em letras minúsculas.
                      Exemplo: joão, carlos, joana, carla.""")

class StudentDataTool(BaseTool):
  name="StudentData"
  description="""Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico.
Passe para essa ferramenta como argumento o nome do estudante."""

  def _run(self, input: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     api_key=os.getenv("OPENAI_API_KEY"))

    parser = JsonOutputParser(pydantic_object=StudentExtractor)
    template = PromptTemplate(
      template="""Você deve analisar a {input} e extrair o nome do usuário informado.
      Entrada:
      -----------------
      {input}
      -----------------
      Formato de saída:
      {output}""",
      input_variables=["input"],
      partial_variables={"output": parser.get_format_instructions()}
    )

    chain = template | llm | parser
    response = chain.invoke({"input": input})

    student = response["student"]
    student = student.lower().strip()
    student_data = find_student_data(student)

    return json.dumps(student_data)

class Grade(BaseModel):
  knowledge_area:str = Field("Nome da área de conhecimento")
  grade:float = Field("Nota na área de conhecimento")

class StudentAcademicProfile(BaseModel):
  name:str = Field("nome do estudante")
  completion_year:int = Field("ano de conclusão")
  grades:List[Grade] = Field("Lista de notas das disciplinas e áreas de conhecimento")
  summary:str = Field("Resumo das principais características desse estudante de forma a torná-lo único e um ótimo potencial estudante para faculdades. Exemplo: só este estudante tem bla bla bla")

class AcademicProfileTool(BaseTool):
  name = "AcademicProfile"
  description = """Cria um perfil acadêmico de um estudante.
  Esta ferramenta requer como entrada todos os dados do estudante.
  Eu sou imcapaz de buscar os dados do estudante.
  Você tem que buscar os dados do estudante antes de me invocar."""

  def _run(self, input:str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     api_key=os.getenv("OPENAI_API_KEY"))

    parser = JsonOutputParser(pydantic_object=StudentAcademicProfile)
    template = PromptTemplate(
      template = """
      - Formate o estudante para seu perfil acadêmico.
      - Com os dados, identifique as opções de universidades sugeridas e cursos compatíveis com o interesse do aluno.
      - Destaque o perfil do aluno dando enfase principalmente naquilo que faz sentido para as instituições de interesse do aluno.
      Persona: Você é uma consultora de carreira e precisa indicar com detalhes, riqueza, mas direta ao ponto para o estudante as opções e consequências possíveis.
      Informações atuais:

      {student_data}

      {output}""",
      input_variables=["student_data"],
      partial_variables={"output": parser.get_format_instructions()})

    chain = template | llm | parser
    response = chain.invoke({"student_data": input})

    return response
