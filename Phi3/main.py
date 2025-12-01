from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import os

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo:str = Field("motivo pelo qual é interessante visitar essa cidade")

class Restaurantes(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurantes:str = Field("Restaurantes recomndados na cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino) ### pydantic difene o formato da resposta de saída, que é em formato json
parseador_restaurante = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template =""" 
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables = {"formato_de_saida": parseador_destino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template =""" 
    Sugira restaurantes populares entre locais em {cidade}.
    {formato_de_saida}
    """,
    partial_variables = {"formato_de_saida": parseador_restaurante.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}"
)

modelo_raw = pipeline(
    task="text-generation",
    model="microsoft/Phi-3-mini-128k-instruct",
    truncation=True
)

modelo = HuggingFacePipeline(pipeline=modelo_raw)

cadeia_1 = prompt_cidade | modelo | parseador_destino
extract_city = RunnableLambda(lambda destino: {"cidade": destino.cidade})
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurante
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

cadeia = cadeia_1 | extract_city | cadeia_2 | cadeia_3

resposta = cadeia.invoke({
    "interesse": "praias"
})
print(resposta.content)