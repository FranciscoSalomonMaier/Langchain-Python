from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain_core.globals import set_debug
import os

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint


load_dotenv()
api_key_hug = os.getenv("HUGG_API_KEY")

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo:str = Field("motivo pelo qual é interessante visitar essa cidade")

parseador = JsonOutputParser(pydantic_object=Destino) ### pydantic difene o formato da resposta de saída, que é em formato json

prompt_cidade = PromptTemplate(
    template =""" 
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables = {"formato_de_saida": parseador.get_format_instructions()}
)

# modelo = InferenceClient("meta-llama/Meta-Llama-3.1-8B-Instruct", token=api_key_hug)

modelo = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B",
    huggingfacehub_api_token=api_key_hug
)

cadeia = prompt_cidade | modelo | parseador

resposta = cadeia.invoke({
    "interesse": "praias"
})
print(resposta.content)