from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

numero_dias = 7
numero_criancas = 2
atividade = "praia"

prompt = f"Crie um roteiro de viagens, para um periodo de {numero_dias}, para uma familia com {numero_criancas} que busca atividades relacionadas a {atividade}"

cliente = OpenAI(api_key=api_key)
resposta = cliente.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente de roteiro de viagens."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(resposta)

resposta_em_texto = resposta.choices[0].message.content
print(resposta_em_texto)