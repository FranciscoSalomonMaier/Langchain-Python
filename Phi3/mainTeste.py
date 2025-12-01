#stabilityai/stablelm-2-1_6b

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug

set_debug(True)

model = pipeline(
    task="text-generation",
    model="microsoft/Phi-3-mini-128k-instruct",
    max_length=256,
    truncation=True
)

llm = HuggingFacePipeline(pipeline=model)

# Create the prompt template
template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand")
# template = PromptTemplate.from_template("Explique em português: {topic} em detalhes, de forma que uma criança de {age} anos consiga entender.")

chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

response = chain.invoke({"topic": topic, "age":age})
print(response)