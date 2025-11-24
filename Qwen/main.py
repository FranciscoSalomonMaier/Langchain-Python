from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


dias = 7
numero_criancas = 2
atividade = "praia"
prompt_template = PromptTemplate(
    template =""" 
    Crie um roteiro de viagem de {dias} dias, 
    para uma família com {numero_criancas} crianças 
    que gostam de {atividade}.
    """
)
prompt_text = prompt_template.format(
    dias=dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)
print(prompt_text)
messages = [
    {"role": "system", "content": "Você é Qwen, criado por Alibaba Cloud. Você é um assistente bacana e útil."},
    {"role": "user", "content": prompt_text}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
