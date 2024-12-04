# Configurações
import os
import openai
from dotenv import load_dotenv
import pickle
import json
import numpy as np
import time
import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

load_dotenv()
os.environ['VOYAGE_API_KEY'] = ('VOYAGE_API_KEY')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Classe VectorDB
class VectorDB:
    def __init__(self, name, api_key=None):
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, data):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts = [f"Heading: {item['chunk_heading']}\n\n Chunk Text: {item['text']}" for item in data]
        self._embed_and_store(texts, data)
        self.save_db()
        print("Vector database loaded and saved.")

    def _embed_and_store(self, texts, data):
        batch_size = 128
        result = []

        for i in range(0, len(texts), batch_size):
            response = openai.Embedding.create(
                input=texts[i: i + batch_size],
                model="text-embedding-ada-002"  # Modelo recomendado para embeddings
            )
            embeddings = [res['embedding'] for res in response['data']]
            result.extend(embeddings)

        self.embeddings = result
        self.metadata = data

    def search(self, query, k=5, similarity_threshold=0.75):
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            response = openai.Embedding.create(
                input=[query],
                model="text-embedding-ada-002"
            )
            query_embedding = response['data'][0]['embedding']
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        # Cálculo da similaridade utilizando produto escalar
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []

        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)

                if len(top_examples) >= k:
                    break
        self.save_db()
        return top_examples

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])

# Função para login no HuggingChat
def huggingchat_login(email, password, cookie_path_dir):
    sign = Login(email, password)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot

# Função para chamar o modelo via HuggingChat
def call_huggingchat(chatbot, prompt, model_name, retries=3, delay=20):
    models = chatbot.get_available_llm_models()
    model_index = next((i for i, m in enumerate(models) if m.id == model_name), None)

    if model_index is not None:
        chatbot.switch_llm(model_index)
        print(f"Modelo '{model_name}' selecionado com sucesso!")
    else:
        raise ValueError(f"Modelo '{model_name}' não encontrado entre os disponíveis.")

    for attempt in range(retries):
        try:
            chatbot.new_conversation(switch_to=True)
            response = chatbot.chat(prompt)
            return response
        except Exception as e:
            print(f"Erro na tentativa {attempt + 1}: {e}")
            if "You are sending too many messages" in str(e):
                time.sleep(delay)
            else:
                break

    return None

# Funções de recuperação e resposta
def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db, chatbot):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    Com base no termo de busca a abaixo, gere uma petição inicial que inclua o contexto recuperado como justificativa.
    <Entrada do usuário>
    {query}
    </Entrada do usuário>
    Você tem acesso aos seguintes documentos, use-os para fundamentar a petição inicial:
    <contexto>
    {context}
    </contexto>
    Por favor, permaneça fiel ao contexto subjacente e só se desvie dele se tiver 100% de certeza de que já sabe a resposta. 
    Construa uma petição simples e direta, evitando preâmbulos como 'Aqui está a resposta', etc.
    # TAREFA: Gere uma pequena petição inicial usando como referência do contexto recuperado. 
    """
    response = call_huggingchat(chatbot, prompt, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct")
    return response

# Aplicação em Streamlit
st.title("Gerador de petição inicial")

# Configuração de Login (E-mail e Senha fixos no código)
EMAIL = "hdsdosol@gmail.com"
PASSWD = "Lisa2210@" 

# Login no HuggingChat
cookie_path_dir = "./cookies/"
chatbot = huggingchat_login(EMAIL, PASSWD, cookie_path_dir)

if chatbot:
    st.success("Login realizado com sucesso!")

    # Carregar o documento e inicializar o banco de vetores
    with open('/home/lisamenezes/RAG-benchmark/data/fundamentos-train.json', 'r') as f:
        fundamentos_data = json.load(f)
    db = VectorDB("fundamentos")
    db.load_data(fundamentos_data)

    # Input de consulta
    query = st.text_input("Insira o assunto da petição inicial:")
    if query:
        # Recuperar contexto e gerar resposta
        results, context = retrieve_base(query, db)
        response = answer_query_base(query, db, chatbot)

        # Estilo customizado com margens
        custom_style = """
        <style>
            .custom-text {
                margin: 20px 50px; /* Ajusta as margens superior/inferior e esquerda/direita */
                padding: 10px;
                background-color: #161A21; /* Fundo suave */
                border-radius: 5px; /* Bordas arredondadas */
                font-family: Arial, sans-serif;
                font-size: 14px;
                line-height: 1.5; /* Melhor espaçamento entre linhas */
                color: #FFFFFF; /* Fonte preta */
            }
        </style>
        """

        # Adicionar CSS ao Streamlit
        st.markdown(custom_style, unsafe_allow_html=True)

        # Exibir contexto recuperado
        st.subheader("Contexto Recuperado")
        st.markdown(f'<div class="custom-text">{context}</div>', unsafe_allow_html=True)

        # Exibir resposta gerada
        st.subheader(f"Petição inicial sobre '{query}':")
        st.markdown(f'<div class="custom-text">{response}</div>', unsafe_allow_html=True)