import os
import openai
from dotenv import load_dotenv

load_dotenv()

os.environ['VOYAGE_API_KEY'] = ('VOYAGE_API_KEY')
openai.api_key = os.getenv("OPENAI_API_KEY")

'''Inicia a Vector DB Class'''

import os
import pickle
import json
import numpy as np
import openai

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



'''Basic RAG'''



from hugchat import hugchat
from hugchat.login import Login
import time
import json

# Configurações de login
EMAIL = "hdsdosol@gmail.com"
PASSWD = "Lisa2210@"
cookie_path_dir = "./cookies/"  # O diretório onde os cookies serão salvos

# Login no HuggingFace
sign = Login(EMAIL, PASSWD)
cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

# Cria o ChatBot com cookies obtidos
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

# Função para chamar o modelo via HuggingChat
def call_huggingchat(prompt, model_name, retries=3, delay=20):
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

# Carregar o documento do seu caminho específico
with open('/home/lisamenezes/RAG-benchmark/data/fundamentos-all.json', 'r') as f:
    fundamentos_data = json.load(f)

# Inicializar o VectorDB com seus dados
db = VectorDB("fundamentos")
db.load_data(fundamentos_data)

def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    Você é um assistente juridico que responde a seguinte pergunta: 
    <pergunta>
    {query}
    </pergunta>
    Você tem acesso aos seguintes documentos, que devem fornecer contexto à medida que responde à consulta:
    <contexto>
    {context}
    </contexto>
    Por favor, permaneça fiel ao contexto subjacente e só se desvie dele se tiver 100% de certeza de que já sabe a resposta. 
    Responda à pergunta agora e evite fornecer preâmbulos como 'Aqui está a resposta', etc.
    """
    
    response = call_huggingchat(prompt, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct")
    return response


'''Exemplo de uso'''  



# Exemplo de uso para realizar uma consulta e gerar uma resposta
query = "posse provisória"
results, context = retrieve_base(query, db)

print("Contexto Recuperado:")
print(context)

# Gerar o texto usando o LLM
response = answer_query_base(query, db)

print("\nTexto Gerado pelo LLM:")
print(response)




'''Validação'''



import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Função para calcular a similaridade de cosseno entre embeddings
def cosine_similarity_evaluation(embeddings_1, embeddings_2):
    return cosine_similarity(embeddings_1, embeddings_2)

# Função para carregar dados do JSON
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Função para gerar embeddings para os dados de teste
def generate_embeddings(data, db):
    texts = [f"Heading: {item['chunk_heading']}\n\nChunk Text: {item['text']}" for item in data]
    embeddings = []
    start_time = time.time()
    
    for text in texts:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / len(data)
    return np.array(embeddings), avg_time

# Carregar os dados de teste
test_data = load_json('/home/lisamenezes/RAG-benchmark/data/fundamentos-train.json')

# Inicializar o banco de dados de vetores e gerar embeddings de teste
db = VectorDB("fundamentos")
db.load_db()  # Carrega os embeddings previamente salvos

# Gerar embeddings para os dados de teste
print("Gerando embeddings para os dados de teste...")
test_embeddings, avg_embedding_time = generate_embeddings(test_data, db)
print(f"Tempo médio de geração dos embeddings: {avg_embedding_time:.4f} segundos por exemplo")

# Avaliar similaridade entre embeddings da base e dados de teste
print("Calculando similaridade de cosseno...")
db_embeddings = np.array(db.embeddings)
similarity_scores = cosine_similarity_evaluation(test_embeddings, db_embeddings)

# Determinar a assertividade do modelo
correct_matches = 0
similarity_threshold = 0.90

for i, scores in enumerate(similarity_scores):
    # Verifica se há pelo menos um score acima do limiar de similaridade
    if np.any(scores >= similarity_threshold):
        correct_matches += 1
        # Visualizar o Chunk, a Reference e o Gerado
        print(f"\n Gerado: {test_data[i]['text']}")
        top_match_idx = np.argmax(scores)
        print(f"Referencia: {db.metadata[top_match_idx]['text']}")
        print(f"Similarity Score: {scores[top_match_idx]:.4f}")

accuracy = correct_matches / len(test_data)
print(f"Assertividade do modelo: {accuracy * 100:.2f}%")

# Exibir resultados
print(f"Número de correspondências corretas: {correct_matches} de {len(test_data)}")


'''validação 2'''


import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Função para calcular a similaridade de cosseno entre embeddings
def cosine_similarity_evaluation(embeddings_1, embeddings_2):
    return cosine_similarity(embeddings_1, embeddings_2)

# Função para carregar dados do JSON
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Função para gerar embeddings para os dados de teste
def generate_embeddings(data, db):
    texts = [f"Heading: {item['chunk_heading']}\n\nChunk Text: {item['text']}" for item in data]
    embeddings = []
    start_time = time.time()
    
    for text in texts:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / len(data)
    return np.array(embeddings), avg_time

# Carregar os dados de teste
test_data = load_json('/home/lisamenezes/RAG-benchmark/data/fundamentos-train.json')

# Inicializar o banco de dados de vetores e gerar embeddings de teste
db = VectorDB("fundamentos")
db.load_db()  # Carrega os embeddings previamente salvos

# Gerar embeddings para os dados de teste
print("Gerando embeddings para os dados de teste...")
test_embeddings, avg_embedding_time = generate_embeddings(test_data, db)
print(f"Tempo médio de geração dos embeddings: {avg_embedding_time:.4f} segundos por exemplo")

# Avaliar similaridade entre embeddings da base e dados de teste
print("Calculando similaridade de cosseno...")
db_embeddings = np.array(db.embeddings)
similarity_scores = cosine_similarity_evaluation(test_embeddings, db_embeddings)

# Determinar a assertividade do modelo
correct_matches = 0
similarity_threshold = 0.75

for i, scores in enumerate(similarity_scores):
    # Verifica se há pelo menos um score acima do limiar de similaridade
    if np.any(scores >= similarity_threshold):
        correct_matches += 1

accuracy = correct_matches / len(test_data)
print(f"Assertividade do modelo: {accuracy * 100:.2f}%")

# Exibir resultados
print(f"Número de correspondências corretas: {correct_matches} de {len(test_data)}")
