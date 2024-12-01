# Bibliotecas necessárias para BM25
import math
import numpy as np
import pandas as pd
import time
from hugchat import hugchat
from hugchat.login import Login

# Classe BM25Okapi
class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self._initialize()

    def _initialize(self):
        nd = {}
        for document in self.corpus:
            frequencies = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
        self._calc_idf(nd)

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        for q in query:
            q_idf = self.idf.get(q, 0)
            for idx, doc in enumerate(self.doc_freqs):
                f = doc.get(q, 0)
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += q_idf * (f * (self.k1 + 1) / denom)
        return scores

# Login no HuggingChat
EMAIL = "hdsdosol@gmail.com"
PASSWD = "Lisa2210@"
cookie_path_dir = "./cookies/"
sign = Login(EMAIL, PASSWD)
cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

# Função para carregar o corpus de um CSV
def load_corpus_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Erro ao carregar o arquivo CSV: {e}")
    print("Colunas disponíveis no CSV:", df.columns)
    text_column = next((col for col in df.columns if "content" in col.lower() or "text" in col.lower()), None)
    if not text_column:
        raise ValueError("Nenhuma coluna de texto encontrada no CSV.")
    return df[text_column].dropna().tolist()

# Função para consulta
def query_rag_with_bm25_and_huggingchat(query_text: str):
    PROMPT_TEMPLATE = """
    Você é um assistente jurídico que responde à seguinte pergunta:
    <pergunta>
    {question}
    </pergunta>
    Você tem acesso aos seguintes documentos, que devem fornecer contexto à medida que responde à consulta:
    <contexto>
    {context}
    </contexto>
    Responda fielmente ao contexto e evite desvios.
    """

    # Tokenizar a consulta
    tokenized_query = query_text.split()

    # Obter scores de relevância
    scores = data_base.get_scores(tokenized_query)

    # Selecionar documentos mais relevantes
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    top_documents = [raw_corpus[i] for i in top_n_indices]

    # Criar o contexto
    context_tex = "\n\n---\n\n".join(top_documents)

    # Formatar o prompt
    prompt = PROMPT_TEMPLATE.format(context=context_tex, question=query_text)

    # Fazer a inferência
    response = chatbot.chat(prompt)

    print("\nResposta gerada:")
    print(response)
    return response

# Pipeline principal
if __name__ == "__main__":
    file_path = "/content/amostra.csv"
    raw_corpus = load_corpus_from_csv(file_path)
    tokenized_corpus = [doc.split() for doc in raw_corpus]
    data_base = BM25Okapi(tokenized_corpus)

    # Exemplo de consulta
    query = "menor de 16 anos"
    response = query_rag_with_bm25_and_huggingchat(query)
    print("\nResposta final:")
    print(response)
