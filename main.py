import os
import openai
from dotenv import load_dotenv
from src.vector import *
import streamlit as st



load_dotenv()
os.environ['VOYAGE_API_KEY'] = ('VOYAGE_API_KEY')
openai.api_key = os.getenv("OPENAI_API_KEY")


# Título da aplicação
st.title("Recuperação com Geração de Resposta (RAG)")

# Escolher o tipo de RAG
choice = st.sidebar.selectbox("Escolha o Tipo de RAG", ["Basic", "Contextual Embedding", "Outros"])

# Função para carregar o documento e inicializar o VectorDB
def load_vector_db():
    with open('data/fundamentos.json', 'r') as f:
        fundamentos_data = json.load(f)
    db = VectorDB("fundamentos")
    db.load_data(fundamentos_data)
    return db

# Função para recuperar o contexto
def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

# Função para responder a uma consulta usando OpenAI
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
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=2500,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content'], context

# Se a opção escolhida for "Basic"
if choice == "Basic":
    st.header("Basic Retrieval-Augmented Generation (RAG)")

    # Carregar o VectorDB
    db = load_vector_db()

    # Entrada de consulta do usuário
    query = st.text_input("Digite sua consulta:")

    if st.button("Obter Resposta"):
        if query:
            with st.spinner("Buscando contexto e gerando resposta..."):
                # Recuperar contexto e gerar resposta
                response,context = answer_query_base(query, db)

                # contexto recuperado
                st.subheader("Contexto recuperado:")
                st.write(context)

                # resposta llm
                st.subheader("Resposta:")
                st.write(response)
        else:
            st.warning("Por favor, insira uma consulta.")

# Se a opção escolhida for "Contextual Embedding"
elif choice == "Contextual Embedding":
    st.header("RAG com Embeddings Contextuais")
    st.write("Essa opção está em desenvolvimento. Aqui você pode usar embeddings mais específicos ou métodos avançados de contextualização.")

# Se a opção escolhida for "Outros"
elif choice == "Outros":
    st.header("Outras Estratégias de RAG")
    st.write("Essa opção está em desenvolvimento. Pode incluir abordagens híbridas ou outras variações de Recuperação com Geração de Resposta.")
