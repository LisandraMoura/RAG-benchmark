# Bibliotecas necessárias para Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import os
from getpass import getpass
import pandas as pd

# Caminhos dos arquivos
CSV_PATH = "/content/amostra.csv"
CHROMA_PATH = "/content/Chroma1"

# Função para carregar documentos de um CSV
def load_documents(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"O arquivo CSV não foi encontrado: {csv_path}")
    csv_loader = CSVLoader(csv_path)
    documents = csv_loader.load()
    print(f"Carregados {len(documents)} documentos.")
    return documents

# Divisor de texto recursivo
def split_documents(documents: list[Document], chunk_size=800, chunk_overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Divididos em {len(texts)} chunks.")
    return texts

# Função para carregar embeddings
def get_embedding_function():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# Função para adicionar chunks ao Chroma
def add_to_chroma(chunks: list[Document], chroma_path: str):
    embedding_fn = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_fn)
    db.add_documents(chunks)
    print(f"Persistência concluída em {chroma_path}.")
    return db

# Pipeline principal
if __name__ == "__main__":
    # Carregar documentos
    documents_csv = load_documents(CSV_PATH)

    # Dividir documentos
    chunks = split_documents(documents_csv)

    # Indexar no Chroma
    chroma_db = add_to_chroma(chunks, CHROMA_PATH)

    # Teste de leitura
    print(f"Chunk de exemplo:\n{chunks[0]}")

    # Consulta no Chroma
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    query = "Me fale sobre o Seguro de pessoas"
    results = db.similarity_search_with_score(query, k=5)
    print("Resultados da consulta:", results)
