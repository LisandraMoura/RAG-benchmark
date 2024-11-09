## Pesquisa Jurídica
#### A) Justificativa: 
Profissionais jurídicos podem usar RAG para extrair rapidamente jurisprudências, estatutos ou textos jurídicos relevantes, agilizando o processo de pesquisa e garantindo uma análise jurídica mais abrangente, seguindo essa ideia, optamos por usar RAG nesse mesmo contexto, para extrair a constituição, código civil, código penal e código do consumidor.

#### B) Proposta: 
A ideia é trazer essas informações de forma clara e eficiente destes extensos documentos, facilitando a visualização de leis, propostas políticas entre outros aspectos jurídicos. Assim, trazendo informação de maneira eficiente e fácil.

C) Método proposto:
Pré-Processamento dos Documentos:
Separação dos documentos por seções: Dividir documentos extensos em seções menores, para o RAG trabalhar de forma mais segmentada e eficiente.
Indexação dos dados: Organizar as seções de maneira eficiente por termos específicos.
Anotação de dados: Marcar trechos relevantes para cada tipo de consulta, como jurisprudência.
Testar com diferentes propostas implementadas:
Implementar e testar o ARAGOG: A ideia é comparar essa técnica com o RAG clássico da nossa implementação.
Implementar e testar o Optimizing Query Generation for Enhanced Document Retrieval in RAG: Comparando com o RAG clássico e com o ARAGOG.
Métricas: Mostrando latência, tempo de recuperação e assertividade. 
Configuração do RAG:
Base de recuperação: Integrar o RAG a uma base de dados específica dos documentos indexados, como o código penal por exemplo. Assim, o RAG pode recuperar seções diretamente de um corpus antes de gerar a resposta.
Modelo de Geração/LLM: Implementar um modelo como o GPT-3.5 ou o T5, treinado para responder a perguntas específicas sobre temas jurídicos.
Aprimoramento da Recuperação: Utilizar embeddings especializados na área jurídica para otimizar a semântica de recuperação dos documentos.
Aplicação de Questionamentos e Geração de Respostas:
Processamento de Perguntas e Consultas: O usuário pode fazer perguntas, como “Qual é a responsabilidade civil do fornecedor pelo produto?” ou “Quais são os projetos de leis mais recentes do partido X?”.
Geração de respostas com base na recuperação: O RAG busca essas informações e gera uma resposta adaptada ao contexto, referenciado nos documentos imputados.
Comparar performances com outras alternativas como:
Busca Semântica com Embeddings Simples: Aplicação de embeddings para localizar os trechos mais relevantes sem a geração de respostas. Avaliar a precisão dos resultados comparando com a capacidade do RAG de interpretar e gerar respostas.
Modelos de Pergunta-Resposta Baseados em Documentos (e.g., BERT-QA): Utilizar um modelo de perguntas e respostas diretamente nos documentos jurídicos. Esta técnica poderá ser menos eficaz para perguntas mais complexas, mas será útil para fins de comparação.
Análise de Recuperação por Similaridade de Contexto: Utilizar algoritmos que analisam a similaridade entre a pergunta e trechos do documento, sem gerar uma resposta adaptada. Esta técnica pode ser útil para identificar quais seções são mais frequentemente recuperadas.

Introdução: 
(a) Justificativa 
(b) Objetivos 
Proposta de Pesquisa:
(a) Método proposto 
(b) Plano de Trabalho


Referências bibliográficas:

ANTHROPIC. *The Best RAG Technique Yet: Anthropics Contextual Retrieval and Hybrid Search*. Disponível em: <https://levelup.gitconnected.com/the-best-rag-technique-yet-anthropics-contextual-retrieval-and-hybrid-search-62320d99004e>. Acesso em: 4 nov. 2024.

TOWARDS AI. *The Best RAG Stack to Date*. Disponível em: <https://pub.towardsai.net/the-best-rag-stack-to-date-8dc035075e13>. Acesso em: 4 nov. 2024.

ZHANG, H.; LIU, X.; YAO, L.; WANG, Y. *Searching for Best Practices in Retrieval-Augmented Generation*. Disponível em: <https://arxiv.org/abs/2407.01219>. Acesso em: 4 nov. 2024.

FAIRBANKS, J. *Local RAG*. Disponível em: <https://github.com/jonfairbanks/local-rag>. Acesso em: 4 nov. 2024.

DATA SCIENCE ACADEMY. *Como RAG (Retrieval-Augmented Generation) Funciona Para Personalizar os LLMs?*. Disponível em: <https://blog.dsacademy.com.br/como-rag-retrieval-augmented-generation-funciona-para-personalizar-os-llms/>. Acesso em: 4 nov. 2024.

LIU, W.; SUN, Y.; ZHAO, Q.; HU, J. *Optimizing Query Generation for Enhanced Document Retrieval in RAG*. Disponível em: <https://arxiv.org/abs/2407.12325>. Acesso em: 4 nov. 2024.

ARAGOG: Advanced RAG Output Grading - https://arxiv.org/abs/2404.01037 (4-nov-24)
