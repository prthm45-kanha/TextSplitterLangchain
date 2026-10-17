from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""
Artificial Intelligence (AI) is a technology that enables machines to perform tasks that usually require human intelligence. It is used in healthcare, education, finance, and transportation. Machine Learning is a subset of AI that allows computers to learn patterns from data and make predictions.
Large Language Models (LLMs) can understand and generate human-like text. However, they have a limited context window and may not always provide accurate information. Retrieval-Augmented Generation (RAG) helps solve this problem by retrieving relevant information from external documents and providing it to the LLM as context.
Text chunking is an important step in a RAG pipeline. It divides large documents into smaller pieces that can be processed and retrieved efficiently. Chunk size determines the approximate length of each chunk, while chunk overlap preserves some shared text between consecutive chunks to maintain context.
LangChain provides text splitters such as RecursiveCharacterTextSplitter to divide text into manageable chunks. These chunks can be converted into embeddings and stored in a vector database such as FAISS or Chroma. When a user asks a question, the system retrieves relevant chunks and passes them to the LLM to generate a more accurate and context-aware response.
"""
splitter=RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=0,
    
)
chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)