from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

text_splitter=SemanticChunker(
    HuggingFaceEmbeddings(),breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = """
Online shopping allows customers to buy products online.
Delivery services help customers receive their orders.
Machine learning helps computers learn from data.
Neural networks are used for image recognition.
"""

docs=text_splitter.create_documents([text])
print(len(docs))
print(docs)


# in this we need to use embedding of openAi or something else. HUgging Face can run this 
# but this want to download something which i am not allowing it to do.
#hence this code is correct just we need to use embedding here