from langchain_text_splitters import CharacterTextSplitter



splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result=splitter.split_text(text)
print(result)