
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
text="""
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return self.name

    def isPassing(self):
        return self.grade


# Using the class
s1 = Student("Pratham", 21, "A")

print(s1.get_details())  # Pratham
print(s1.isPassing())   # A
"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

result=splitter.split_text(text)

print(len(result))

print(result[1])