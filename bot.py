import constants
import os
import sys
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]

#loader = DirectoryLoader(".", glob="*.txt")
loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

#it merges the local and outside data
print(index.query(query, llm=ChatOpenAI()))

