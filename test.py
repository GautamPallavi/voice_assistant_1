from PyPDF2 import PdfReader
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from typing_extensions import Concatenate
import re
os.environ["OPENAI_API_KEY"]="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"
pdfreader=PdfReader("/content/123.pdf")
raw_text=""
for i,page in enumerate(pdfreader.pages):
  content=page.extract_text()
  if content:
    raw_text+=content
pattern=r'[^a-zA-Z0-9\s]'
cleaned_text = re.sub(pattern, '', raw_text)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=10
)
text=text_splitter.split_text(cleaned_text)
embeddings=OpenAIEmbeddings()
search=FAISS.from_texts(text,embeddings)
query="what is Multimodal and interdisciplinary composition"
docs_and_score=search.similarity_search_with_score(query)
score=docs_and_score[0][1]
if score<0.5:
  chain=load_qa_chain(OpenAI(),chain_type="stuff")
  doc=search.similarity_search(query)
  result=chain.run(input_documents=doc,question=query)
  print(result)
else:
  from openai import OpenAI
  client = OpenAI(
    api_key='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'
)
  response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "you are a helpful and friendly chatbot designed to provide users with informative and engaging conversation. Your responses should be concise, easy to understand, and friendly. If you encounter a question that you don't have enough information on, politely suggest that the user might want to look for the answer online or consult a professional. Remember, you should never ask for nor reveal personal information in any conversation. Your main goal is to assist and entertain users to the best of your ability within these guidelines and give the generic answer.and if the greeting message are there then you always greet"},
                {"role": "user", "content":query}
            ]
        )
  result=response.choices[0].message.content.strip()
  print(result)