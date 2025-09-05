import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import  ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

# step 1 - indexing(document ingestion)
video_id = "-HzgcbRXUK8"

try:
    ytt_api = YouTubeTranscriptApi()
    transcript_data = ytt_api.fetch(video_id, languages=['en'])
    transcript = " ".join(snippet.text for snippet in transcript_data)
    print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")

# step 1 - text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
chunks = splitter.create_documents(transcript)
print(chunks)


