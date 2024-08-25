from fastapi import FastAPI, HTTPException,UploadFile, File,Request, Query
import time
from together import Together
from firecrawl import FirecrawlApp
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
import pytesseract
from PIL import Image
import cohere
import speech_recognition as sr
import faiss
import numpy as np
from nltk.tokenize import word_tokenize
from pathlib import Path
from typing import Optional
import urllib.parse

app = FastAPI()

tesseract_cmd = os.getenv("TESSERACT_PATH", "tessaercat path")
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

llm = Together(api_key=("api key")) #Get a together.ai api key 
co = cohere.Client("api key") # get a cohere api key


def log_time(func):# for measuring the time taken by each function call
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Finished {func.__name__} in {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

async def loading_data(parser):
    result = await parser.aload_data("uploaded_file.pdf")
    return result

async def load_data_pdf(content):
    parser = LlamaParse(
        api_key="api key",# get a LLama parse api key
        result_type="text",
        language="en",
        skip_diagonal_text=True,
        page_seperator="******",
        do_not_unroll_columns=True
    )
    output_path = Path("uploaded_file.pdf")
    with open(output_path, 'wb') as f:
        f.write(content)
    try:
        documents = await loading_data(parser)
    except Exception as e:
        return e
    print(len(documents))
    final_data = ""
    for document in documents:
        data = document.text
        data = "\n".join([line.strip() for line in data.split("\n") if line.strip()])
        data = data.replace('\n', '  ')
        final_data = final_data + data
    return final_data

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # try:
        #     pdf_stream = BytesIO(contents)
        # except Exception as e:
        #     return f"byteio error {e}"
        try:
            extracted_data = await load_data_pdf(contents)
        except Exception as e:
            return f"function error{e}"
        
        return extracted_data
    except Exception as e:
        return e

@log_time
def load_data_video(url):
    pattern = re.compile(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*')
    match = pattern.search(url)
    id=match.group(1) if match else None
    if id ==None:
        return "Invalid URL"
    else:
        try:
            raw_data=YouTubeTranscriptApi.get_transcript(id)
            data = " ".join([entry['text'] for entry in raw_data])
            return data
        except Exception as e:
            return f" Sorry! Right now only English language is supported. Wait for updates. {e}"

@app.post("/uploadYoutubevideo/")
async def upload_video(url:str = Query(..., title="Video URL")):
    extracted_data = load_data_video(url)
    return extracted_data
      
@log_time
def load_data_audio(file):
    output_path = Path("uploaded_audio_file.wav")
    with open(output_path,"wb") as audio_data:
        audio_data.write(file)
    data = ""
    r = sr.Recognizer()
    try:
        with sr.AudioFile("uploaded_audio_file.wav") as source:
            audio_text = r.listen(source)
            try:
                text = r.recognize_google(audio_text)
                data += text 
            except sr.UnknownValueError:
                return "Speech recognition could not understand audio"
            except sr.RequestError:
                return "Could not request results from Google Speech Recognition service"
    except OSError:
        return "Error opening the audio file. Please try in WAV format"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    
    return data

@app.post("/uploadaudio/")
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()
    extracted_data = load_data_audio(contents)
    
    return extracted_data

@log_time
def load_data_text(data):
    return data


@app.get("/uploadtext/")
async def upload_text(data: str = Query(...)):
    extracted_data = load_data_text(data)
    return extracted_data

@log_time
def load_data_ocr(file,file_name):
    file_extension = file_name.split('.')[-1].lower()
    # valid_extensions = ['png', 'jpg', 'jpeg', 'gif']
    # if file_extension not in valid_extensions:
    #     return "Unsupported file type"
    output_path = Path(f"uploaded_image_file.{file_extension}")
    with open(output_path,"wb") as image_data:
        image_data.write(file)
    try:
        image_file = Image.open(output_path)
        image_data = pytesseract.image_to_string(image_file)
    except Exception as e:
        return e
    return image_data

@app.post("/uploadocr/")
async def upload_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    file_name = file.filename
    extracted_data = load_data_ocr(contents,file_name)
    
    return extracted_data


@log_time
def load_data_website(url):
    app = FirecrawlApp(api_key="api") # get a firecrawl api key
    data=app.scrape_url(url,{
            'extractorOptions': {
                'mode': 'llm-extraction'
            }})
    if data["metadata"]["pageStatusCode"]==400:
        print("Either the website doesnt exist or it is temporarily unavailable")
    else:
        return data["content"]
    
@app.post("/uploadwebsite/")
async def upload_website(url: str = Query(...)):
    extracted_data = load_data_website(url)
    return extracted_data

@log_time
def short_summarize_text(data):
    data2=''
    for i in data:
        data2+=i
    try:
        response = llm.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",# you can choose the llm model of your choice(llama 3.1 was a bit unstable)
        messages=[
        {"role": "system", "content": '''IDENTITY and PURPOSE
        You are an expert content summarizer. You take content in and output a Markdown formatted summary using the format below.

        OUTPUT SECTIONS
            Combine all of your understanding of the content into a single, 20-word sentence in a section called ONE SENTENCE SUMMARY:.
            Output the most important points of the content as a list with no more than 15 words per point and no more than 6 points into a section called MAIN POINTS:.
            Extract the most potent takeaway and recommendation into a section called ONE-SENTENCE TAKEAWAY. This should be a 15-word sentence that captures the most important essence of the content.
            
        OUTPUT INSTRUCTIONS
            Create the output using the formatting above.
            You only output human readable Markdown.
            Output numbered lists, not bullets.
            Do not output warnings or notes—just the requested sections.
            Do not repeat items in the output sections.
            Do not start items with the same opening words'''},
        {"role": "user", "content": data2},
        ],
        )
        final_response=(response.choices[0].message.content)
    except Exception as e:
        final_response="Reduce the size of your data and try again" + e.message
    return final_response


@app.post("/shortsummarize/")
async def short_summarize(request: Request):
    try:
        payload = await request.json()
        data = payload.get("data")
        if data is None:
            return "no data provided" 
        data = urllib.parse.unquote(data)
        summarized_data = short_summarize_text([data])
        return summarized_data
    except Exception as e:
        return e

@log_time
def summarize_text(data):
    data1=""
    for i in data:
        data1=data1+i
    try:
        response = llm.chat.completions.create(
            
            model="mistralai/Mistral-7B-Instruct-v0.3",# you can choose the llm model of your choice(llama 3.1 was a bit unstable)
            messages=[
                {"role": "system", "content":'''
    IDENTITY and PURPOSE
    You extract surprising, insightful, and interesting information from text content. 

    Think step-by-step about how to achieve the best possible results by following the steps below.
    STEPS
        Extract a summary of the content in 75-100 words, including who is presenting and the content being discussed into a section called SUMMARY.
        Extract 10-15 of the most surprising, insightful, and/or interesting ideas from the input in a section called IDEAS:.
        Extract 10 of the best insights from the input and from a combination of the raw input and the IDEAS above into a section called INSIGHTS. These INSIGHTS should be fewer, more refined, more insightful, and more abstracted versions of the best ideas.
        Extract 15 of the most surprising, insightful, and/or interesting valid facts about the greater world that were mentioned in the content into a section called FACTS:.
        Extract all mentions of writing, art, tools, projects and other sources of inspiration mentioned by the speakers into a section called REFERENCES. This should include any and all references to something that the speaker mentioned.
        Output a list of the 5 best takeaways from the content in a section called TAKEAWAYS:.
        Extract the 5 of the most surprising, insightful, and/or interesting recommendations that can be collected from the content into a section called RECOMMENDATIONS.

    OUTPUT INSTRUCTIONS
        Write the IDEAS bullets as exactly 15 words.
        Write the RECOMMENDATIONS bullets as exactly 15 words.
        Write the FACTS bullets as exactly 15 words.
        Write the INSIGHTS bullets as exactly 15 words.
        Do not repeat ideas, quotes, facts, or resources.
        Do not start items with the same opening words.
        Ensure you follow ALL these instructions when creating your output.'''},{"role": "user", "content": data1},],)
        final_response=response.choices[0].message.content
    except Exception as e:
        final_response="Reduce the size of your data and try again"+e.message
    return final_response

@app.post("/longsummarize/")
async def long_summarize_text(request: Request):
    try:
        payload = await request.json()
        data = payload.get("data")
        if data is None:
            return "no data provided" 
        data = urllib.parse.unquote(data)
        summarized_data = summarize_text([data])
        return summarized_data
    except Exception as e:
        return e

@log_time
def create_tagline(data):
    data1=""
    for i in data:
        data1=data1+i
    try:
        response = llm.chat.completions.create(
            
            model="mistralai/Mistral-7B-Instruct-v0.3",# you can choose the llm model of your choice(llama 3.1 was a bit unstable)
            messages=[
                {"role": "system", "content":'''
    IDENTITY and PURPOSE
        You have the purpose of reading the entire content and giving providing a suitable tagline for the data.
    
    OUTPUT INSTRUCTIONS
        The tagline should not be more than one line.

 '''},{"role": "user", "content": data1},],)
        final_response=response.choices[0].message.content
    except Exception as e:
        final_response="Reduce the size of your data and try again"+e.message
    return final_response

@app.post("/generate_tagline/")
async def tagline(request: Request):
    try:
        payload = await request.json()
        data = payload.get("data")
        if data is None:
            return "no data provided" 
        data = urllib.parse.unquote(data)
        tagline = create_tagline([data])
        print(tagline)
        return tagline
    except Exception as e:
        return e

# online tabular responses

@log_time
def split_data(data, base_chunk_size=100):
    splitted_data = []
    for document in data:
        words = word_tokenize(document)
        dynamic_chunk_size = max(base_chunk_size, len(words) // 10)# you can also fix your chunk size
        if dynamic_chunk_size > 500:
            dynamic_chunk_size = 300
        chunks = [' '.join(words[i:i + dynamic_chunk_size]) for i in range(0, len(words), dynamic_chunk_size)]
        for chunk in chunks:
            splitted_data.append({"text": chunk})
    return splitted_data

@app.post("/split/")
async def split_text(request: Request):
    body = await request.json()
    data = body.get('data')
    base_chunk_size = body.get('base_chunk_size', 100)  # Default to 100 if not provided
    if not data or not isinstance(data, list):
        return {"error": "Invalid input. 'data' should be a list of strings."}
    
    splitted_data = split_data(data, base_chunk_size)
    return splitted_data

@log_time
def generate_embeddings(data):
    response = co.embed(
        model="embed-english-v3.0",# you can change the model as per your choice
        texts=data,
        input_type="classification"
    )
    return response.embeddings

@app.post("/embeddings/")
async def get_embeddings(request: Request):
    body = await request.json()
    data = body.get('data')
    if not data or not isinstance(data, list):
        return {"error": "Invalid input. 'data' should be a list of strings."}
    
    embeddings = generate_embeddings(data)
    return embeddings

@log_time
def vectorize_data(embeddings):
    embedding_dim = len(embeddings[0])
    if not all(len(embedding) == embedding_dim for embedding in embeddings):
        raise ValueError("Inconsistent embedding dimensions detected")
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings_array = np.asarray(embeddings)
    index.add(embeddings_array) 
    return index, embedding_dim

@app.post("/vectorize/")
async def vectorize_data_endpoint(request: Request):
    body = await request.json()
    splitted_data = body.get('splitted_data')
    # if not splitted_data or not isinstance(splitted_data, list):
    #     return {"error": "Invalid input. 'splitted_data' should be a list of dictionaries with 'text' key."}
    index, embedding_dim = vectorize_data(splitted_data)

    return index, embedding_dim
@log_time
def retriever(query, faiss_index, documents, k=10):
    query_embedding = generate_embeddings([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k)
    result_indices = indices[0].tolist()
    retrieved_documents = [documents[i] for i in result_indices]
    return retrieved_documents

@log_time
def data_check(query, results):
    return True

@log_time
def rerank(query, faiss_index, documents):
    results = retriever(query, faiss_index, documents)
    if data_check(query, results):
        rerank_docs = co.rerank(query=query, documents=results, return_documents=True, top_n=25, model="rerank-english-v2.0")# you can chhose a model of your choice
        top_responses = []
        for result in rerank_docs.results[:1]:
            top_responses.append(result.document.text)
        return top_responses
    else:
        return "Data entered is not of any useful reference to the query"

@app.post("/rerank/")
async def rerank_documents(request: Request):
    body = await request.json()
    query = body.get('query')
    faiss_index = body.get('faiss_index')
    documents = body.get('documents')

    if not query or not isinstance(query, str):
        return {"error": "Invalid input. 'query' should be a string."}
    if not faiss_index or not documents or not isinstance(documents, list):
        return {"error": "Invalid input. 'faiss_index' and 'documents' should be provided and 'documents' should be a list."}

    reranked_documents = rerank(query, faiss_index, documents)
    return reranked_documents


@log_time
def response(data,query):
    new_data=split_data(data)
    print("splitted")
    splitted_data = [entry['text'] for entry in new_data]   
    embeddings= generate_embeddings(splitted_data)
    print("emebeddings done")
    faiss_index, embedding_dim = vectorize_data(embeddings)
    print("vectorised")
    vector_response = rerank(query, faiss_index, splitted_data)
    print("reranked")
    response = llm.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",# choose a model of your choice 
        messages=[
            {"role": "system", "content":''' You are a model whose purpose is:
             1. check if the answer is some kind of an error message - if yes return it as it is
             2. Analyse the question and answer
             3. Leave otu parts of the answer which dont seem relevant to the question
             4. Reframe the answer in  a human readble format.
             5. Dont show in the answer that you had previously recieved some kind of quesiton answer.
             The User will provide you with the input in the form of " Question is ... and the answer is ..."
             '''},
             {"role": "user", "content": f''' Question is {query} and the answer is {vector_response}'''}])
    
    return response.choices[0].message.content

@app.post("/response/")
async def generate_response(request: Request):
    body = await request.json()
    data = body.get('data')
    query = body.get('query')
    data=urllib.parse.unquote(data)
    data=[data]
    query=urllib.parse.unquote(query)
    if not data or not isinstance(data, list):
        return {"error": "Invalid input. 'data' should be a list of strings."}
    if not query or not isinstance(query, str):
        return {"error": "Invalid input. 'query' should be a string."}

    try:
        result = response(data, query)
        return result
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
