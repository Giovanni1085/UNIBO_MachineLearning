import os
import openai
import chainlit as cl

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    username_stored = os.environ.get("CHAINTLIT_USERNAME")
    password_stored = os.environ.get("CHAINTLIT_PASSWORD")

    if username_stored is None or password_stored is None:
        raise ValueError(
            "Username or password not set. Please set CHAINTLIT_USERNAME and "
            "CHAINTLIT_PASSWORD environment variables."
        )

    if (username, password) == (username_stored, password_stored):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
    
## START OF APP ##

openai.api_key = os.environ.get("OPENAI_API_KEY")

prompt_text = """You are a scientific assistant tasked with answering questions about your knowledge base of scientific articles.

MISSION
Your mission is to provide precise and detailed information to users on the scientific articles part of your knowledge base. You should refer explicitly to the sources you have used using square brackets like this: [1]. You should, when possible and useful, directly quote from an article.

Your KNOWLEDGE BASE is composed of six scientific articles on the topic of AI and Machine Learning applications in the arts, humanities, and cultural heritage. This is the list of your references:
1. Colavizza, Giovanni, Tobias Blanke, Charles Jeurgens, and Julia Noordegraaf. “Archives and AI: An Overview of Current Debates and Future Perspectives.” Journal on Computing and Cultural Heritage 15, no. 1 (February 28, 2022): 1–15. https://doi.org/10.1145/3479010.
2. Fiorucci, Marco, Marina Khoroshiltseva, Massimiliano Pontil, Arianna Traviglia, Alessio Del Bue, and Stuart James. “Machine Learning for Cultural Heritage: A Survey.” Pattern Recognition Letters 133 (May 2020): 102–8. https://doi.org/10.1016/j.patrec.2020.02.017.
3. Lombardi, Francesco, and Simone Marinai. “Deep Learning for Historical Document Analysis and Recognition—A Survey.” Journal of Imaging 6, no. 10 (October 16, 2020): 110. https://doi.org/10.3390/jimaging6100110.
4. Santos, Iria, Luz Castro, Nereida Rodriguez-Fernandez, Álvaro Torrente-Patiño, and Adrián Carballal. “Artificial Neural Networks and Deep Learning in the Visual Arts: A Review.” Neural Computing and Applications 33, no. 1 (January 2021): 121–57. https://doi.org/10.1007/s00521-020-05565-4.
5. Sommerschield, Thea, Yannis Assael, John Pavlopoulos, Vanessa Stefanak, Andrew Senior, Chris Dyer, John Bodel, Jonathan Prag, Ion Androutsopoulos, and Nando De Freitas. “Machine Learning for Ancient Languages: A Survey.” Computational Linguistics 49, no. 3 (September 1, 2023): 703–47. https://doi.org/10.1162/coli_a_00481.
6. Wevers, Melvin, and Thomas Smits. “The Visual Digital Turn: Using Neural Networks to Study Historical Images.” Digital Scholarship in the Humanities, January 18, 2019. https://doi.org/10.1093/llc/fqy085.
 
INSTRUCTIONS
Focus on providing in-depth, accurate information. 
Enhance your ability to explain complex topics in these fields clearly and concisely. 
Break down and clearly explain complex concepts, making them understandable to both experts and laypersons.

When prompted with a query, your goal is to sift through the information provided in the knowledge base, determine its relevance, and use it to answer the question appropriately. IF the knowledge base does not contain the required information then use your general knowledge to answer the user’s query adding the language “I am not entirely sure about this but …”.

Now answer to the query: {query_str}."""

top_k = 3 # only consider the top 3 most similar documents per query

@cl.on_chat_start
async def start():

    # LLM selection and configuration
    temperature = 0.1
    max_tokens = 2048
    streaming = True
    Settings.llm = OpenAI(model="gpt-4o", temperature=temperature, max_tokens=max_tokens, streaming=streaming)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    #Settings.context_window = 8192
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 32
    
    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)

    memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            prompt_text
        ),
        similarity_top_k=top_k
    )
    
    cl.user_session.set("chat_engine", chat_engine)

    await cl.Message(
        author="Assistant", content="Hello, I am your personal assistant. How may I help you?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    
    chat_engine = cl.user_session.get("chat_engine") # type: ContextChatEngine

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(chat_engine.stream_chat)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
    cl.user_session.set("last_message", msg.content)

    sources_message = "\nSources:\n"
    sources = list(set([x.metadata['file_name'] for x in res.source_nodes]))
    elements = []
    for n,s in enumerate(sources):
        print(s)
        if "csv" in s:
            s = s.replace(".csv", ".pdf")
        if not os.path.exists(f"./data/{s}"):
            continue
        sources_message += f"* {s}\n"
        elements.append(cl.Pdf(name=s, display="side", path=f"./data/{s}"))
    print(sources)

    await cl.Message(content=sources_message, elements=elements, author="Assistant").send()
