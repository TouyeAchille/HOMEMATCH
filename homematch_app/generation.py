import os
from pathlib import Path
import shutil
import logging
import dotenv
import mlflow
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables from .env file
dotenv.load_dotenv()

# logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# create experiment
mlflow.set_experiment("homematch")


def load_db_vectors():

    # Ensure API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")

    Openai_Embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )

    # load to local db vectors
    text_db = Chroma(
        embedding_function=Openai_Embeddings,
        collection_name="openai",
        persist_directory="~/HOMEMATCH/data_storage/db_vectors_listings",
    )

    return text_db

@mlflow.trace(span_type="Chain", name="llm_text_retrieval")
def llm_text_retrieval(user_query):

    system_prompt = """You are a helpful AI assistant. Your task is to identify the best property matches for buyers preferences based solely on the context provided.

    Focus on finding properties that align closely with these preferences and provide the most suitable options
    """

    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=2000, api_key=os.getenv("OPENAI_API_KEY"))

    # Create custom prompts
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

    human_message_prompt = HumanMessagePromptTemplate.from_template(
        """Based on the preferences provided below, recommend the most suitable properties. Use only the given context to ensure relevance and accuracy.  
                                                                    Buyer Preferences: {user_query}  
                                                                    Available Properties (Context): {context}  
                                                                    Provide a clear and concise recommendation highlighting the best matches. If no suitable properties are found, state that no relevant matches are available
                                                                    """
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # Create the retriever
    retriever = load_db_vectors().as_retriever(search_kwargs={"k": 4}).invoke(user_query)

    text_snippets = [text.page_content for text in retriever]

    # Combine text ontexts
    combined_text_context = "\n".join(text_snippets)

    text_generation = chat_prompt | llm_openai | StrOutputParser()

    # Generate a response
    response = text_generation.invoke(
        {"user_query": user_query, "context": combined_text_context}
    )

    return response


