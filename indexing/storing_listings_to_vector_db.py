import os
import shutil
import logging
import argparse

from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args) -> None:

    # load listings
    listings = TextLoader(file_path=args.path_to_generate_listings_txt_file).load()

    # instantiate  embeddings models
    logger.info("text emdedding using openai embeddings")
    Openai_Embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(listings)

    # Define the persist directory to store local db vectors
    filepath = os.path.join(os.getcwd(), "db_vectors_listings")

    # Clear the directory if it exists
    shutil.rmtree(filepath, ignore_errors=True)

    # stores listing and emddeding listing to vector db (chroma)
    logger.info("stores listing and emddeding listing to vector db (chroma)")
    listing_vectorstore_db = Chroma.from_documents(
        docs, Openai_Embeddings, collection_name="openai", persist_directory=filepath
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Indexing real estate listings.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--path_to_generate_listings_txt_file",
        type=Path,
        help="provide generate listings txt file for indexing ",
        required=True,
    )

    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":

    print("\n")

    print("*" * 50)
    logger.info("Start indexing listings")
    print("*" * 50)

    print("\n")

    # parse args
    args = parse_args()

    # run main function
    go(args)

    # add space in logs
    print("\n")

    print("*" * 50)
    logger.info("store indexing listings to db chroma successfully")
    print("*" * 50)
