import os
import logging
import argparse

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args)-> None:
	# instantiate  embeddings models 

	listing_loader=TextLoader(file_path=args.path_to_listings_txt_file)

	listings=listing_loader.load()
	
	logger.info(" text emdedding using openai embeddings")
	Openai_Embeddings = OpenAIEmbeddings(
			model="text-embedding-3-small", 
			api_key=os.getenv("OPENAI_API_KEY"),
			show_progress_bar=True
		)

	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
	docs = text_splitter.split_documents(listings)

	
	# stores listing and emddeding listing to vector db (chroma)
	listing_vectorstore_db = Chroma.from_documents(
			docs,
			Openai_Embeddings,
			collection_name="openai",
			persist_directory="./db_listings_vectors"
		)
	

	print("\n")

	print('total numbers of listing store:', len(docs))

	print("*" * 30)

	print("\n")

	print ("sample listing store in chroma db:", docs[0])



def parse_args():

	parser = argparse.ArgumentParser(
		description="Provide text prompt to Large Language Model (LLM) to generate at least 10 diverse and realistic real estate listings containing facts about the real estate.",
		fromfile_prefix_chars="@",
	)

	parser.add_argument(
		"--path_to_listings_txt_file",
		type=Path,
		help="provide text to LLM to generate real estate listings",
		required=True,
	)

	args = parser.parse_args()

	return args


# run script
if __name__ == "__main__":

	print("*" * 60)
	logger.info("Start store listings")
	print("*" * 60)

	print("\n")

	# parse args
	args = parse_args()
	# run main function
	resp=go(args)
	print(resp)

	# add space in logs
	print("\n")

	print("*" * 60)
	logger.info("End store listings")
	print("*" * 60)
