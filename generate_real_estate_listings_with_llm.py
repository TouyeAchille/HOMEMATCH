import os
import logging
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7
    )

    prompt = PromptTemplate.from_template(
        "Generating Real Estate Listings:{text_prompt}"
    )

    chain = prompt | llm | StrOutputParser()

    chat_response = chain.invoke({"text_prompt": args.input_text_prompt})

    # store generate listing in txt file
    with open("output_generate_listing.txt", "w") as fout:
        fout.write(chat_response)

    return chat_response



def parse_args():

    parser = argparse.ArgumentParser(
        description="Provide text prompt to Large Language Model (LLM) to generate at least 10 diverse and realistic real estate listings containing facts about the real estate.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_text_prompt",
        type=str,
        help="provide text to LLM to generate real estate listings",
        required=True,
    )

    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":

    print("*" * 60)
    logger.info("Start generate listings")
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
    logger.info("End generate listings")
    print("*" * 60)
