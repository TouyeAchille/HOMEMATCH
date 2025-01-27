import os
import logging
import argparse

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args) -> None:

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    chain = model | StrOutputParser()

    response = chain.invoke(input=args.input_prompt)

    # store generate listing in txt file
    filepath = os.path.join(os.getcwd(), "generate_listings.txt")

    with open(filepath, "w") as f:
        f.write(response)

    return response


def parse_args():

    parser = argparse.ArgumentParser(
        description="Provide text prompt to Large Language Model (LLM) to generate at least 10 diverse and realistic real estate listings containing facts about the real estate.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_prompt",
        type=str,
        help="provide text to LLM to generate real estate listings",
        required=True,
    )

    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":

    print("\n")

    print("*" * 40)
    logger.info("Start generate listings")
    print("*" * 40)

    print("\n")

    # parse args
    args = parse_args()

    # run main function
    resp = go(args)
    print(resp)

    # add space in logs
    print("\n")

    print("*" * 40)
    logger.info("End generate listings")
    print("*" * 40)
