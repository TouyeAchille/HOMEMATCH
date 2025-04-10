import os
import logging
import argparse
import dotenv
import mlflow
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.caches import BaseCache

#ChatOpenAI.__annotations__["cache"] = BaseCache
#ChatOpenAI.model_rebuild()
# Load environment variables from .env file
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Create a new MLflow Experiment
mlflow.set_experiment("homematch")

@mlflow.trace(span_type="generate_listings", name="generate_listings")
def generate_listings(args) -> None:

    """
    Generate real estate listings using a Large Language Model (LLM) based on the provided text prompt.
    Args:
        args (argparse.Namespace): Command-line arguments containing the input prompt and output directory.
    Returns:
        response (str): The generated real estate listings.
    """

    # Ensure API key is set 
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    chain = model | StrOutputParser()

    response = chain.invoke(input=args.input_text_prompt)

    # store generate listing in txt file
    filepath = os.path.join(args.dirpath_to_save_output_artifact, "listings.txt")


    with open(filepath, "w") as f:
        f.write(response)

    return response


def parse_args():

    parser = argparse.ArgumentParser(
        description="Provide text prompt to Large Language Model (LLM) to generate at least 10 diverse and realistic real estate listings containing facts about the real estate.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--dirpath_to_save_output_artifact",
        type=str,
        help="provide directory path to store generate real estate listings",
        required=True,
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

    print("\n")

    print("*" * 40)
    logger.info("Start generate listings")
    print("*" * 40)

    print("\n")

    # parse args
    args = parse_args()

    # run main function
    resp = generate_listings(args)
    print(resp)

    # add space in logs
    print("\n")

    print("*" * 40)
    logger.info("End generate listings")
    print("*" * 40)
