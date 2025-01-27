import os
import gradio as gr
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings




def load_db_vectors():

    Openai_Embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )

    # path to local db vectors
    filepath = os.path.join(os.getcwd(), "indexing", "db_vectors_listings")

    text_db = Chroma(
        embedding_function=Openai_Embeddings,
        collection_name="openai",
        persist_directory=filepath,
    )

    return text_db


def llm_text_retrieval(user_query):

    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=2000)

    text_generation = RetrievalQA.from_chain_type(
        llm=llm_openai,
        chain_type="stuff",
        retriever=load_db_vectors().as_retriever(search_kwargs={"k": 4}),
    )

    response = text_generation.invoke(user_query)["result"]

    return response


def collect_preferences(
    bedrooms: int,
    bathrooms: int,
    location: str,
    budget: float,
    additional_requirements: str,
    min_house_size: int,
):

    # buyers preferences as user_query
    query = f"""
		 bedrooms: {bedrooms},
		"bathrooms":{bathrooms},
		"location": {location},
		"budget":   {budget},
		"min_house_size": '{min_house_size},
		"additional_requirements": {additional_requirements}
		"""
    return query


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# HomeMatch")

    gr.Markdown(
        """We’re here to help you find the perfect home! Please provide your preferences below to make sure we match you with the best options
				Once you provide your preferences, we’ll help you find the best matches that suit your needs!"""
    )

    # Input for structured questions (query user)
    bedrooms = gr.Number(label="Number of Bedrooms", value=None)
    bathrooms = gr.Number(label="Number of Bathrooms", value=None)
    min_house_size = gr.Number(label="minimum house size (in Sqft)", value=None)
    location = gr.Textbox(label="Preferred Location", placeholder="e.g., Los Angeles")
    budget = gr.Number(label="Budget (in USD)", value=None)
    additional_requirements = gr.Textbox(
        label="Additional Requirements",
        placeholder="""e.g., garden, swimming pool, garage, Which transportation options are important to you? What are most important things for you in choosing this property?, etc..  """,
    )

    # Output
    user_query = gr.Textbox(label="Collecting Buyer Preferences", visible=False)

    llm_response = gr.Textbox(label="AI HomeMatch Assistant", visible=True)

    # Submit button
    submit_btn = gr.Button("Submit")

    submit_btn.click(
        fn=collect_preferences,
        inputs=[
            bedrooms,
            bathrooms,
            location,
            budget,
            additional_requirements,
            min_house_size,
        ],
        outputs=user_query,
    )

    submit_btn.click(fn=llm_text_retrieval, inputs=user_query, outputs=llm_response)


if __name__ == "__main__":
    # Launch the app
    demo.launch(share=False)
