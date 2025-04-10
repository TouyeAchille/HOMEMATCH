import gradio as gr
from generation import llm_text_retrieval


def collect_preferences(
    bedrooms: int,
    bathrooms: int,
    location: str,
    budget: float,
    additional_requirements: str,
    min_house_size: int,
):
    """
    Collect user preferences for home search.
    Args:
        bedrooms (int): Number of bedrooms.
        bathrooms (int): Number of bathrooms.
        location (str): Preferred location.
        budget (float): Budget in USD.
        additional_requirements (str): Additional requirements.
        min_house_size (int): Minimum house size in Sqft.
    Returns:
        str: Formatted query string with user preferences.
    """
    # Collecting user preferences
    # bedrooms, bathrooms, location, budget, additional_requirements, min_house_size
    # buyers preferences as user_query
    query = f""" 
		 bedrooms: {bedrooms},
		"bathrooms":{bathrooms},
		"location": {location},
		"budget":   ${budget},
		"min_house_size": {min_house_size} Sqft,
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
    submit_btn = gr.Button("Submit", variant="primary")

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



# run script
if __name__ == "__main__":
    demo.launch(share=False, debug=True)

   