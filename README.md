# HomeMatch : Personalized Real Estate Agent
The goal is to create a personalized experience for each buyer, making the property search process more engaging and tailored to individual preferences. This application leverages large language models LLMs and vector databases to transform standard real estate listings into personalized narratives that resonate with potential buyers' unique preferences and needs.

### Core Components of HomeMatch
Understanding Buyer Preferences:
- Buyers will input their requirements and preferences, such as location, property type, budget, amenities, and lifestyle choices.
- The application uses LLMs to interpret these inputs in natural language, understanding nuanced requests beyond basic filters.
### Integrating with a Vector Database:
- Connect "HomeMatch" with a vector database, where all available property listings are stored.
- Utilize vector embeddings to match properties with buyer preferences, focusing on aspects like neighborhood vibes, architectural styles, and proximity to specific amenities
### Personalized Listing Description Generation:
- For each matched listing, we use an LLM to rewrite the description in a way that highlights aspects most relevant to the buyer’s preferences.
- Ensure personalization emphasizes characteristics appealing to the buyer without altering factual information about the property.
### Listing Presentation:
- Output the personalized listing(s) as a text description of the listing.

# Installation


# Configuration Note
Before running the experiments, make sure to configure the following:

Environment Variables: Set the following environment variables if you are using OpenAI LLMs and embedding models:
OPENAI_API_KEY: Required for accessing OpenAI LLMs, Required for accessing OpenAI embedding models.

Store your 

