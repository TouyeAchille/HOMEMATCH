generate_listings:
	python generate_real_estate_listings_with_llm.py --input_text_prompt "$$(cat prompt.txt)"

store_listings:
	python  storing_listings_vector_db.py  --path_to_listings_txt_file ./output_generate_listing.txt
