filepath := ~/HOMEMATCH/data_storage/db_vectors_listings

matching:
	@echo "Running retrieval_generation.py with the following parameters:"
	@echo "Path to DB Vectors: $(filepath)"
	@echo "User Query: $(user_query)"
	@echo "Running the command..."
	@echo "--------------------------------"
	python retrieval_generation.py --path_to_db_vectors $(filepath)  --user_query "$(user_query)"
