db_vector_dir_path := ~/HOMEMATCH/data_storage/

# Path to the file containing the listings
filepath := ~/HOMEMATCH/data_storage/listings.txt

indexing:
	@echo "----------------------------------------"
	@echo "Indexing listings to vector database..."
	@echo "Path to the file containing the listings: $(filepath)"
	@echo "Directory path to store vector database: $(db_vector_dir_path)"
	@echo "Running the Python script to index listings..."
	@echo "----------------------------------------"
	python  storing_listings_to_vector_db.py  --path_to_generate_listings_txt_file  $(filepath)  --dirpath_to_store_db_vectors  $(db_vector_dir_path)
