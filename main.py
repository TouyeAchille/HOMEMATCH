import os
import subprocess
import hydra
import mlflow 
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# This automatically read the config yaml file when you run your application
@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig):

    # Accessing the original working directory for generate listing
	filepath1=os.path.join(get_original_cwd(), "dataset")

	# run generate listings
	subprocess.run(["make", "generate_listings"], cwd=filepath1)

	# Accessing the original working directory for store listing
	filepath2=os.path.join(get_original_cwd(), "indexing")

	# run store listings to chromadb database
	subprocess.run(["make", "store_listings"], cwd=filepath2)





if __name__ == "__main__":
    go()

