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
	filepath1=os.path.join(get_original_cwd(), "generate_listings")
	# run generate listings
	subprocess.run(["make", "generate_listings"], cwd=filepath1)



	# Accessing the original working directory for store listing (indexing step)
	filepath2=os.path.join(get_original_cwd(), "indexing")
	# run store listings to chromadb database
	subprocess.run(["make", "indexing"], cwd=filepath2)


	# Accessing the original working directory for retrieval generation 
	filepath3=os.path.join(get_original_cwd(), "retrieval_generation")
      
	#with open("~/HOMEMATCH/retrieval_generation/query.txt") as f:
		#user_query = f.read().strip()  

	# run store listings to chromadb database
	subprocess.run('make matching user_query="$$(cat query.txt)"', cwd=filepath3, shell=True)


if __name__ == "__main__":
    go()

