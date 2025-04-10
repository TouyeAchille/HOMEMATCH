
   How to run this code :

1 - Start create and activate virtual enviromement 
   before use command below, you need to have miniconda or anaconda installed
      - conda env create -f conda.yaml
      - conda activate homematch_venv

2 set you openai api key in the file .env      

3 -  run Homatch app
        go to directory  homematch_app 
        then run : ``python HomeMatch.py``
        then enter your buyer preference and click submit

4- Optionally : 

if you just want to check the rag pipeline
   - you can run command : ``python main.py``
   
   - you can also decide to change prompt template input, used to generate listing :
                go to the folder `generate_listings`, then change prompt in the file input_prompt.txt
                then run pipeline  with command `python main.py` 

   - you can also decide to change user query input, used to generate matching :
                go to the folder `retrieval_generation`, then change query in the file query.txt
                then run pipeline  with command `python main.py`              

