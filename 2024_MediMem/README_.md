# Topic 2: Long-term Memory Using Entity and Chromadb
The model is structured with a non-parametric memory that stores important elements of a user query as entities in ChromaDB, and retrieves relevant memories based on the user query.

## Installation
Tested on Windows 10 with Python 3.9 and CUDA version 11.6 or higher. And NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 2080 Ti.
The NVIDIA GeForce RTX 3090 alone is not working due to an out of memory error.

The mem0ai library on GitHub is continuously updated, so only version 0.1.7 is allowed. 
Then, go to the mem0ai folder installed in the Anaconda virtual environment and replace the files mem0/memory/main.py, mem0/configs/prompts.py, mem0/embeddings/bge_base.py, mem0/vector_stores/chroma.py, and mem0/llms/phi3.py with the submitted files.
After that, to recognize the replaced LLM, go to mem0/utils/factory.py and add "phi3": "mem0.llms.phi3.Phi3LLM" to the ‘provider_to_class’ dictionary.

```bash
├── mem0
│   ├── client
│   ├── configs
│   │    └─prompts.py
│   ├── embeddings
│   │    └─bge_base.py
│   ├── graphs
│   ├── llms
│   │    └─phi3.py
│   ├── memory
│   │    └─main.py
│   ├── proxy
│   ├── utils
│   │    └─factory.py
│   ├── vector_stores
│   │    └─chroma.py
```

How to Install GPU-Enabled llama-cpp-python (Visual Studio 2019 and CMake must be installed beforehand):

```bash
1-1. set FORCE_CMAKE=1 && set CMAKE_ARGS=-DLLAMA_CUBLAS=on
1-2. If DLLAMA_CUBLAS doesn’t work, use the following alternative: 
     set FORCE_CMAKE=1 && set CMAKE_ARGS=-DGGML_CUDA=on

2-1. pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
2-2. pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir -vv
```
If DLLAMA_CUBLAS fails after running 1-1, you must run 1-2 DGGML_CUDA. 
Then run 2. If you want to see the installation process of 2-1 in detail, run 2-2.
Installing the GPU version may take about 2 hours.

### Other library versions to note
```bash
pip install mem0ai==0.1.7
pip install chromadb==0.5.23
pytorch <= 2.4.1 (for onnxruntime-gpu)
python -m spacy download en
pip install transformers==4.46.3 (under 4.47 version)
```

For all other dependencies, you can install them using the requirements submitted.

`pip install -r requirements.txt`

## Demo
The attached demo folder's 004.db (demo/004.db) is needed to check the user's personal information, and the 004 folder (demo/004) is the chromadb folder that stores the memory content. (The attached folder contains an empty database, provided only to indicate its location.)
1. After running server.py, execute the client.py file once the execution is complete.
2. If you enter a user ID, the chat session will start immediately if the user is already registered. At this point, the 004.db data from the demo folder that was created in advance should be available. (At this time, the name of the database is the user's ID, and if the user is not registered, registration will not occur at this time. You can register it separately using the save_base function in server.py.)
3. During the conversation, enter dialogue sentences and press enter. (To store user information, only sentences related to diseases, symptoms, and medications should be entered. Any content unrelated to these topics will not be saved.)
4. To exit the conversation session, type the keyword "Exit" and press enter.

## Server Client Setup
On the server side, run [server.py]. This will start a server that listens for incoming connections.
On the client side, run [client.py]. This will connect to the server and start a dialogue session.
Websocket url = ‘ws://localhost:8000/LLM’ 


## RAG-BASED MEMORY

<p align="center">
<img src="./images/RAG_based_memory_overview.png"/>
RAG-BASED MEMORY OVERALL STRUCTURE
</p>

<p align="center">
<img src="./images/GUI_OVERVIEW.png"/>  
RAG-BASED MEMORY SYSTEM: GUI OVERVIEW
</p>

## Concept
- Older memories are deleted using the number of mentions and the mentioned date along the Ebbinghaus' forgetting curve. (Forget memroy)
- A small language model (Phi-3-mini-4k-instruct 3.8B) compresses information from the user's words in the conversation and stores it as a vector embedding in chromadb or updates the memory if it already exists. (Extract entity)
- Relevant memories are retrieved from the database using cosine similarity between embeddings of of the memories and the user's query (Search top K related data)
- Related data and entities are checked to see which data the entities are related to using a small language model to decide whether to update or add. (Search related memroy and select the related memory's IDs)
- Currently, the queries are relatively simple, but we are implementing Memorag in preparation for when more complex queries need to be used in the future. (Generate derived query)


## Forget memory

<p align="center">
<img src="./images/forgetting_memory.png" width="750" height="250"/>
FORGETTING CURVE EQUATION
</p>

Here, N serves to make the slope of the graph gradually gentler as days pass and repetitions occur, thus making forgetting happen later over time, as shown in the figure below.
And if it is repeated 6 times, it will go into long-term memory and not be deleted.

- citation: https://www.rebuildingeducation.com/the-algorithm-that-saves-lives/

<p align="center">
<img src="./images/forget_rep.png"/>   
GRAPH CHANGES BASED ON THE NUMBER OF MENTIONS ON DIFFERENT DATES
</p>

The figure below visualizes how the slope of the graph becomes more gentle as the repetition occurs on different dates.

<p align="center">
<img src="./images/graph_change.png"/>    
CHANGE IN GRAPH SLOPE WITH REPETITION
</p>

## PROCESS AND RESULT
The example picture below shows how the query is stored and how the memory is updated. And it shows an example of how the final answer of AI is given.

<p align="center">
<img src="./images/RAG_based_memory_process.png"/>  
RAG-BASED MEMORY PROCESS WITH EXAMPLE
</p>

<p align="center">
<img src="./images/ai_answer.png" width="650" height="200"/>  
AI ANSWER OF EXAMPLE
</p>



## Extract entity
- ### __Dataset__   
1. Medialog dataset (num 100) 
https://huggingface.co/datasets/UCSD26/medical_dialog/blob/main/medical_dialog.py  
2. MSC-self-instruct dataset (num 100)  
https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct

- ### How to evaluate
1. Run `phi3-tool calls_medical.py` to get results for medialog (the ground truth is positive for all examples).
2. Run `phi3-tool calls_common.py` to get results for msc-self-instruct (the ground truth is negative for all examples).
3. From 1 and 2, the results are saved in a text file `eval_file/extract_entity/extract_entity_medical.txt` and `eval_file/extract_entity/extract_entity_common.txt`.
4. You need to visually check the results of each text file to calculate the statistics. The conditions for evaluating this are:

    i. From this file `eval_file/extract_entity/extract_entity_medical.txt` remove all sentences that are not related to the user (i.e. "What is diabetes?": not related / "I have a headache, what should I do": related)  
    ii. In `eval_file/extract_entity/extract_entity_medical.txt` each example is classified as TP (true positive) if all entities extracted are correct, and is classified as FN (false negative) if even one is wrong.  
    iii. From this file `eval_file/extract_entity/extract_entity_common.txt` remove all sentences that are related to medical care (i.e. "I went for a medical checkup yesterday")  
    iv. In `eval_file/extract_entity/extract_entity_common.txt` each example is classified as FP (false positive) if anything is extracted, if nothing is extracted it is classified as TN (true negative).


```bash
├── eval_file
│   ├── extract_entity
│   │   ├──phi3_tool calls_common.py
│   │   ├──phi3_tool calls_medical.py
```


- ### RESULT
- TP (True Positives): entity is present and is extracted correctly  
- TN (True Negatives): entity is not present and nothing is extracted  
- FP (False Positives): entity is not present but entity is extracted  
- FN (False Negatives): entity is present but is not extracted  

<p align="center">
<img src="./images/extract_entity_confusion_matrix.png" width="450" height="200"/>  
EXTRACT ENTITY CONFUSION MATRIX
</p>

<p align="center">
<img src="./images/example.png"/>  
RESULTS GENERATED DURING THE PROCESS
</p>

## Generate derived query

- ### Dataset
1. harry potter (HARRY POTTER AND THE CHAMBER OF SECRETS)
https://github.com/qhjqhj00/MemoRAG/blob/main/examples/harry_potter.txt
2. Create complex queries that require a combination of fragments and answers
Example: How is 'prejudice' addressed in this book?, What's the book's main theme?  
    - 30 generated queries file path: `eval_file/derived_queries_search/harry_queries.txt`

- ### How to evaluate   
1. Create a Harry Potter memory database with `extract_harry_entity.py`.  
2. Create two answers with `harry_derived_query_Quantitative.py`, one created by searching with the original query and one created by searching with the derived query (I created answers to each query without a for loop). You can see the **Derived query** and **Answer with derived query search** and **Answer with original query search** in the example below.  
3. Create MemoRAG results with `memorag_code.py`. You can find the required dependencies in the `requirements_memorag.txt` file. For an example answer, see the **Answer of MemoRAG** below.   
4. Create a prompt to be inserted into llm with the three answers created in steps 2 and 3 added using `harry_derived_test_Quantitative.py`.  
5. Insert the created prompt into chatgpt-4o-mini (web interface) to evaluate the three answers. Examples of evaluation results are as follows.  


```bash
Example output of Ranking score
(1) 'r': 2
(2) 'r': 1
(3) 'r': 3
```


```bash
├── eval_file
│   ├── derived_queries_search
│   │   ├──extract_harry_entity.py
│   │   ├──harry_derived_query_Quantitative.py
│   │   ├──harry_derived_test_Quantitative.py
│   │   ├──harry_queries.txt
│   │   ├──memorag_code.py
│   │   ├──requirements_memorag.txt
```


<p align="center">
<img src="./images/relation.png"/>
RELATIONSHIP BETWEEN ORIGINAL QUERY, CONTEXT, AND DERIVED QUERIES
</p> 

<p align="center">
<img src="./images/derived_query_example.png"/>
COMPARISON OF QUERY SEARCH AND ANSWER, DERIVED QUERY SEARCH AND ANSWER, AND MEMORAG ANSWER EXAMPLE
</p>

Table above shows the results of three methods for generating answers: using only the query, using additional queries derived from the original query, and using MemoRAG. The answer generated with only the query contains some correct and incorrect parts, such as errors about relationships between Percy, Ginny, Tom Riddle, and Harry. The MemoRAG result explains the overall relationships well, but briefly and concisely. The answer derived from additional queries is generally correct, similar to the MemoRAG result, but more detailed. It clarifies that while Ginny and Harry are not in a perfect relationship, Ginny’s crush on Harry confirms their romantic connection.

<p align="center">
<img src="./images/Results_of_quantitative_analysis.png"/>
RESULTS OF QUANTITATIVE ANALYSIS
</p>

We generated 30 queries that require considering the complex and holistic parts of the Harry Potter data. And we evaluated them using gpt-4o-mini to measure the response accuracy and ranking score. Response correctness evaluates whether the response contains the correct answer to the search question (label: {0: wrong, 0.5: partial, 1: correct}). Ranking score ranks the outputs of three methods for the same question and context. The model's score is calculated using s = 1/r, where r = 1, 2, 3 represent relative rankings. The MemoRAG used MemoRAG-lite provided by the MemoRAG github . The default setting was used, and the generate model was Qwen2.5-1.5B-Instruct. MemoRAG-lite answer (Qwen) is the same as the default in the previous process, but qwen2.5-1.5B-Instruct is used when generating the final answer with the search result and query, and MemoRAG -lite answer (mistral) is the same as the default setting in the previous process, but the difference is that the Mistral-7B-Instruct-v0.3 GGUF model is used as the final answer generation model.