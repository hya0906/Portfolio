import json
from mem0.configs.prompts import derived_query_prompt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import warnings
import copy
import threading
import queue
warnings.filterwarnings('ignore')
import asyncio
from llama_cpp import Llama
import time
import ast
import json
from langchain import PromptTemplate
import os
import copy
from datetime import datetime
import sqlite3
import math
import random
import pytz
import numpy as np
from mem0 import Memory
#pip install mem0ai==0.1.7
#pip install chromadb==0.5.23
# pytorch <= 2.4.1 (for onnxruntime-gpu)
#python -m spacy download en
# pip install transformers==4.46.3 (under 4.47 version)
app = FastAPI()


class Conversation_entity:
    def __init__(self, llm):
        self.llm = llm
        self.user_name = ""
        self.user_id = ""
        self.user_meta = ""
        self.save_folder_path = "demo"
        self.flag = 0
        self.exist_entity = 1
        self.m = None
        self.last_log = ""
        self.q = queue.Queue()
        self.load_chromadb()
        try:
            if not os.path.exists(f"./{self.save_folder_path}"):
                os.makedirs(f"./{self.save_folder_path}")
        except OSError:
            print("Error: Failed to create the directory.")

    async def get_all_history_log(self, websocket, meta):
        conn = aa.m.db.connection.cursor()
        if meta == "check_id":
            conn.execute("SELECT COUNT(*) FROM history")
            self.history_log_count = conn.fetchone()[0]

            conn.execute("SELECT * FROM history")
            table = conn.fetchall()
            self.last_log = table[-1]
            data_ = {"type": 2, "data": table[-30:]}
            for i in table[-30:]:
                print("===", i)

        elif meta == "add_data": #확실하게 맨 마지막 내용이 나오는지 다시 수정필요. id필요할듯
            conn.execute("SELECT COUNT(*) FROM history")
            count = conn.fetchone()[0]
            N = count - self.history_log_count
            self.history_log_count = count

            # 맨 마지막 데이터 조회
            conn.execute(f"SELECT * FROM history LIMIT {N} OFFSET (SELECT COUNT(*) FROM history) - {N};")
            table = conn.fetchall()
            if self.last_log != table:
                data_ = {"type": 2, "data": table}
                self.last_log = table
            else:
                data_ = {"type": 2, "data": ["None"]}

        data = json.dumps(data_)
        print("===================\n", data)
        await websocket.send_text(data)


    async def get_all_entities(self, websocket):
        await asyncio.sleep(1)
        # 모든 데이터 조회
        entities = self.m.get_all(user_id=self.user_name)["memories"]
        # for i in entities:
        #     print("entities: ",i)
        data_ = {"type": 3, "data": entities}
        data = json.dumps(data_)
        # print(data)
        print("end!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        await websocket.send_text(data)
    def load_chromadb(self):
        db_path = "./demo/004"
        try:
            if not os.path.exists(db_path):
                os.makedirs(db_path)
        except OSError:
            print("Error: Failed to create the directory.")

        #특정 카테고리만
        # self.custom_prompt = """
        # You are an expert at extracting entities containing symptoms, diseases_diagnosed, medicines.
        # Only extract and do not add any explanations or notes.
        # Entity Format: List of dict, where each dict having format as {{"{{entity key}}": {{value of entity}}"}}
        # VERY IMPORTANT: MUST omit categories that have not been mentioned.
        #
        # You need to extract only below mentioned entities and save entity to the memory:
        # symptoms: str, Symptoms are confined to only physical signs related to healthcare.
        # diseases_diagnosed: str, represents a disease diagnosed the user has.
        # medicines: str, represents prescription drugs that the user is taking
        #
        # Here are some few shot examples:
        # Input: I’m feeling quite fatigued again this week. And I am taking Losartan.
        # Output: {{'symptoms': 'feeling quite fatigued', 'medicines': 'taking Losartan'}} END
        #
        #
        # Input: I have diabetes and don't have a headache.
        # Output: {{'diseases_diagnosed': 'have diabetes, don't have a headache'}} END
        # END OF EXAMPLES
        #
        # OUTPUT FORMAT:
        # - symptoms: ''
        # - diseases_diagnosed: ''
        # - medicines: ''
        #
        # Deduce the symptoms, diseases_diagnosed, and medicines from the provided text.
        # Just return the symptoms, diseases_diagnosed, and medicines in bullet points:
        # Natural language text: {user_input}
        # """

        self.custom_prompt = """
        You are an expert at extracting entities containing symptoms, diseases, medicines.
        You should only extract information related to the user.
        IMPORTANT: Never extract entities that exist in the question. Only extract and do not add any explanations or notes.
        IMPORTANT: Even without commas or periods, the question part should be identified based on the sentence structure.

        Only the entity categories mentioned below should be extracted:
        symptoms: str, Symptoms are confined to only physical signs related to healthcare.
        diseases: str, represent the illnesses that the user has been diagnosed with or is related to.
        medications: str, are substances used to treat or prevent diseases and conditions.
        Intervention: str, is a term that encompasses all medical actions, including tests, treatments, and procedures, as well as medical interventions.

        Here are some few shot examples:
        Input query: I’m feeling quite fatigued again this week. And I am taking Losartan.
        Output: - symptoms: 'feeling quite fatigued', - medications: 'taking Losartan'  END

        Input query:  He mentioned something about an autoimmune disease, but he’s not sure yet.
        Output: - diseases: 'Not sure, but possibly an autoimmune disease.'  END

        Input query: I had a sore throat for 3 days but with no fever should i be tested for covid-19?
        Output: - symptoms:'sore throat for 3 days'  END
        END OF EXAMPLES

        OUTPUT FORMAT:
        - symptoms: ''
        - diseases: ''
        - medications: ''
        - Intervention: ''
        
        Input query: {user_input}
        """

        #추출하는 부분 추가로 수정 좀더 정확하게 추출하도록!!!!24.11.10 common category
        # self.custom_prompt ="""
        #         You are an expert at extracting entities from input.
        #         VERY IMPORTANT: Never, ever generate content that has not been mentioned.
        #         VERY IMPORTANT: Entity Format: List of dict, where each dict having format as {{"{{entity key}}": {{value of entity}}"}}
        #
        #         IMPORTANT: If any of the sentences are related to the topic, then extract it into one sentence and the other topic produces a different entity.
        #         Example:
        #         Input:  I'm so sorry to hear that! How on earth did that happen?
        #         Output: {{}} END
        #
        #         Input: I work out a few times a week and a trainer came over and asked if I was interested in wrestling and gave it a go! Have you ever wrestled?
        #         Output: {{{{"Motive": a trainer came over and asked if I was interested in wrestling}}, {{"Activities": Works out a few times a week}}, {{"Interests": Interested in wrestling}}}} END
        #
        #         Input: I was three when I learned to play guitar!
        #         Output: {{"Skills": Learned to play guitar when I was 3}} END
        #
        #         Input: I have a headache. what should I do?
        #         Output: {{"Symptom": Have a headache}} END
        #         End of Example
        #
        #         Deduce the entities from the provided text.
        #         Just return the entities in bullet points:
        #         Natural language text: {user_input}
        #         User/Agent details: {metadata}
        #
        #         Constraint for deducing the facts, preferences, and memories about entities:
        #         - The facts, preferences, and memories about entities should be concise and informative.
        #         - Don't start by "The person likes Pizza". Instead, start with "Likes Pizza".
        #         - Don't remember the user/agent details provided. Only remember the facts, preferences, and memories about entities.
        #
        #         Deduced facts, preferences, and memories:
        #         """

        config = {
            # "graph_store": {
            #     "provider": "neo4j",
            #     "config": {
            #         "url": "bolt://localhost:7687",#"neo4j+s//641e5c53.databases.neo4j.io:7687",#"bolt://localhost:7687",neo4j://localhost:7687
            #         "username": "neo4j",
            #         "password": "isplhsko"
            #     }
            # },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "test",
                    "path": db_path,
                },
            },
            "llm": {
                "provider": "phi3",
            },
            "embedder": {
                "provider": "bge_base",
                # "model": "BAAI/bge-large-en",
            },
            "history_db_path": db_path + '\\history.db',
            "custom_prompt": self.custom_prompt, #instruction,
            "version": "v1.1"
        }

        self.m = Memory.from_config(config_dict=config)
        print(self.m)

    #datetime.now().strftime('%Y-%m-%d')
    async def get_answer(self, prompt, websocket: WebSocket, entity = ""):
        _DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = f"""You are an assistant to a human, powered by a large language model trained by OpenAI.
        You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
        As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
        You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions.
        You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.
        Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics.
        Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.

        VERY IMPORTANT: Focus on relatively recent history information when providing your answer. If there is no stored personal information related to the question, reply that you do not know.
        today date: {datetime.now(pytz.timezone("US/Pacific")).isoformat()}

        IMPORTANT: User basedata:
        {{basedata}}

        Context:
        {{history}}
        {{entities}}

        Last line:
        Human: {{query}}
        You:"""


        ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
            input_variables=["entities", "history", "query", "basedata"],
            template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        )
        temp = ENTITY_MEMORY_CONVERSATION_TEMPLATE.format(query=prompt, entities=entity,
                                                          history="", basedata=self.user_meta,
                                                          device="cuda")
        print("@@@@@@",entity)
        # print("LLM instruction", temp)
        # answer = self.llm(temp, max_tokens=300, stop=["Human:", "You: ", "AI:", "User:", "        ", "Current context:"])
        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": temp,
                }
            ],
            stream=True,
            max_tokens=300,
            stop=["Human:", "You: ", "AI:", "User:", "        ", "Current context:"]
        )

        for chunk in output:
            delta = chunk['choices'][0]['delta']
            await asyncio.sleep(0.005)

            if 'role' in delta:
                data_ = {"type": 1, "data": "Assistant: "}
                data = json.dumps(data_)
                await websocket.send_text(data)
            elif 'content' in delta:
                data_ = {"type": 1, "data": delta["content"]}
                data = json.dumps(data_)
                await websocket.send_text(data)

        data_ = {"type": 1, "data": "END"}
        data = json.dumps(data_)
        await websocket.send_text(data)

    # def check_user(self):
    #     db_file = f"./{self.save_folder_path}/{self.user_id}.db"
    #     if not os.path.isfile(db_file):  # if db does not exist
    #         self.create_tables()

    def ckeck_id(self):
        db_file = f"./{self.save_folder_path}/{self.user_id}.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM Basedata")
        rows = cursor.fetchall()
        if not rows:
            return True
        else:
            return False

    def get_meta(self):
        db_file = f"./{self.save_folder_path}/{self.user_id}.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Basedata")
        rows = cursor.fetchall()
        print(rows)
        self.user_name = rows[0][1]
        self.user_meta = str(
            {"saved_date": rows[0][0], "user_name": rows[0][1], "user_id": rows[0][2], "gender": rows[0][3],
             "user_age": rows[0][4]})
        conn.close()

    def save_base(self, data):
        db_file = f"./{self.save_folder_path}/{self.user_id}.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM Basedata")
        rows = cursor.fetchall()
        if not rows:
            cursor.execute('''
            INSERT INTO Basedata (date, name, id, gender, age)
            VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().strftime('%Y-%m-%d'), data["name"], data["id"], data["gender"], data["age"]))
            conn.commit()
        conn.close()

    def check_record_count(self):
        collection = aa.m.vector_store.collection

        MAX_RECORDS = 1000  # 최대 레코드 수
        current_count = collection.count() # 현재 레코드 수 확인

        if current_count < MAX_RECORDS:
            print("데이터 개수: ", current_count)
        else:
            print("데이터 한도를 초과했습니다.")

            min_value = 1
            min_d = None
            for saved_data in self.m.vector_store.list()[0]:
                days_diff = self._get_date_difference(saved_data.payload["created_at"].split("T")[0],
                                                      datetime.now(pytz.timezone("US/Pacific")).isoformat().split("T")[0])
                # retention_probability = self.forgetting_curve(days_diff, saved_data.payload["memory_strength"])
                retention_probability = self.forgetting_with_repetition(days_diff, None)
                if retention_probability < min_value:
                    min_value = retention_probability
                    min_d = saved_data

                self.m.delete(min_d.id)

    def forget_data(self):
        if self.m.vector_store.list()[0]:
            for saved_data in self.m.vector_store.list()[0]:
                if saved_data.payload['shift_count'] == -1:
                    continue
                if "updated_at" in saved_data.payload:
                    before = saved_data.payload["updated_at"].split("T")[0]  # .split(".")[0]
                else:
                    before = saved_data.payload["created_at"].split("T")[0]  # .split(".")[0]
                days_diff = self._get_date_difference(before,
                                                      "2024-12-30")#datetime.now(pytz.timezone("US/Pacific")).isoformat().split("T")[0])
                # retention_probability = self.forgetting_curve(days_diff, saved_data.payload["memory_strength"])
                retention_probability = self.forgetting_with_repetition(days_diff,
                                                                        saved_data.payload["memory_strength"],
                                                                        saved_data.payload['shift_count'])
                print(days_diff, saved_data.payload["memory_strength"], retention_probability, "\n", saved_data)
                # Keep the memory with the retention_probability
                if 0.3 > retention_probability:
                    print(self.m._delete_memory_tool(saved_data.id))
            print("============================================\n")

    def _get_date_difference(self, date1: str, date2: str) -> int:
        date_format = "%Y-%m-%d"
        d1 = datetime.strptime(date1, date_format)
        d2 = datetime.strptime(date2, date_format)
        return (d2 - d1).days

    def forgetting_curve(self, t, S): #forgetting curve 기울기 조정해야할듯
        """
        Calculate the retention of information at time t based on the forgetting curve.

        :param t: Time elapsed since the information was learned (in days).
        :type t: float
        :param S: Strength of the memory.
        :type S: float
        :return: Retention of information at time t.
        :rtype: float
        Memory strength is a concept used in memory models to represent the durability or stability of a memory trace in the brain.
        In the context of the forgetting curve, memory strength (denoted as 'S') is a parameter that
        influences the rate at which information is forgotten.
        The higher the memory strength, the slower the rate of forgetting,
        and the longer the information is retained.
        """
        return math.exp(-t / (5 * S))

        # # Example usage
        # t = 1  # Time elapsed since the information was learned (in days)
        # S = 7  # Strength of the memory

        # retention = forgetting_curve(t, S)
        # print("Retention after", t, "day(s):", retention)

    def forgetting_with_repetition(self, t, S, shift):
        return np.exp(-t / (math.pow(2, shift + 1) *S))


    async def predict(self, prompt, websocket: WebSocket):
        start = time.time()
        thread1 = threading.Thread(target=self.m.add, args=(prompt,), kwargs={"user_id": self.user_name, "prompt": self.custom_prompt}) #"filters":{"categories":"diseases_diagnosed"}
        thread1.start()

        while True:
            print(self.m.flag)
            time.sleep(0.5)
            if self.m.flag != None:
                break

        if self.m.flag == True:
            thread2 = threading.Thread(target=self.m.search, args=(prompt,),
                                       kwargs={"user_id": self.user_name, "limit": 5, "q": self.q})  # threshold 0.2
            thread2.start()
            thread2.join()
            a = self.q.get()

            # print("1234", self.m.history(memory_id=a["memories"][0]["id"]))
            print("1234", a)
            copied_get = copy.deepcopy(a["memories"])
            for i in copied_get:
                if i["updated_at"] != None:
                    del i["created_at"]
                else:
                    del i["updated_at"]
                del i["id"]
                del i["hash"]
                del i['score']
            await self.get_answer(prompt, websocket, copied_get)
        else:
            derived_prompt = derived_query_prompt["sur"].format(question=prompt)
            derived_query = \
            self.m.llm.client(derived_prompt, temperature=0.8, max_new_tokens=128, repetition_penalty=1.1)[0][
                "generated_text"]
            final_query = derived_query.replace(derived_prompt, "") + "\n" + prompt
            # print("final query!!: ", final_query)

            thread2 = threading.Thread(target=self.m.search, args=(final_query,),
                                       kwargs={"user_id": self.user_name, "limit": 5, "q": self.q})  # threshold 0.2
            thread2.start()
            thread2.join()
            a = self.q.get()

            # print("1234", self.m.history(memory_id=a["memories"][0]["id"]))
            print("1234", a)
            copied_get = copy.deepcopy(a["memories"])
            for i in copied_get:
                if i["updated_at"] != None:
                    del i["created_at"]
                else:
                    del i["updated_at"]
                del i["id"]
                del i["hash"]
                del i['score']

            await self.get_answer(prompt, websocket, copied_get)

        end = time.time()
        print(f"{end - start:.5f} sec\n")
        for i in self.m.get_all(user_id=self.user_name)["memories"]:
            print(i)
            # print(self.m.history(i["id"]))
        print("\n========================================\n\n")



gen_stn = Llama(
        model_path="C:\\Users\\711_2\\Downloads\\mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        verbose=False, n_ctx=12300, n_batch=512, n_threads=4, n_gpu_layers=-1)
aa = Conversation_entity(gen_stn)

# 유저 ID 목록 (예시)
user_ids = {"user1", "user2", "user3", "004"}


@app.websocket("/LLM")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if aa.flag == 0:
        try:
            # ID 확인 단계
            user_id = await websocket.receive_text()
            if user_id in user_ids:
                aa.flag = 1
                aa.user_id = user_id
                # aa.save_base(user_id) #필요없음 나중에 아이디 없으면 추가하는 부분 추가
                aa.get_meta()
                data_ = {"type": 0, "data": "ok exist"}
                data = json.dumps(data_)
                await websocket.send_text(data)
                aa.forget_data()
                await aa.get_all_history_log(websocket, "check_id")
                await aa.get_all_entities(websocket)
            else:
                data_ = {"type": 0, "data": "ID does not exist"}
                data = json.dumps(data_)
                await websocket.send_text(data)

        except Exception as e:
            print(f"1WebSocket error: {e}")

    elif aa.flag == 1:
        try:
            data = await websocket.receive_text()
            # aa.check_record_count()
            if data == "Exit":
                await websocket.close()
                print("Connection closed.")

            await aa.predict(data, websocket)
            await aa.get_all_history_log(websocket, "add_data")
            await aa.get_all_entities(websocket)

        except WebSocketDisconnect:
            print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
