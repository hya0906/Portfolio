import queue
import threading
import warnings
warnings.filterwarnings("ignore")
import copy
import json
import math
import os
import sqlite3
import time
from datetime import datetime
import numpy as np
import pytz
from langchain import PromptTemplate
from llama_cpp import Llama
from mem0 import Memory
from mem0.configs.prompts import derived_query_prompt
from datasets import load_dataset
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pprint
PATH_TO_DBS = "./demo"

# File = open("no_f_curve_result.txt", "a", encoding="utf-8")

def get_user_ids():
    files = os.listdir(PATH_TO_DBS)
    user_ids = [user.split(".")[0] for user in files if user.endswith(".db")]
    return user_ids


name = "realtime_test_yuna_new"
class Conversation_entity:
    def __init__(self, llm):
        self.llm = llm
        self.llm_answer = ""
        self.dialog_history = ""
        self.user_name = ""
        self.user_id = ""
        self.user_meta = ""
        self.save_folder_path = f"demo/{name}"
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

    def get_all_entities(self):
        # Retrieve all entities
        entities = self.m.get_all()["memories"]
        data_ = {"type": 3, "data": entities}
        data = json.dumps(data_)
        print("Data retrieval complete")
        return data

    def load_chromadb(self):
        db_path = f"./{self.save_folder_path}/{self.user_id}"
        try:
            if not os.path.exists(db_path):
                os.makedirs(db_path)
        except OSError:
            print("Error: Failed to create the directory.")

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

        config = {
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
            },
            "history_db_path": os.path.join(db_path, "history.db"),
            "custom_prompt": self.custom_prompt,
            "version": "v1.1",
        }

        self.m = Memory.from_config(config_dict=config)


    def get_answer(self, prompt, entity=""):
        _DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = f"""You are an AI assistant chatbot based on a large language model, designed to provide medical advice to users.
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
        
        Memory context:
        {{entities}}

        Last line:
        Human: {{query}}
        You:"""

        ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
            input_variables=["entities", "history", "query", "basedata"],
            template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        )
        temp = ENTITY_MEMORY_CONVERSATION_TEMPLATE.format(
            query=prompt,
            entities=entity,
            history=self.dialog_history,
            basedata=self.user_meta,
            device="cuda",
        )
        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": temp,
                }
            ],
            stream=True,
            max_tokens=300,
            stop=["Human:", "You: ", "AI:", "User:", "        ", "Current context:"],
        )

        for chunk in output:
            if 'role' in chunk['choices'][0]['delta']:
                print("AI: ", end='')
            elif chunk['choices'][0]['delta']:
                print(chunk['choices'][0]['delta']["content"], end='')
                self.llm_answer += chunk['choices'][0]['delta']["content"]

        with open(f"realtime_result.txt", "a", encoding="utf-8") as file:
            file.write("AI: "+self.llm_answer+"\n\n")
        self.update_dialog_history(prompt)
        self.llm_answer = ""
        print("\n")

    def update_dialog_history(self, prompt):
        self.dialog_history += json.dumps(
            {"client": prompt, "assistant": self.llm_answer}
        )
        # print(self.dialog_history)
        dict_list = []
        start = 0

        while start < len(self.dialog_history):
            end = self.dialog_history.find("}", start) + 1
            if end == 0:
                break
            dict_list.append(json.loads(self.dialog_history[start:end]))
            start = end

        # Delete the first dictionary if the number of dictionaries exceeds 5.
        if len(dict_list) > 5:
            dict_list = dict_list[1:]

        # Convert back to a string.
        self.dialog_history = "".join(json.dumps(d) for d in dict_list)

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

    def get_meta(self):  # Retrieve user base data
        db_file = f"./{self.save_folder_path}/{self.user_id}.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Basedata")
        rows = cursor.fetchall()
        print(rows)
        self.user_name = rows[0][1]
        self.user_meta = str(
            {
                "saved_date": rows[0][0],
                "user_name": rows[0][1],
                "user_id": rows[0][2],
                "gender": rows[0][3],
                "user_age": rows[0][4],
            }
        )
        conn.close()

    def save_base(self, data):  # Save user base data
        db_file = f"./{self.save_folder_path}/{self.user_id}.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS Basedata (
                    date datetime,
                    name tect,
                    id integer,
                    gender text,
                    age integer
                )"""
        )
        cursor.execute("SELECT * FROM Basedata")
        rows = cursor.fetchall()
        if not rows:
            cursor.execute(
                """
            INSERT INTO Basedata (date, name, id, gender, age)
            VALUES (?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().strftime("%Y-%m-%d"),
                    data["name"],
                    data["id"],
                    data["gender"],
                    data["age"],
                ),
            )
            conn.commit()
        conn.close()

    def check_record_count(self):
        collection = aa.m.vector_store.collection

        MAX_RECORDS = 1000  # 최대 레코드 수
        current_count = collection.count()  # 현재 레코드 수 확인

        if current_count < MAX_RECORDS:
            print("데이터 개수: ", current_count)
        else:
            print("데이터 한도를 초과했습니다.")

            min_value = 1
            min_d = None
            for saved_data in self.m.vector_store.list()[0]:
                days_diff = self._get_date_difference(
                    saved_data.payload["created_at"].split("T")[0],
                    datetime.now(pytz.timezone("US/Pacific")).isoformat().split("T")[0],
                )
                # retention_probability = self.forgetting_curve(days_diff, saved_data.payload["memory_strength"])
                retention_probability = self.forgetting_with_repetition(days_diff, None)
                if retention_probability < min_value:
                    min_value = retention_probability
                    min_d = saved_data

                self.m.delete(min_d.id)

    def forget_data(self):  # delete memory
        if self.m.vector_store.list()[0]:
            for saved_data in self.m.vector_store.list()[0]:
                if (
                    saved_data.payload["shift_count"] == -1
                ):  # Long term memory is not deleted.
                    continue
                if "updated_at" in saved_data.payload:
                    before = saved_data.payload["updated_at"].split("T")[0]
                else:
                    before = saved_data.payload["created_at"].split("T")[0]
                days_diff = self._get_date_difference(
                    before,
                    datetime.now(pytz.timezone("US/Pacific")).isoformat().split("T")[0],
                )
                # retention_probability = self.forgetting_curve(days_diff, saved_data.payload["memory_strength"]) # base forgetting curve
                retention_probability = self.forgetting_with_repetition(
                    days_diff,
                    saved_data.payload["memory_strength"],
                    saved_data.payload["shift_count"],
                )
                # Keep the memory with the retention_probability
                if 0.3 > retention_probability:
                    print(self.m._delete_memory_tool(saved_data.id))
            print("============================================\n")

    def _get_date_difference(
        self, date1: str, date2: str
    ) -> int:  # Calculate date difference
        date_format = "%Y-%m-%d"
        d1 = datetime.strptime(date1, date_format)
        d2 = datetime.strptime(date2, date_format)
        return (d2 - d1).days

    def forgetting_curve(self, t, S):  # base forgetting curve
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

    def forgetting_with_repetition(
        self, t, S, shift
    ):  # forgetting curve with repetition
        return np.exp(-t / (math.pow(2, shift + 2) * S))

    def predict(self, prompt, today_date, final=None):
        start = time.time()
        thread1 = threading.Thread(target=self.m.add, args=(prompt,), kwargs={"user_id": self.user_name, "prompt": self.custom_prompt, "today_date": today_date},)  # add memory


        # while True:  # Check if there is an entity to be extracted or not
        #     time.sleep(0.1)
        #     if self.m.flag != None:
        #         print(self.m.flag)
        #         break
        #
        # if (
        #         (self.m.flag == False) and (final == True)
        # ):  # Create derived query if there are entities to be extracted
        #     derived_prompt = derived_query_prompt["sur"].format(question=prompt)
        #     derived_query = self.m.llm.client(
        #         derived_prompt,
        #         temperature=0.8,
        #         max_new_tokens=128,
        #         repetition_penalty=1.1,
        #     )[0]["generated_text"]
        #     final_query = derived_query.replace(derived_prompt, "") + "\n" + prompt
        #     print("final query: ", final_query)
        # elif (self.m.flag == False) and (final == None):
        #     final_query = prompt
        # elif (self.m.flag == True):  # Do not create derived queries if no entities are extracted
        #     final_query = prompt

        if final == True:
            derived_prompt = derived_query_prompt["sur"].format(question=prompt)
            derived_query = self.m.llm.client(
                derived_prompt,
                temperature=0.8,
                max_new_tokens=128,
                repetition_penalty=1.1,
            )[0]["generated_text"]
            final_query = derived_query.replace(derived_prompt, "") + "\n" + prompt

            print("final query: ", final_query)

            with open(f"realtime_result.txt", "a", encoding="utf-8") as file:
                file.write("DERIVED QUERY: ")
                pprint.pprint(final_query, stream=file)
                file.write("\n")

        else:
            final_query = prompt

        thread2 = threading.Thread(
            target=self.m.search,
            args=(final_query,),
            kwargs={"user_id": self.user_name, "limit": 5, "q": self.q},
        )
        thread2.start()
        thread1.start()
        thread1.join()
        thread2.join()
        a = self.q.get()
        end = time.time()
        print(f"{end - start:.5f} sec\n")
        print("\n========================================\n\n")
        if final == True:
            with open(f"realtime_result.txt", "a", encoding="utf-8") as file:
                file.write("MEMORY: ")
                pprint.pprint(a["memories"], stream=file)
                file.write("\n")
            
        print("SEARCHED MEMORIES: ", a)
        copied_get = copy.deepcopy(a["memories"])
        for i in copied_get:
            # if i["updated_at"] != None:
            #     del i["created_at"]
            # else:
            #     del i["updated_at"]
            del i["id"]
            del i["hash"]
            del i["score"]

        if final != True:
            pass
        else:
            return self.get_answer(prompt, copied_get)


gen_stn = Llama(
    model_path="C:\\Users\\711_2\\Desktop\\NLP_emotion\\Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
    # model_path="./Mistral-7B-Instruct-v0.3.IQ1_M.gguf",
    verbose=False,
    n_ctx=12300,
    n_batch=512,
    n_threads=4,
    n_gpu_layers=-1,
)
aa = Conversation_entity(gen_stn) #gen_stn

user_ids = get_user_ids()


aaa = 0
bbb = 0


global my_date
def main():
    idx = "yuna"
    print("user id: ", idx, "user name: ", str(idx))
    aa.user_id = idx
    aa.user_name = str(idx)
    aa.forget_data()
    aa.dialog_history = ""
    while True:
        sentence = input("User: ")

        with open(f"realtime_result.txt", "a", encoding="utf-8") as file:
            file.write("idx: " + str(idx) + "\n")
            file.write("User: "+sentence)

        print(str(datetime.today()).split(" ")[0])
        aa.predict(sentence, str(datetime.today), True)  # date, id

        for i in aa.m.get_all(user_id=aa.user_name)["memories"]:
            print(i)

        with open("realtime_result.txt", "a", encoding="utf-8") as file:
            file.write("ANSWER: " + aa.llm_answer + "\n\n")




if __name__ == "__main__":
    main()
