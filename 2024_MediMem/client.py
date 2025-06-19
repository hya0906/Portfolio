import asyncio
import websockets
import json
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import threading


# UI Builder Class
class MemoryUI():
    def __init__(self):
        self.websocket = None  # Initialize WebSocket variable
        self.is_id_verified = False

        # Create asyncio event loop
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.run_loop, daemon=True).start()

        self.create_ui()

    async def connect_websocket(self): # Connect to the WebSocket server
        self.websocket = await websockets.connect("ws://localhost:8000/LLM")

    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def on_submit_user_id(self): # Submit user ID on button click
        user_id = self.user_id_entry.get()
        self.text_area_output.insert(tk.END, f"Submitted User ID: {user_id}\n")
        asyncio.run_coroutine_threadsafe(self.connect_and_verify(user_id), self.loop)

    def clear_tree_table(self):
        # 전체 항목 삭제
        for item in self.tree_entities.get_children():
            self.tree_entities.delete(item)

    async def populate_table(self): #event logs table update
        await asyncio.sleep(0.1)
        while True:
            logs = await self.websocket.recv()
            logs = json.loads(logs)
            # print("############logs: ", logs)
            #"Created_at", "Updated_at", "New_memory", "Old_memory", "Event"
            if logs["type"] == 2:
                for i in logs["data"]:
                    if i[5] != "DELETE":
                        self.tree.insert("", "end", values=(i[6].split(".")[0], i[7], i[3], i[2], i[5]))
                    else:
                        if i[3] == None:
                            self.tree.insert("", "end", values=(None, None, i[2], None, i[5]))
                        else:
                            self.tree.insert("", "end", values=(None, None, i[3], None, i[5]))
                    print("1")
                self.tree.see(self.tree.get_children()[-1])  # 마지막 항목으로 스크롤 내리기
                break
            else:
                print("again111")
                # If not done, schedule the check again
                self.loop.call_soon_threadsafe(self.populate_entities)



    async def populate_entities(self): #extracted entities table update
        while True:
            await asyncio.sleep(1.5)
            logs = await self.websocket.recv()
            logs = json.loads(logs)
            print("entities: ", logs)
            #"Created_at", "Last_mentioned", "Memory", "Memory_strength"
            if logs["type"] == 3:
                self.clear_tree_table()
                print("delete complete!!!!!")
                for i in logs["data"]:
                    self.tree_entities.insert("", "end", values=(
                    i["created_at"].split(".")[0], i['updated_at'], i["memory"], i["metadata"]["memory_strength"]))
                self.tree_entities.see(self.tree_entities.get_children()[-1])  # 마지막 항목으로 스크롤 내리기
                break
            else:
                print("again")
                # If not done, schedule the check again
                self.loop.call_soon_threadsafe(self.populate_table)

        print("break end")

    async def connect_and_verify(self, user_id): #이미 존재하는 user id를 제출
        await self.send_message(user_id)
        response = await self.websocket.recv()
        response = json.loads(response)
        if response["type"] == 0 and response["data"] == "ok exist":
            self.is_id_verified = True
            self.text_area_output.insert(tk.END, "ID verified. You can send messages now.\n")
            self.text_area_output.see(tk.END)
            asyncio.run_coroutine_threadsafe(self.populate_table(), self.loop)
            asyncio.run_coroutine_threadsafe(self.populate_entities(), self.loop)

        elif response["type"] == 0 and response["data"] != "ok exist":
            self.text_area_output.insert(tk.END, "ID verification failed. Try again\n")
            self.text_area_output.see(tk.END)
        return

    async def send_message(self, message): #server에 데이터 제출
        await self.connect_websocket()

        try:
            await self.websocket.send(message)

        except Exception as e:
            print(f"Error: {e}")

    def on_submit_data(self): # After submitting user query, transfer it to the output window
        user_input = self.text_area_input.get("1.0", tk.END).strip()
        if user_input:  # Add only if there is input content
            self.text_area_input.delete("1.0", tk.END)  # Reset input field
            # Display the result in the second text box
            self.text_area_output.insert(tk.END, "User: " + user_input + "\n")  # Add to the output text box
            self.text_area_output.see(tk.END)  # Scroll to the bottom

            asyncio.run_coroutine_threadsafe(self.send_message(user_input), self.loop) #server에 user query 제출
            dd = asyncio.run_coroutine_threadsafe(self.tt(), self.loop) #answer 출력

            self.loop.call_soon_threadsafe(self.check_dd_completion, dd) #answer 출력 후 memory 출력 표 업데이트

    def check_dd_completion(self, future):
        if future.done():
            result = future.result()  # Get the result if needed
            # Now call populate_table and populate_entities
            dd1 = asyncio.run_coroutine_threadsafe(self.populate_table(), self.loop)
            self.loop.call_soon_threadsafe(self.check_dd_completion1, dd1)
        else:
            # If not done, schedule the check again
            self.loop.call_soon_threadsafe(self.check_dd_completion, future)

    def check_dd_completion1(self, future1):
        if future1.done():
            asyncio.run_coroutine_threadsafe(self.populate_entities(), self.loop)
        else:
            # If not done, schedule the check again
            self.loop.call_soon_threadsafe(self.check_dd_completion1, future1)

    async def tt(self): # Function to display the answer from the LLM
        c = 0
        a = self.websocket

        while a == self.websocket:
            await asyncio.sleep(0.1)
            if a != self.websocket:
                break

        async for message in self.websocket:
            message = json.loads(message)
            if message["type"] == 1 and message["data"] == "END": # End of streaming
                self.text_area_output.insert(tk.END, "\n\n") # Notify the end of output in the widget
                self.text_area_output.see(tk.END)  # Scroll to the bottom
                break

            elif message["type"] == 1 and message["data"] != "END":  # During streaming
                self.text_area_output.insert(tk.END, message["data"])  # Add one word at a time to the output text box
                c += 1
                if c % 10 == 0:
                    self.text_area_output.see(tk.END)  # Scroll to the bottom

        return "done"

    def create_ui(self): # Create UI
        # Create main window
        self.root = tk.Tk()
        self.root.title("Non-parametric Memory")

        # Set window size
        self.root.geometry("1200x800")  # 너비 800px, 높이 600px

        # Create left frame
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create the first text box and label
        label1 = tk.Label(left_frame, text="Extracted entities")
        label1.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.tree_entities = ttk.Treeview(left_frame, columns=("Created_at", "Last_mentioned", "Memory", "Memory_strength"), show='headings')
        self.tree_entities.heading("Created_at", text="Created_at", anchor=tk.CENTER)
        self.tree_entities.column("Created_at", width=(len("Created_at") - 3) * 5)
        self.tree_entities.heading("Last_mentioned", text="Last_mentioned", anchor=tk.CENTER)
        self.tree_entities.column("Last_mentioned", width=(len("Last_mentioned") - 3) * 5)
        self.tree_entities.heading("Memory", text="Memory", anchor=tk.CENTER)
        self.tree_entities.column("Memory", width=(len("Memory") - 2) * 10)
        self.tree_entities.heading("Memory_strength", text="Memory_strength", anchor=tk.CENTER)
        self.tree_entities.column("Memory_strength", width=(len("Memory_strength") - 4) * 3)

        # Create scrollbar
        self.scrollbar_entities = tk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree_entities.yview)
        self.tree_entities.configure(yscrollcommand=self.scrollbar_entities.set)

        # Place Treeview and scrollbar
        self.tree_entities.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.scrollbar_entities.grid(row=1, column=1, sticky='ns')

        # Create the second text box and label
        label2 = tk.Label(left_frame, text="Event Logs")
        label2.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        # Create Treeview
        self.tree = ttk.Treeview(left_frame, columns=("Created_at", "Updated_at", "New_memory", "Old_memory", "Event"), show='headings')
        self.tree.heading("Created_at", text="Created_at", anchor=tk.CENTER)
        self.tree.column("Created_at", width=(len("Created_at") - 3) * 10)
        self.tree.heading("Updated_at", text="Updated_at", anchor=tk.CENTER)
        self.tree.column("Updated_at", width= (len("Updated_at") - 3) * 10)
        self.tree.heading("New_memory", text="New_memory", anchor=tk.CENTER)
        self.tree.column("New_memory", width=(len("New_memory") - 2) * 10)
        self.tree.heading("Old_memory", text="Old_memory", anchor=tk.CENTER)
        self.tree.column("Old_memory", width=(len("Old_memory") - 2) * 10)
        self.tree.heading("Event", text="Event", anchor=tk.CENTER)
        self.tree.column("Event", width= (len("Event") - 2) * 10)

        # Create scrollbar
        self.scrollbar = tk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # Place Treeview and scrollbar
        self.tree.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.scrollbar.grid(row=3, column=1, sticky='ns')

        # Create right frame
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)

        # User ID label and input field
        user_id_label = tk.Label(right_frame, text="User ID:")
        user_id_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.user_id_entry = tk.Entry(right_frame, width=40)
        self.user_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # User ID submit button
        submit_id_button = tk.Button(right_frame, text="Submit ID", command=self.on_submit_user_id)
        submit_id_button.grid(row=0, column=2, padx=5, pady=5)

        # Add input data label
        label3 = tk.Label(right_frame, text="Input data:")
        label3.grid(row=1, column=0, padx=5, pady=5, sticky="w", columnspan=3)

        # Add input text box
        self.text_area_input = scrolledtext.ScrolledText(right_frame, width=40, height=3)  # Set the height to 3
        self.text_area_input.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Data submit button
        submit_data_button = tk.Button(right_frame, text="Submit Data", command=self.on_submit_data)
        submit_data_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew", columnspan=3)

        # Output data label
        label4 = tk.Label(right_frame, text="Output:")
        label4.grid(row=4, column=0, padx=5, pady=5, sticky="w", columnspan=3)

        # Output text box
        self.text_area_output = scrolledtext.ScrolledText(right_frame, width=40, height=10)  # Set the height to 10
        self.text_area_output.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Adjust the ratio of the frame
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_rowconfigure(3, weight=1)
        right_frame.grid_rowconfigure(2, weight=0)  # Adjust the ratio for the input text box
        right_frame.grid_rowconfigure(5, weight=1)  # Adjust the ratio for the output text box

        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_columnconfigure(1, weight=0)  # Set column for the scrollbar
        right_frame.grid_columnconfigure(1, weight=1)  # Adjust the ratio for the input text box


async def main():
    ui = MemoryUI()
    ui.root.mainloop() # Start UI loop

asyncio.run(main())