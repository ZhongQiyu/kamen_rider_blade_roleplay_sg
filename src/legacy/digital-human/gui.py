# gui.py

import tkinter as tk
from tkinter import scrolledtext
from .components import GameController, YourKamenRiderBladeAgent, GameDialogueManager, generate_emotional_response

class KamenRiderGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("假面骑士剑 - 对话智能体")
        self.game_controller = GameController()
        self.dialogue_agent = YourKamenRiderBladeAgent()
        self.game_dialogue_manager = GameDialogueManager(self.game_controller, self.dialogue_agent)

        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD)
        self.chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry_frame = tk.Frame(root)
        self.entry_frame.pack(padx=10, pady=10, fill=tk.X)

        self.entry_field = tk.Entry(self.entry_frame)
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.entry_frame, text="发送", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

    def send_message(self, event=None):
        user_input = self.entry_field.get()
        self.entry_field.delete(0, tk.END)
        self.display_message("用户", user_input)

        response = generate_emotional_response(user_input, self.game_dialogue_manager)
        self.display_message("假面骑士剑", response)

    def display_message(self, sender, message):
        self.chat_window.configure(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"{sender}: {message}\n")
        self.chat_window.configure(state=tk.DISABLED)
        self.chat_window.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    gui = KamenRiderGameGUI(root)
    root.mainloop()
