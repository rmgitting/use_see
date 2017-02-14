import threading
import tkinter as tk
from tkinter import ttk
import os


class GUIThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.frame = None
        self.count = 1

    def run(self):
        root = tk.Tk()
        self.frame = Frame(root)
        root.mainloop()

    def add_progress_bar(self):
        progress = ttk.Progressbar(self.frame, orient="horizontal",
                                   length=500, mode="determinate")
        progress.grid(row=self.count, columnspan=5, sticky='news', pady=(5, 20))
        self.count += 1

        ttk.Separator(self.frame, orient='horizontal').grid(row=self.count, columnspan=5, sticky='ew')
        self.count += 1

        label = ttk.Label(self.frame, text="")
        label.grid(row=self.count, columnspan=5, pady=(10, 0))
        self.count += 1

        return label, progress


class Frame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()

    def init_gui(self):
        """Builds GUI."""
        self.root.title('Threads Progress')
        self.grid(column=0, row=0, sticky='nsew')
        self.root.protocol('WM_DELETE_WINDOW', Frame.close)

    @staticmethod
    def close():
        print("Aborted by user")
        os._exit(0)
