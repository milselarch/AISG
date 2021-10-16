import threading
import tkinter as tk
import time
import queue

from multiprocessing import Queue, Process


class GUI(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.queue = queue.Queue()

        self.test_button = tk.Button(self, command=self.tb_click)
        self.test_button.configure(
            text="Start", background="Grey",
            padx=50
        )
        self.test_button.pack(side=tk.TOP)
        self.pack()

    def tb_click(self):
        ThreadedTask(self.queue).start()
        self.master.after(100, self.process_queue)

    def process_queue(self):
        try:
            msg = self.queue.get_nowait()
            # Show result of the task if needed
            print(msg)
        except queue.Empty:
            self.master.after(100, self.process_queue)


class ThreadedTask(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        time.sleep(5)  # Simulate long running process
        self.queue.put("Task finished")


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Test Button")
    main_ui = GUI(root)
    root.mainloop()