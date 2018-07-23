import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import *


class Page(ttk.Frame):
    def __init__(self, *args, **kwargs):
        ttk.Frame.__init__(self, *args, **kwargs)
    def show(self):
        self.lift()
    def write(self):
        print('lift')
