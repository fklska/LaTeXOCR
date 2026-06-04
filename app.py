import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageDraw
import numpy as np
from predict import SymbolRecognizer
import os

class SymbolRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Распознаватель рукописных символов")
        self.window.geometry("700x600")
        self.window.configure(bg='#f0f0f0')
        
        # Загрузка модели
        self.model = SymbolRecognizer()
        
        # Настройка рисования
        self.canvas_size = 300
        self.setup_canvas()
        
        # Интерфейс
        self.setup_ui()
        
        self.last_x = None
        self.last_y = None
        self.brush_size = 12
    
    def setup_canvas(self):
        """Настройка поля для рисования"""
        self.canvas = Canvas(self.window, width=self.canvas_size, 
                             height=self.canvas_size, bg='white', cursor='cross')
        self.canvas.pack(pady=10)
        
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
    
    def setup_ui(self):
        """Настройка кнопок и меток"""
        button_frame = Frame(self.window, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        Button(button_frame, text="Распознать", command=self.recognize,
               font=("Arial", 12), bg='#4CAF50', fg='white', padx=20).pack(side='left', padx=5)
        
        Button(button_frame, text="Очистить", command=self.clear_canvas,
               font=("Arial", 12), bg='#f44336', fg='white', padx=20).pack(side='left', padx=5)
        
        # Результат
        self.result_label = Label(self.window, text="Нарисуйте символ",
                                   font=("Arial", 24, "bold"), bg='#f0f0f0')
        self.result_label.pack(pady=20)
        
        self.confidence_label = Label(self.window, text="",
                                       font=("Arial", 12), bg='#f0f0f0')
        self.confidence_label.pack()
    
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.brush_size, fill='black',
                                    capstyle='round', smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                          fill=0, width=self.brush_size)
        self.last_x = x
        self.last_y = y
    
    def reset(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Нарисуйте символ")
        self.confidence_label.config(text="")
    
    def recognize(self):
        """Распознавание нарисованного символа"""
        symbol, confidence = self.model.predict(self.image)
        self.result_label.config(text=symbol)
        self.confidence_label.config(text=f"Уверенность: {confidence:.1%}")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = SymbolRecognitionApp()
    app.run()