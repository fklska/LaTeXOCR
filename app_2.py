import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, LabelFrame, messagebox, Listbox
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "model", "class_names.pkl")

IMAGE_SIZE = (64, 64)
CANVAS_SIZE = 300

# Словарь для преобразования в читаемые символы
DISPLAY_MAP = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
    'pi': 'π', 'sigma': 'σ', 'theta': 'θ', 'lambda': 'λ',
    'times': '×', 'div': '÷', 'pm': '±', 'infty': '∞',
    'sqrt': '√', 'prime': "'", 'lparen': '(', 'rparen': ')',
    'eq': '=', 'plus': '+', 'minus': '-', 'slash': '/',
    'cdot': '·', 'leq': '≤', 'geq': '≥', 'neq': '≠',
    'rightarrow': '→', 'leftarrow': '←', 'sum': '∑', 'int': '∫'
}

# Словарь для преобразования в LaTeX (только для кнопки)
LATEX_MAP = {
    'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma',
    'delta': '\\delta', 'pi': '\\pi', 'sigma': '\\sigma',
    'theta': '\\theta', 'lambda': '\\lambda', 'times': '\\times',
    'div': '\\div', 'pm': '\\pm', 'infty': '\\infty',
    'sqrt': '\\sqrt', 'prime': "'", 'lparen': '(', 'rparen': ')',
    'eq': '=', 'plus': '+', 'minus': '-', 'slash': '/',
    'cdot': '\\cdot', 'leq': '\\leq', 'geq': '\\geq', 'neq': '\\neq',
    'rightarrow': '\\rightarrow', 'leftarrow': '\\leftarrow',
    'sum': '\\sum', 'int': '\\int'
}

def to_display(symbol):
    """Преобразует имя символа в читаемый вид"""
    # Если это одиночная буква или цифра - возвращаем как есть
    if len(symbol) == 1 and symbol.isalnum():
        return symbol
    # Иначе ищем в словаре
    return DISPLAY_MAP.get(symbol, symbol)

def to_latex(symbol):
    """Преобразует символ в LaTeX"""
    if len(symbol) == 1 and symbol.isalnum():
        return symbol
    return LATEX_MAP.get(symbol, symbol)

class SymbolRecognizerApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Распознаватель рукописных символов")
        self.window.geometry("900x700")
        self.window.configure(bg='#f0f0f0')
        
        # Загрузка модели и классов
        self.load_model_and_classes()
        
        # Создание интерфейса
        self.setup_ui()
        
        # Инициализация для рисования
        self.setup_canvas()
        
        self.expression_history = []  # хранит символы (например 'x', '+', '1')
        self.last_recognized = None
    
    def load_model_and_classes(self):
        """Загружает обученную модель и список классов"""
        try:
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("Ошибка", f"Файл модели не найден:\n{MODEL_PATH}")
                self.window.quit()
                return
            
            if not os.path.exists(CLASSES_PATH):
                messagebox.showerror("Ошибка", f"Файл классов не найден:\n{CLASSES_PATH}")
                self.window.quit()
                return
            
            self.model = load_model(MODEL_PATH)
            
            with open(CLASSES_PATH, 'rb') as f:
                self.class_names = pickle.load(f)
            
            print(f"✅ Модель загружена: {len(self.class_names)} классов")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{e}")
            self.window.quit()
    
    def get_expression_display(self):
        """Возвращает читаемое представление текущего выражения"""
        return ''.join(to_display(s) for s in self.expression_history)
    
    def get_expression_latex(self):
        """Возвращает LaTeX представление текущего выражения"""
        return ''.join(to_latex(s) for s in self.expression_history)
    
    def update_expression_display(self):
        """Обновляет отображение выражения"""
        display_str = self.get_expression_display()
        self.expression_label.config(text=display_str if display_str else "")
    
    def setup_ui(self):
        top_frame = Frame(self.window, bg='#f0f0f0')
        top_frame.pack(pady=10)
        
        self.status_label = Label(top_frame, text="Нарисуйте символ и нажмите 'Распознать'", 
                                   font=("Arial", 12), bg='#f0f0f0')
        self.status_label.pack()
        
        main_frame = Frame(self.window, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Левая панель - рисование
        left_frame = Frame(main_frame, bg='white', relief='ridge', bd=2)
        left_frame.pack(side='left', padx=10, pady=10)
        
        self.canvas = Canvas(left_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white', cursor='cross')
        self.canvas.pack()
        
        # Правая панель - результаты
        right_frame = Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', padx=10, pady=10, fill='both', expand=True)
        
        # Результат распознавания
        result_frame = LabelFrame(right_frame, text="Результат распознавания", 
                                   font=("Arial", 12, "bold"), bg='#f0f0f0')
        result_frame.pack(fill='x', pady=5)
        
        self.result_label = Label(result_frame, text="---", font=("Arial", 32, "bold"), 
                                   bg='#f0f0f0', fg='green')
        self.result_label.pack(pady=20)
        
        # Уверенность
        confidence_frame = LabelFrame(right_frame, text="Уверенность", 
                                       font=("Arial", 12, "bold"), bg='#f0f0f0')
        confidence_frame.pack(fill='x', pady=5)
        
        self.confidence_label = Label(confidence_frame, text="---", font=("Arial", 14), 
                                       bg='#f0f0f0')
        self.confidence_label.pack(pady=10)
        
        # Топ-5 альтернатив
        top5_frame = LabelFrame(right_frame, text="Топ-5 альтернатив", 
                                 font=("Arial", 12, "bold"), bg='#f0f0f0')
        top5_frame.pack(fill='both', expand=True, pady=5)
        
        self.top5_listbox = Listbox(top5_frame, height=5, font=("Arial", 11))
        self.top5_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Текущее выражение (в читаемом виде)
        expression_frame = LabelFrame(right_frame, text="Текущее выражение", 
                                       font=("Arial", 12, "bold"), bg='#f0f0f0')
        expression_frame.pack(fill='x', pady=5)
        
        self.expression_label = Label(expression_frame, text="", font=("Arial", 18, "bold"), 
                                       bg='#f0f0f0', fg='blue', wraplength=300)
        self.expression_label.pack(pady=10)
        
        # Кнопки
        button_frame = Frame(self.window, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        Button(button_frame, text="Распознать", command=self.recognize_symbol, 
               font=("Arial", 12), bg='#4CAF50', fg='white', padx=20, pady=5).pack(side='left', padx=5)
        
        Button(button_frame, text="Очистить поле", command=self.clear_canvas, 
               font=("Arial", 12), bg='#f44336', fg='white', padx=20, pady=5).pack(side='left', padx=5)
        
        Button(button_frame, text="Добавить в выражение", command=self.add_to_expression, 
               font=("Arial", 12), bg='#2196F3', fg='white', padx=20, pady=5).pack(side='left', padx=5)
        
        Button(button_frame, text="Очистить выражение", command=self.clear_expression, 
               font=("Arial", 12), bg='#FF9800', fg='white', padx=20, pady=5).pack(side='left', padx=5)
        
        Button(button_frame, text="Показать LaTeX", command=self.show_latex, 
               font=("Arial", 12), bg='#9C27B0', fg='white', padx=20, pady=5).pack(side='left', padx=5)
    
    def setup_canvas(self):
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        self.brush_size = 12
        self.last_x = None
        self.last_y = None
    
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                    width=self.brush_size, fill='black', 
                                    capstyle='round', smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=self.brush_size)
        self.last_x = x
        self.last_y = y
    
    def reset(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="---")
        self.confidence_label.config(text="---")
        self.top5_listbox.delete(0, tk.END)
        self.status_label.config(text="Поле очищено")
    
    def preprocess_image(self):
        resized = self.image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(resized) / 255.0
        img_array = img_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
        return img_array
    
    def recognize_symbol(self):
        img_array = np.array(self.image)
        if img_array.max() == 255 and img_array.min() == 255:
            self.status_label.config(text="Поле пустое! Нарисуйте символ.")
            return
        
        processed_img = self.preprocess_image()
        predictions = self.model.predict(processed_img, verbose=0)
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        
        best_idx = top5_indices[0]
        best_class = self.class_names[best_idx]
        best_confidence = predictions[0][best_idx] * 100
        
        # Показываем распознанный символ в читаемом виде
        display_symbol = to_display(best_class)
        self.result_label.config(text=display_symbol)
        self.confidence_label.config(text=f"{best_confidence:.1f}%")
        
        self.top5_listbox.delete(0, tk.END)
        for idx in top5_indices:
            class_name = self.class_names[idx]
            confidence = predictions[0][idx] * 100
            display_name = to_display(class_name)
            self.top5_listbox.insert(tk.END, f"{display_name}: {confidence:.1f}%")
        
        self.status_label.config(text=f"Распознано: {display_symbol} (уверенность {best_confidence:.1f}%)")
        self.last_recognized = best_class  # сохраняем оригинальное имя
    
    def add_to_expression(self):
        if self.last_recognized:
            self.expression_history.append(self.last_recognized)
            self.update_expression_display()
            self.status_label.config(text=f"Добавлено: {to_display(self.last_recognized)}")
            self.clear_canvas()
        else:
            self.status_label.config(text="Сначала распознайте символ!")
    
    def clear_expression(self):
        self.expression_history = []
        self.update_expression_display()
        self.status_label.config(text="Выражение очищено")
    
    def show_latex(self):
        if not self.expression_history:
            messagebox.showwarning("Предупреждение", "Нет выражения для отображения!")
            return
        
        latex_str = self.get_expression_latex()
        
        latex_window = tk.Toplevel(self.window)
        latex_window.title("LaTeX представление")
        latex_window.geometry("500x250")
        latex_window.configure(bg='#f0f0f0')
        
        Label(latex_window, text="LaTeX код:", font=("Arial", 12, "bold"), bg='#f0f0f0').pack(pady=10)
        
        # Показываем формулу
        formula_label = Label(latex_window, text=f"${latex_str}$", font=("Arial", 16), 
                               bg='#f0f0f0', fg='blue')
        formula_label.pack(pady=10)
        
        text_widget = tk.Text(latex_window, height=3, font=("Courier", 12))
        text_widget.pack(fill='x', expand=True, padx=20, pady=10)
        text_widget.insert('1.0', f"${latex_str}$")
        text_widget.config(state='disabled')
        
        def copy_to_clipboard():
            latex_window.clipboard_clear()
            latex_window.clipboard_append(f"${latex_str}$")
            messagebox.showinfo("Успех", "LaTeX скопирован в буфер обмена!")
        
        Button(latex_window, text="Копировать LaTeX", command=copy_to_clipboard,
               font=("Arial", 11), bg='#4CAF50', fg='white', padx=20).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = SymbolRecognizerApp(root)
    root.mainloop()