import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, Scale
from PIL import Image, ImageDraw, ImageTk
from inference import predict_from_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

window = tk.Tk()
window.title("LaTeX OCR — Блокнот для ввода формул")
window.geometry("1400x980")
window.configure(bg="#2d2d2d")

last_x, last_y = None, None
canvas_width, canvas_height = 1366, 768

# Поле для рисования
canvas = Canvas(window, width=canvas_width, height=canvas_height, bg='black', highlightthickness=0)
canvas.pack(pady=10)

image = Image.new("RGB", (canvas_width, canvas_height), color='black')
draw = ImageDraw.Draw(image)

def render_latex(latex_string):
    """Генерирует изображение формулы из строки LaTeX с помощью matplotlib"""
    if not latex_string.strip():
        return None
    try:
        fig = plt.figure(figsize=(10, 1.5), facecolor='#2d2d2d')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Оборачиваем в знаки доллара для активации MathText движка
        ax.text(0.5, 0.5, f"${latex_string}$", size=22, color='#2ecc71', ha='center', va='center')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return Image.open(buf)
    except Exception as e:
        print(f"Ошибка визуализации LaTeX: {e}")
        return None

def start_paint(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def paint(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x and last_y:
        r = brush_slider.get() # Получаем текущий радиус из ползунка
        canvas.create_line(last_x, last_y, x, y, fill='white', width=r*2, capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill='white', width=r*2)
    last_x, last_y = x, y

def stop_paint(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<Button-1>", start_paint)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", stop_paint)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill='black')
    result_entry.config(state='normal')
    result_entry.delete(0, tk.END)
    result_entry.insert(0, "Холст очищен")
    result_entry.config(state='readonly')
    render_label.config(image='')
    render_label.image = None

def recognize_expression():
    latex_result = predict_from_image(image)
    
    # Обновляем текстовое поле (делаем доступным для записи, меняем текст, закрываем)
    result_entry.config(state='normal')
    result_entry.delete(0, tk.END)
    
    if latex_result.strip() == "":
        result_entry.insert(0, "Формула не распознана")
        render_label.config(image='')
    else:
        result_entry.insert(0, latex_result)
        
        # Запускаем рендеринг визуального LaTeX
        rendered_img = render_latex(latex_result)
        if rendered_img:
            img_tk = ImageTk.PhotoImage(rendered_img)
            render_label.image = img_tk  # Сохраняем ссылку, чтобы сборщик мусора не удалил фото
            render_label.config(image=img_tk)
        else:
            render_label.config(image='')
            
    result_entry.config(state='readonly')

def copy_to_clipboard():
    window.clipboard_clear()
    text_to_copy = result_entry.get()
    if text_to_copy not in ["", "Холст очищен", "Формула не распознана"]:
        window.clipboard_append(text_to_copy)
        btn_copy.config(text="Скопировано!", bg="#f1c40f", fg="black")
        window.after(1500, lambda: btn_copy.config(text="Копировать строку", bg="#34495e", fg="white"))

# Панель инструментов
button_frame = Frame(window, bg="#2d2d2d")
button_frame.pack(fill=tk.X, padx=20, pady=5)

btn_clear = Button(button_frame, text="Очистить", command=clear_canvas, font=("Arial", 11, "bold"), bg="#e74c3c", fg="white", padx=10, pady=5)
btn_clear.pack(side=tk.LEFT, padx=5)

# Ползунок размера кисти
brush_slider = Scale(button_frame, from_=1, to=20, orient=tk.HORIZONTAL, label="Размер кисти", bg="#2d2d2d", fg="white", highlightthickness=0, font=("Arial", 9))
brush_slider.set(5)
brush_slider.pack(side=tk.LEFT, padx=15)

btn_recognize = Button(button_frame, text="Распознать LaTeX", command=recognize_expression, font=("Arial", 11, "bold"), bg="#2ecc71", fg="white", padx=15, pady=5)
btn_recognize.pack(side=tk.LEFT, padx=5)

# Окно вывода текстового кода (теперь это Entry с возможностью выделения и копирования)
result_entry = tk.Entry(button_frame, font=("Consolas", 14), bg="#1e1e1e", fg="#f1c40f", bd=5, relief=tk.FLAT, width=50)
result_entry.insert(0, "Нарисуйте формулу и нажмите 'Распознать LaTeX'")
result_entry.config(state='readonly')
result_entry.pack(side=tk.LEFT, padx=15, fill=tk.X, expand=True)

btn_copy = Button(button_frame, text="Копировать строку", command=copy_to_clipboard, font=("Arial", 11, "bold"), bg="#34495e", fg="white", padx=10, pady=5)
btn_copy.pack(side=tk.RIGHT, padx=5)

# Поле для вывода отрендеренной картинки LaTeX
render_label = Label(window, bg="#2d2d2d")
render_label.pack(pady=10)

window.mainloop()