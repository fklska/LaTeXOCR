import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import torch
from torchvision import transforms as T
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import io
from models import TrOCR, ViT_TDecoder
from inference import inference_cnn_lstm, inference_attention_GRU
from matplotlib import pyplot as plt

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = "cpu"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Im2LaTeX")

        # Параметры кисти
        self.brush_size = tk.IntVar(value=5)
        self.drawing_color = (255, 255, 255)   # белый
        self.bg_color = (0, 0, 0)              # чёрный

        # PIL-изображение и инструмент рисования
        self.img = Image.new("RGB", (1366, 768), self.bg_color)
        self.draw = ImageDraw.Draw(self.img)

        # Предыдущие координаты для линии
        self.prev_x = None
        self.prev_y = None

        # Canvas
        self.canvas = tk.Canvas(root, width=1366, height=768, bg="black", cursor="cross")
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Панель управления
        panel = tk.Frame(root)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        tk.Label(panel, text="Размер кисти:").pack()
        tk.Scale(panel, from_=2, to=30, variable=self.brush_size, orient=tk.HORIZONTAL).pack()

        tk.Button(panel, text="Очистить", command=self.clear_canvas).pack(pady=5)
        tk.Button(panel, text="Распознать LaTeX", command=self.predict).pack(pady=5)

        tk.Label(panel, text="LaTeX-код:").pack()
        self.text_latex = tk.Text(panel, height=5, width=40)
        self.text_latex.pack()

        tk.Label(panel, text="Формула:").pack()
        self.render_label = tk.Label(panel)
        self.render_label.pack(pady=5)

        # Показать начальное изображение
        self.update_canvas()

    def update_canvas(self):
        self.tk_image = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def paint(self, event):
        x, y = event.x, event.y
        r = self.brush_size.get()
        if self.prev_x is not None and self.prev_y is not None:
            # Рисуем линию от предыдущей точки до текущей (сплошная кисть)
            self.canvas.create_line(
                self.prev_x, self.prev_y, x, y,
                fill="white", width=r, capstyle=tk.ROUND, smooth=True
            )
            self.draw.line(
                [self.prev_x, self.prev_y, x, y],
                fill=self.drawing_color, width=r
            )
        else:
            # Первая точка – рисуем круг
            self.canvas.create_oval(
                x - r//2, y - r//2, x + r//2, y + r//2,
                fill="white", outline="white"
            )
            self.draw.ellipse(
                [x - r//2, y - r//2, x + r//2, y + r//2],
                fill=self.drawing_color
            )
        self.prev_x = x
        self.prev_y = y

    def reset(self, event):
        self.prev_x = None
        self.prev_y = None

    def clear_canvas(self):
        self.img = Image.new("RGB", (1366, 768), self.bg_color)
        self.draw = ImageDraw.Draw(self.img)
        self.canvas.delete("all")
        self.update_canvas()
        self.text_latex.delete(1.0, tk.END)
        self.render_label.config(image="")

    def predict(self):
        # Предобработка
        img_input = self.img.copy()
        tensor = transform(img_input).unsqueeze(0).to(device)

        # Генерация (заменить на вызов реальной модели)
        latex_code = TrOCR.predict_latex(img_input)

        self.text_latex.delete(1.0, tk.END)
        self.text_latex.insert(tk.END, latex_code)
        self.render_latex(latex_code)

    def render_latex(self, latex_str):
        if not latex_str:
            return
        fig = Figure(figsize=(4, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        try:
            ax.text(0.5, 0.5, f"${latex_str}$", ha='center', va='center', fontsize=18,
                    transform=ax.transAxes)
        except:
            ax.text(0.5, 0.5, "Ошибка рендеринга", ha='center', va='center')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        pil_img = Image.open(buf)
        pil_img = pil_img.resize((400, 150))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.render_label.config(image=tk_img)
        self.render_label.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()