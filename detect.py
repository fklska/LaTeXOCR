import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO

yolo_model = YOLO('best.pt')
classes = {0: '!', 1: '(', 2: ')', 3: '+', 4: ',', 5: '-', 6: '.', 7: '/',
    8: '0', 9: '1', 10: '2', 11: '3', 12: '4', 13: '5', 14: '6', 15: '7',
    16: '8', 17: '9', 18: '=', 19: 'A', 20: 'B', 21: 'C', 22: 'E', 23: 'F',
    24: 'G', 25: 'H', 26: 'I', 27: 'L', 28: 'M', 29: 'N', 30: 'P', 31: 'R',
    32: 'S', 33: 'T', 34: 'V', 35: 'X', 36: 'Y', 37: '[', 38: '\\Delta',
    39: '\\alpha', 40: '\\beta', 41: '\\cos', 42: '\\div', 43: '\\exists',
    44: '\\forall', 45: '\\gamma', 46: '\\geq', 47: '\\gt', 48: '\\in',
    49: '\\infty', 50: '\\int', 51: '\\lambda', 52: '\\ldots', 53: '\\leq',
    54: '\\lim', 55: '\\log', 56: '\\lt', 57: '\\mu', 58: '\\neq',
    59: '\\phi', 60: '\\pi', 61: '\\pm', 62: '\\prime', 63: '\\rightarrow',
    64: '\\sigma', 65: '\\sin', 66: '\\sqrt', 67: '\\sum', 68: '\\tan',
    69: '\\theta', 70: '\\times', 71: '\\{', 72: '\\}', 73: ']', 74: 'a',
    75: 'b', 76: 'c', 77: 'd', 78: 'e', 79: 'f', 80: 'g', 81: 'h', 82: 'i',
    83: 'j', 84: 'k', 85: 'l', 86: 'm', 87: 'n', 88: 'o', 89: 'p', 90: 'q',
    91: 'r', 92: 's', 93: 't', 94: 'u', 95: 'v', 96: 'w', 97: 'x', 98: 'y', 99: 'z', 100: '|'}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Formula Recognizer")
        self.root.geometry("1400x700")
        
        # Левая панель - рисование (большая)
        left_frame = tk.Frame(root, width=700)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="1. Нарисуйте формулу", font=('Arial', 12, 'bold')).pack()
        self.canvas = tk.Canvas(left_frame, width=650, height=500, bg='white', relief='sunken')
        self.canvas.pack(pady=5, fill='both', expand=True)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        self.last_x, self.last_y = None, None
        self.lines = []
        
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Распознать", command=self.recognize, bg='green', fg='white', font=('Arial', 10), width=12).pack(side='left', padx=10)
        tk.Button(btn_frame, text="Очистить", command=self.clear, bg='red', fg='white', font=('Arial', 10), width=12).pack(side='left', padx=10)
        
        # Правая панель (два окна)
        right_frame = tk.Frame(root, width=550)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        right_frame.pack_propagate(False)
        
        # Верхнее правое окно - изображение с разметкой
        top_frame = tk.LabelFrame(right_frame, text="2. Изображение с разметкой YOLO", font=('Arial', 10, 'bold'))
        top_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.image_label = tk.Label(top_frame, relief='sunken', bg='white')
        self.image_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Нижнее правое окно - символы
        bottom_frame = tk.LabelFrame(right_frame, text="3. Распознанные символы", font=('Arial', 10, 'bold'))
        bottom_frame.pack(fill='both', expand=True)
        
        self.symbols_text = tk.Text(bottom_frame, width=30, height=10, font=('Courier', 11))
        self.symbols_text.pack(pady=10, padx=10, fill='both', expand=True)
        
        self.photo_image = None
        
    def draw(self, event):
        if self.last_x and self.last_y:
            line = self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=3, fill='black')
            self.lines.append(line)
        self.last_x, self.last_y = event.x, event.y
    
    def reset(self, event):
        self.last_x, self.last_y = None, None
    
    def clear(self):
        self.canvas.delete("all")
        self.lines.clear()
        self.symbols_text.delete(1.0, tk.END)
        self.image_label.config(image='')
        self.photo_image = None
    
    def canvas_to_image(self):
        img = Image.new('RGB', (self.canvas.winfo_width(), self.canvas.winfo_height()), 'white')
        draw = ImageDraw.Draw(img)
        
        for item in self.lines:
            coords = self.canvas.coords(item)
            if len(coords) >= 4:
                for i in range(0, len(coords)-2, 2):
                    draw.line([coords[i], coords[i+1], coords[i+2], coords[i+3]], fill='black', width=3)
        
        img.save("temp.png")
        return "temp.png"
    
    def recognize(self):
        if not self.lines:
            return
        
        img_path = self.canvas_to_image()
        results = yolo_model(img_path)
        boxes = results[0].boxes
        
        img_cv = cv2.imread(img_path)
        h, w = img_cv.shape[:2]
        
        if boxes and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            indices = np.argsort(xyxy[:, 0])
            
            symbols = []
            for idx in indices:
                sym = classes.get(cls_ids[idx], '?')
                symbols.append(sym)
                
                x1, y1, x2, y2 = map(int, xyxy[idx])
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv, sym, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            self.symbols_text.delete(1.0, tk.END)
            self.symbols_text.insert(1.0, ' '.join(symbols))
            
            # Пропорциональное изменение размера
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Сохраняем пропорции
            max_size = 400
            ratio = min(max_size / w, max_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_pil = img_pil.resize((new_w, new_h))
            
            self.photo_image = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=self.photo_image)
        else:
            self.symbols_text.delete(1.0, tk.END)
            self.symbols_text.insert(1.0, "Ничего не найдено")

# root = tk.Tk()
# app = App(root)
# root.mainloop()

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()