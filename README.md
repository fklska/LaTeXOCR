# LaTeXOCR

[Колаб](https://colab.research.google.com/drive/117HU_5Th7UJWIGrOgKiIeJrjZSDvco5V?usp=sharing) c базовыми моделями
[Колаб с трансформерами](https://colab.research.google.com/drive/1nOZ-FvKxIdTtEYvFr4FJzLkK0xiYP2qH?usp=sharing)
[Колаб с Qwen3-VL](https://colab.research.google.com/drive/10uYBsdJGk4oCAgwEQ2L0sTed10oB0OOm?usp=sharing)
[Колаб с TrOCR](https://colab.research.google.com/drive/1__Qugi9P3t6H9k5diwpPyXD1Gfymh7KR?usp=sharing)
[Презентация](https://docs.google.com/presentation/d/1p9upgW641H031HGDMKUfeG8mRdRJGd1KzzhaVzJ-q_g/edit?slide=id.g3e8c74becf6_0_0#slide=id.g3e8c74becf6_0_0)


## Запуск:
```bash
python -m venv venv
pip install -r requirements.txt
source venv/Scripts/activate
python main.py
```
Есть другие ветки с запуском YOLO модели, там свой Readme


В папке models, загрузка различных моделей по умолчанию стоит TrOCR-LaTex - файнтьюн TrOCR
Сами модели лежат в карточке https://huggingface.co/fklska с подписью OCR соответственно

!!! Важно у моделей разный интерфейс, нужно помемнять 116 строчку в файле main.py. Откровенно результат удалось добиться только на TrOCR которая стоит по умолчанию, CNN-RNN / TDecoder / GPT2 - очень плохо работают. 

Если хочется поставить CNN - RNN модели их инференс настроен через функции из файла `inference.py`, соответвсенно нужно импортировать нужную функцию и заменить строчку 116 в main.py, там строчка ожидает вывод модели. У ViT_GPT2 своя функция в файле, у Vit_TDecoder функция в классе модели 

Чтобы запустить Qwen3-VL нужно приложить некоторые усилия, скачать билд llama-cpp. Файлы с запуском Qwen3-VL в другой ветве