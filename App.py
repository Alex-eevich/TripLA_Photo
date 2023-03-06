import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Label

from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np

root = tk.Tk()
root.title("TripLA_Photo.exe")

# создание главного канваса
main_canvas = tk.Canvas(root, width=1262, height=655, bg='#474A51')
main_canvas.pack()

# создание внутреннего канваса для изображения
image_canvas = tk.Canvas(main_canvas, width=900, height=600, bg='white')
image_canvas.place(x=10, y=45)

# создание ползунка для гамма-коррекции1
gamma_slider1 = tk.Scale(main_canvas, from_=1.0, to=10.0, orient=tk.HORIZONTAL, length=300, label="Гамма-коррекция (light -)")
gamma_slider1.place(x=950, y=90)
# создание ползунка для гамма-коррекции2
gamma_slider2 = tk.Scale(main_canvas, from_=1.0, to=10.0, orient=tk.HORIZONTAL, length=300, label="Гамма-коррекция (light +)")
gamma_slider2.place(x=950, y=200)

# создание ползунка для соляризации
solarize_slider = tk.Scale(main_canvas, label="Соляризация", from_=0, to=255, orient=tk.HORIZONTAL, length=300)
solarize_slider.place(x=950, y=310)

# создание ползунка для степенного преобразования
power_slider = tk.Scale(main_canvas, label="Степенное преобразование", from_=3, to=11, orient=tk.HORIZONTAL, length=300)
power_slider.place(x=950, y=420)


def load_image():

    # открытие диалогового окна выбора файла
    filename = filedialog.askopenfilename(initialdir="/", title="Выберите изображение", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))

    # загрузка изображения
    img1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_height, img_width = img1.shape

    image_canvas.imageX = img_width
    image_canvas.imageY = img_height

    # конвертация изображения из BGR в RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # задаём изображению размеры канваса для грамотного отображения
    resized_width = image_canvas.winfo_width()
    resized_height = image_canvas.winfo_height()

    # меняем размер изображения
    img = cv2.resize(img, (resized_width, resized_height))

    # создание объекта изображения Pillow
    pil_image = Image.fromarray(img)

    # создание объекта изображения Tkinter
    tk_image = ImageTk.PhotoImage(master=main_canvas, image=pil_image)

    # сохранение ссылки на объект изображения
    image_canvas.image = tk_image

    # отображение изображения на канвасе
    image_canvas.create_image(0, 0, image=tk_image, anchor=tk.NW)

    image_canvas.image1 = img
    image_canvas.image_res = img
    return img



def apply_filter1_1():
    # получаем изображение из канваса
    pil_image = image_canvas.image1
    img = np.asarray(pil_image)

    # применение гамма-коррекции
    gamma = gamma_slider1.get()
    img = np.power(img.astype(float) / 255.0, gamma)
    img = (img * 255).astype('uint8')

    # создание объекта изображения Pillow
    pil_image = Image.fromarray(img)

    # создание объекта изображения Tkinter
    tk_image = ImageTk.PhotoImage(master=main_canvas, image=pil_image)

    # сохранение ссылки на объект изображения
    image_canvas.image = tk_image

    # отображение изображения на канвасе
    image_canvas.create_image(0, 0, image=tk_image, anchor=tk.NW)

    image_canvas.image1 = img


def apply_filter1_2():
    # получаем изображение из канваса
    pil_image = image_canvas.image1
    img = np.asarray(pil_image)

    # применение гамма-коррекции
    gamma = gamma_slider2.get() / 10
    img = np.power(img.astype(float) / 255.0, gamma)
    img = (img * 255).astype('uint8')

    # создание объекта изображения Pillow
    pil_image = Image.fromarray(img)

    # создание объекта изображения Tkinter
    tk_image = ImageTk.PhotoImage(master=main_canvas, image=pil_image)

    # сохранение ссылки на объект изображения
    image_canvas.image = tk_image

    # отображение изображения на канвасе
    image_canvas.create_image(0, 0, image=tk_image, anchor=tk.NW)

    image_canvas.image1 = img

def apply_filter2():
    # получаем изображение из канваса
    pil_image = image_canvas.image1
    img = np.asarray(pil_image)

    # применение соляризации
    '''threshold = solarize_slider.get()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(img)
    mask = cv2.inRange(v, threshold, 255 - threshold)
    v = cv2.bitwise_xor(v, mask)

    img = cv2.merge((h, s, v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)'''

    threshold = solarize_slider.get()
    img = np.where(img < threshold, 255 - img, img)
    img = np.where(img >= threshold, 255 - img, img)

    # создание объекта изображения Pillow
    pil_image = Image.fromarray(img)

    # создание объекта изображения Tkinter
    tk_image = ImageTk.PhotoImage(master=main_canvas, image=pil_image)

    # сохранение ссылки на объект изображения
    image_canvas.image = tk_image

    # отображение изображения на канвасе
    image_canvas.create_image(0, 0, image=tk_image, anchor=tk.NW)

    image_canvas.image1 = img

def apply_filter3():
    pil_image = image_canvas.image1
    img = np.asarray(pil_image)

    # применение степенного преобразования
    power = power_slider.get()
    if power > 1 and power % 2 == 1:
        '''img_gray = Image.fromarray(img).convert('L')
        img_eq = ImageOps.equalize(img_gray)
        img_pow = ImageOps.autocontrast(img_eq).point(lambda x: ((x / 255.0) - 0.5) * 2.0 if x > 0 else 0).point(lambda x: pow(x, power))
        img = np.array(img_pow.convert('RGB'))'''

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32')
        img = img / 255.0  # нормализация значений от 0 до 1
        img = 5 * np.power(img, power) - 32
        img = img * 255.0  # масштабирование значений до диапазона [0, 255]
        img = img.astype('uint8')

        # создание объекта изображения Pillow
        pil_image = Image.fromarray(img)

        # создание объекта изображения Tkinter
        tk_image = ImageTk.PhotoImage(master=main_canvas, image=pil_image)

        # сохранение ссылки на объект изображения
        image_canvas.image = tk_image

        # отображение изображения на канвасе
        image_canvas.create_image(0, 0, image=tk_image, anchor=tk.NW)

        image_canvas.image1 = img


def reset_filters():
    pil_image = image_canvas.image_res
    img = np.asarray(pil_image)

    # создание объекта изображения Pillow
    pil_image = Image.fromarray(img)

    # создание объекта изображения Tkinter
    tk_image = ImageTk.PhotoImage(master=main_canvas, image=pil_image)

    # сохранение ссылки на объект изображения
    image_canvas.image = tk_image

    # отображение изображения на канвасе
    image_canvas.create_image(0, 0, image=tk_image, anchor=tk.NW)

    image_canvas.image1 = img

def save_image():
    aX = image_canvas.imageX
    aY = image_canvas.imageY
    pil_image = image_canvas.image1
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")])
    if not save_path:
        return
    resized_image = cv2.resize(pil_image, (aX, aY))
    if resized_image is not None:
        resized_pil_image = Image.fromarray(resized_image)
        resized_pil_image.save(save_path)
    else:
        print("Ошибка при изменении размера изображения.")



# создание кнопки для применения фильтров
button_apply_filters = tk.Button(main_canvas, text="Сбросить изменения", command=reset_filters)
button_apply_filters.place(x=950, y=600)

# создание кнопки для применения фильтров
button_apply_filters = tk.Button(main_canvas, text="Скачать изображение", command=save_image)
button_apply_filters.place(x=1125, y=600)

# создание кнопки для загрузки изображения
button_load = tk.Button(main_canvas, text="Загрузить изображение", command=load_image)
button_load.place(x=10, y=10)

# создание кнопки для загрузки изображения
button_load = tk.Button(main_canvas, text="Применить", command=apply_filter1_1)
button_load.place(x=1180, y=160)

# создание кнопки для загрузки изображения
button_load = tk.Button(main_canvas, text="Применить", command=apply_filter1_2)
button_load.place(x=1180, y=270)

# создание кнопки для загрузки изображения
button_load = tk.Button(main_canvas, text="Применить", command=apply_filter2)
button_load.place(x=1180, y=380)

# создание кнопки для загрузки изображения
button_load = tk.Button(main_canvas, text="Применить", command=apply_filter3)
button_load.place(x=1180, y=490)


# запуск главного цикла обработки событий
root.mainloop()