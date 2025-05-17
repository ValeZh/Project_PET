import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
import os
# === ФУНКЦІЇ ДЛЯ ОБРОБКИ ВІДЕО І КАДРІВ ===

# Завантаження конкретного кадру з відео за його номером
def load_frame_by_number(video_path, frame_number):
    """
    Завантажує вказаний кадр з відеофайлу.

    :param video_path: Шлях до відеофайлу.
    :param frame_number: Номер кадру, який потрібно витягнути.
    :return: Кортеж (успішність: bool, кадр: np.ndarray)
    """
    cap = cv2.VideoCapture(video_path)  # відкриваємо відеофайл
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # загальна кількість кадрів
    frame_number = max(0, min(frame_number, total_frames - 1))  # обмежуємо номер кадру в допустимих межах
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # переходимо до потрібного кадру
    success, frame = cap.read()  # читаємо кадр
    cap.release()  # закриваємо відео
    return success, frame  # повертаємо статус та сам кадр

# Перетворення зображення з BGR (стандарт у OpenCV) у HSV та RGB
def convert_color_spaces(frame):
    """
    Перетворює кольорове зображення з BGR у HSV та RGB простори.

    :param frame: Зображення у форматі BGR (np.ndarray).
    :return: Кортеж (HSV-зображення, RGB-зображення)
    """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # для виділення кольору
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # для збереження у правильному вигляді
    return frame_hsv, frame_rgb

# Застосування морфологічних операцій до маски для очищення шумів
def clean_mask(mask, kernel_size=(3, 3), iterations=1):
    """
    Очищає бінарну маску за допомогою морфологічних операцій (open+close).

    :param mask: Вхідна бінарна маска (np.ndarray).
    :param kernel_size: Розмір ядра для морфологічних операцій.
    :param iterations: Кількість ітерацій операцій.
    :return: Очищена маска (np.ndarray)
    """
    kernel = np.ones(kernel_size, np.uint8)  # створення ядра для морфології
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=iterations)  # видалення шумів
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)  # заповнення дірок
    return mask

# Отримання контурів об'єктів з бінарної маски
def get_contours(mask):
    """
    Виділяє контури на основі бінарної маски.

    :param mask: Бінарна маска (np.ndarray).
    :return: Список контурів (List[np.ndarray]).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Малює мінімальні охоплюючі кола навколо кожного знайденого об'єкта
# Повертає shape_map — словник з центром об'єкта, площею та коефіцієнтом форми
def draw_enclosing_circles(image, contours, min_area=5):
    """
      Малює мінімальні кола навколо кожного контуру і обчислює площу та коефіцієнт форми.

      :param image: Зображення (RGB або подібне), на якому малюється.
      :param contours: Список контурів.
      :param min_area: Мінімальна площа для об'єкта, щоб бути врахованим.
      :return: Кортеж (зображення з колами, булевий прапорець знайдених, shape_map)
               shape_map: Dict[(x:int, y:int), (area:int, form_coef:float)]
      """
    img_out = image.copy()  # копія зображення для малювання
    shape_map = {}  # {(x,y): (area, form_coef)} — інформація по кожному об'єкту
    any_found = False
    for cnt in contours:
        area = cv2.contourArea(cnt)  # площа контуру
        if area > min_area:  # ігноруємо дуже малі об'єкти
            (x, y), r = cv2.minEnclosingCircle(cnt)  # мінімальне коло, що охоплює контур
            center = (int(x), int(y))
            r = int(r)
            if r > 0:
                form_coef = round(area / (np.pi * r * r), 3)  # коефіцієнт форми = площа / площа кола
                cv2.circle(img_out, center, r, (0,255,0), 2)  # малюємо коло
                shape_map[center] = (int(area), form_coef)  # зберігаємо дані
                any_found = True
    # Сортуємо об'єкти за площею у спадному порядку
    shape_map = dict(sorted(shape_map.items(), key=lambda item: item[1][0], reverse=True))
    return img_out, any_found, shape_map

# Створює маску для заданого відтінку (Hue) у HSV просторі
def get_color_mask(hsv_img, hue_center, hue_range=10):
    """
        Створює бінарну маску для вказаного відтінку (hue) у HSV просторі.

        :param hsv_img: Зображення у HSV просторі.
        :param hue_center: Центральне значення hue (0–179).
        :param hue_range: Допуск навколо hue_center.
        :return: Бінарна маска (np.ndarray)
        """
    low = np.array([max(hue_center - hue_range, 0), 100, 100])  # нижня межа по H, S, V
    high= np.array([min(hue_center + hue_range, 179), 255, 255])  # верхня межа
    return cv2.inRange(hsv_img, low, high)  # бінарна маска

# === АНАЛІЗ КАДРІВ ЗА КОЛЬОРОМ ===

# Аналіз одного кольору: створення маски, пошук об'єктів, обрахунок площі і форми
def analyze_single_color(video_path, frame_number, hue_center):
    """
      Аналізує один кадр на наявність об'єктів заданого кольору (hue).

      :param video_path: Шлях до відеофайлу.
      :param frame_number: Номер кадру для аналізу.
      :param hue_center: Значення відтінку кольору у HSV-просторі.
      :return: Словник з результатами:
          {
              "shape_map": Dict[(x, y), (area:int, form_coef:float)],
              "paths":     Dict[str, str],  # шляхи до зображень
              "area":      int  # загальна площа маски
          }
      """
    success, frame = load_frame_by_number(video_path, frame_number)
    if not success:
        raise RuntimeError(f"Не вдалося зчитати кадр {frame_number}")

    hsv, rgb = convert_color_spaces(frame)
    mask = clean_mask(get_color_mask(hsv, hue_center))  # створення маски і її очищення
    contours = get_contours(mask)  # отримання контурів
    masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)  # накладаємо маску на оригінал
    circ_img, found, shape_map = draw_enclosing_circles(masked_rgb, contours)  # будуємо кола і форму

    # Збереження оброблених зображень
    out_dir = f"saved_frames/frame_{frame_number}"
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "original": os.path.join(out_dir, f"hue_{hue_center}_original.png"),
        "mask":     os.path.join(out_dir, f"hue_{hue_center}_mask.png"),
        "circles":  os.path.join(out_dir, f"hue_{hue_center}_circles.png"),
    }
    cv2.imwrite(paths["original"], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["mask"],     cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["circles"],  cv2.cvtColor(circ_img,    cv2.COLOR_RGB2BGR))

    area = int(cv2.countNonZero(mask))  # загальна площа маски

    return {
        "shape_map": shape_map,  # {(x, y): (area, form_coef)}
        "paths":     paths,
        "area":      area
    }

# Порівняння двох кольорів: аналіз hue1 і hue2, повертає shape_map для кожного
def compare_two_colors_by_area(video_path, frame_number, hue1, hue2):
    """
    Порівнює два відтінки (hue) на одному кадрі за площею та формою знайдених об'єктів.

    :param video_path: Шлях до відеофайлу.
    :param frame_number: Номер кадру для аналізу.
    :param hue1: Перше значення hue.
    :param hue2: Друге значення hue.
    :return: Словник {hue: результат як у analyze_single_color()}
    """
    success, frame = load_frame_by_number(video_path, frame_number)
    if not success:
        raise RuntimeError(f"Не вдалося зчитати кадр {frame_number}")

    hsv, rgb = convert_color_spaces(frame)
    out_dir = f"saved_frames/frame_{frame_number}"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for hue in (hue1, hue2):
        mask = clean_mask(get_color_mask(hsv, hue))
        contours = get_contours(mask)
        masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
        circ_img, found, shape_map = draw_enclosing_circles(masked_rgb, contours)

        # Збереження результатів по кожному hue
        paths = {
            "original": os.path.join(out_dir, f"hue_{hue}_original.png"),
            "mask":     os.path.join(out_dir, f"hue_{hue}_mask.png"),
            "circles":  os.path.join(out_dir, f"hue_{hue}_circles.png"),
        }
        cv2.imwrite(paths["original"], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(paths["mask"],     cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(paths["circles"],  cv2.cvtColor(circ_img,    cv2.COLOR_RGB2BGR))

        area = int(cv2.countNonZero(mask))  # площа всієї кольорової області

        results[hue] = {
            "shape_map": shape_map,
            "paths":     paths,
            "area":      area
        }

    return results

def save_analysis_results_to_file(filename, hue_data_dict, frame_number):
    """
    Зберігає результати аналізу в конкретний файл, без створення папок.
    :param filename: Повна назва вихідного .txt файлу (без створення папок).
    :param hue_data_dict: Словник {hue: {'shape_map': ..., 'area': ..., 'paths': ...}}.
    :param frame_number: Номер кадру, додається у текст.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Аналіз кадру #{frame_number}\n")
        f.write("=" * 50 + "\n")

        for hue, data in hue_data_dict.items():
            f.write(f"\n🎨 КОЛІР (Hue) = {hue}\n")
            f.write(f"Загальна площа: {data['area']} пікселів\n")
            f.write(f"Кількість об'єктів: {len(data['shape_map'])}\n")

            for i, ((x, y), (area, coef)) in enumerate(data['shape_map'].items(), start=1):
                f.write(f"\n   Об'єкт #{i}\n")
                f.write(f"      Центр: ({x}, {y})\n")
                f.write(f"      Площа: {area} пікселів\n")
                f.write(f"      Коефіцієнт форми: {coef}\n")

        f.write("\nЗавершено.\n")

    print(f"✅ Результати збережено у файл: {filename}")

# Функція для створення PDF-звіту
def generate_pdf_report(filename, frame_number, best_hue, best_data):
    """
    Створює PDF-звіт з результатами аналізу для hue з найбільшою площею.
    Використовується шрифт DejaVuSans з підтримкою кирилиці.

    :param filename: Назва PDF-файлу (наприклад: "report.pdf").
    :param frame_number: Номер кадру.
    :param best_hue: Hue з найбільшою площею.
    :param best_data: Словник {"shape_map", "paths", "area"}.
    """
    font_name = "DejaVuSans"
    try:
        pdfmetrics.registerFont(TTFont(font_name, "DejaVuSans.ttf"))
    except Exception as e:
        print("⚠️ Не знайдено DejaVuSans.ttf. Використовується Times-Roman.")
        font_name = "Times-Roman"

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Заголовок
    c.setFont(font_name, 16)
    c.drawString(2 * cm, height - 2 * cm, f"Звіт аналізу кадру №{frame_number}")

    # Основна інформація
    c.setFont(font_name, 12)
    c.drawString(2 * cm, height - 3.2 * cm, f"Hue з найбільшою площею: {best_hue}")
    c.drawString(2 * cm, height - 4 * cm, f"Загальна площа: {best_data['area']} пікселів")
    c.drawString(2 * cm, height - 4.8 * cm, f"Кількість знайдених об'єктів: {len(best_data['shape_map'])}")

    # Додаємо зображення
    image_path = best_data["paths"]["circles"]
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img.thumbnail((15 * cm, 15 * cm))
        img_path_temp = "temp_report_image.jpg"
        img.save(img_path_temp)
        c.drawImage(img_path_temp, 2 * cm, height - 17 * cm)

    # Виводимо до 5 об'єктів
    c.setFont(font_name, 11)
    y = height - 18.5 * cm
    for i, ((x, y_pos), (area, coef)) in enumerate(best_data["shape_map"].items()):
        if i >= 5:
            break
        c.drawString(2 * cm, y, f"Об'єкт #{i+1}: Центр=({x}, {y_pos}), Площа={area}, КФ={coef}")
        y -= 0.6 * cm

    c.showPage()
    c.save()
    print(f"✅ PDF звіт збережено: {filename}")

# === ГОЛОВНА ТОЧКА ВХОДУ ===

def main():
    video_path = "./istockphoto-1256494688-640_adpp_is.mp4"
    frame_number = 1
    hue1 = 0
    hue2 = 30

    # Аналіз одного кольору
    print("\n🎨 === АНАЛІЗ ОДНОГО КОЛЬОРУ ===")
    result = analyze_single_color(video_path, frame_number, hue_center=hue1)
    print(f"\n▶️ Hue = {hue1}")
    print(f"Загальна площа маски: {result['area']} пікселів")
    print(f"Кількість знайдених об'єктів: {len(result['shape_map'])}")

    for i, ((x, y), (area, coef)) in enumerate(result['shape_map'].items(), start=1):
        print(f"\n🔹 Об'єкт #{i}")
        print(f"   Центр: ({x}, {y})")
        print(f"   Площа: {area} пікселів")
        print(f"   Коефіцієнт форми: {coef}")

    # Зберігаємо результат одного кольору у файл
    save_analysis_results_to_file("results", {hue1: result}, frame_number)

    # Порівняння двох кольорів
    print("\n⚖️ === ПОРІВНЯННЯ ДВОХ КОЛЬОРІВ ===")
    comparison = compare_two_colors_by_area(video_path, frame_number, hue1, hue2)

    for hue in [hue1, hue2]:
        data = comparison[hue]
        print(f"\n🎯 Hue = {hue}")
        print(f"Загальна площа маски: {data['area']} пікселів")
        print(f"Кількість знайдених об'єктів: {len(data['shape_map'])}")

        for i, ((x, y), (area, coef)) in enumerate(data['shape_map'].items(), start=1):
            print(f"\n   🔸 Об'єкт #{i}")
            print(f"      Центр: ({x}, {y})")
            print(f"      Площа: {area} пікселів")
            print(f"      Коефіцієнт форми: {coef}")

    # Зберігаємо результати обох кольорів у файл
    save_analysis_results_to_file("compare_hues_0_30.txt", comparison, frame_number)


    # === Генеруємо PDF для hue з найбільшою площею ===
    best_hue = max(comparison.items(), key=lambda item: item[1]["area"])[0]
    best_data = comparison[best_hue]
    generate_pdf_report("analysis_report.pdf", frame_number, best_hue, best_data)


if __name__ == "__main__":
    main()
