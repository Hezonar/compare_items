""" v2.0
 - Теперь картинки хранятся на локальном компьютере
 - Программа не завершается без слова stop.
 - Каждый раз на вход подается -f PATH -s PATH и выполняется программа

 - Сделать path_new (отдельную переменную), т.к. теперь есть как URL, так и path_file
 - Сделать текстовый файл обработанных JSON's и добавлять туда каждый раз новый JSON
 - Итог сохранять в папку
"""
import config
import logging
import datetime
import time
import json
import shutil
import os
import requests
from pathlib import Path

import tensorflow as tf
import turicreate as tc
import numpy as np

import nltk
from nltk.corpus import stopwords
from langdetect import detect
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def find_new_json():
    files_in_dir = os.listdir(config.path_to_scan_json)
    for j in range(len(files_in_dir)):
        if files_in_dir[j] != config.json_done_db:
            if files_in_dir[j] not in done_json:
                if files_in_dir[j] != '':
                    if files_in_dir[j] != -1:
                        return config.path_to_scan_json + files_in_dir[j]
    return -1


def cosin(a, b):
    """
    Функция считает cos расстояние
    """
    aLength = np.linalg.norm(a)
    bLength = np.linalg.norm(b)

    return np.dot(a, b) / (aLength * bLength)

def str_to_vector(text):
    """
    Функция считает веса по кол-ву слов в тексте
    """
    try:
        words = word_tokenize(text)
        filtered_words = []

        for w in words:
            if w.lower() not in stop_words:
                filtered_words.append(w.lower())

        dict_with_words = dict()

        for word in filtered_words:
            if word in dict_with_words:
                dict_with_words[word] += 1
            else:
                dict_with_words[word] = 1

        count_all_words = len(filtered_words)

        for word in dict_with_words:
            dict_with_words[word] = dict_with_words[word] / count_all_words

        return dict_with_words

    except Exception as e:
        if config.is_debug:
            print(e)
        return []



def find_cos_range_2_text(text1, text2, lang_detect):
    try:
        if lang_detect:
            if detect(text1) != 'ru':
                text1 = GoogleTranslator(source=detect(text1), target='russian').translate(text1)
            if detect(text2) != 'ru':
                text1 = GoogleTranslator(source=detect(text2), target='russian').translate(text2)

        our_article = str_to_vector(text1)
        next_article = str_to_vector(text2)

        if our_article and next_article:
            for w in our_article:
                if w not in next_article:
                    next_article[w] = 0

            for w in next_article:
                if w not in our_article:
                    our_article[w] = 0

            vector_our_article = []
            vector_next_article = []

            for word in our_article:
                vector_our_article.append(our_article[word])
                vector_next_article.append(next_article[word])

            return cosin(vector_our_article, vector_next_article)
        else:
            return -1
    except Exception as e:
        if config.is_debug:
            print(e)
        return -1


def compare_price(price1, price2):
        try:
            if float(price2) > float(price1):
                price_sim = (float(price1) / float(price2))
            else:
                price_sim = (float(price2) / float(price1))
            return price_sim
        except:
            return -1


def read_img_from_dir(path, expand=False):
    """
    Предобработка изображения по его пути
    """
    # expand - добавить доп. слой или нет
    try:
        img_one = tf.io.read_file(path)
        img_one = tf.image.convert_image_dtype(tf.io.decode_png(img_one, channels=3), dtype='float32')  # * 1./255
        img_one = tf.image.resize(img_one, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
        if expand:
            img_one_final = tf.expand_dims(img_one, 0)
        else:
            img_one_final = img_one

        return img_one_final

    except Exception as e:
        if config.is_debug:
            print(e)
        logging.error("Couldn't open image from {}".format(path))
        return -1


def CompareSiamese():
    """
    Быстрое сравнение Сиамской сетью
    """
    try:
        compare_image_one = []
        compare_image_two = []

        img1 = read_img_from_dir(new_path_original)
        for j in range(len(new_path_other)):
            img2 = read_img_from_dir(new_path_other[j])
            compare_image_one.append(img1)
            compare_image_two.append(img2)

        comparing_image = [tf.convert_to_tensor(compare_image_one),
                           tf.convert_to_tensor(compare_image_two)]
        return siamese_model.predict(comparing_image)

    except Exception as e:
        if config.is_debug:
            print(e)
        logging.error("Can't calculate fast Siamese Model!")
        return []


def calc_identity(j):
    # j - номер сравнимаего объекта
    # Каждый параметр - 1 поинт
    # Можно менять вес параметра
    max_points = 0
    points = 0
    if config.is_debug:
        print("Comparing with ads_id - {}".format(data["other_goods"][j]["ads_id"]))
        print("Image link - {}".format(data["other_goods"][j]["image"]))

    # turi create
    if ranks_img[j] != -1:
        a = 1 - ranks_img[j]/len(distance_imgs_turi)  # от 0 до 1. БЕССМЫСЛЕННО, если изображений мало
        points += config.w_tc * a  # вес = 1
        max_points += config.w_tc
        if config.is_debug:
            print(f"turi model points - {a}, range[0:1]")

    else:
        logging.error("Can't compare image turi model with ads_id - {}".format(data["other_goods"][j]["ads_id"]))

    # siamese model
    if sm_grade[j] != -1:
        ms = sm_grade[j]  # от 0 до 1
        points += config.w_sm * ms  # вес = 1
        max_points += config.w_sm
        if config.is_debug:
            print(f"siamese model points - {ms}, range[0:1]")
    else:
        logging.error("Can't compare image siamese model with ads_id - {}".format(data["other_goods"][j]["ads_id"]))


    # names
    if names_cos_ranges[j] != -1:
        c = names_cos_ranges[j] * 1.5  # от 0 до 1, но заниженная оценка
        if c > 1:
            c = 1
        points += config.w_name * c  # вес = 1
        max_points += config.w_name
        if config.is_debug:
            print(f"name cos range - {c}, range[0:]")

    # desc
    if desc_cos_ranges[j] != -1:
        d = desc_cos_ranges[j] * 1.5  # от 0 до 1, но заниженная оценка
        if d > 1:
            d = 1
        points += config.w_desc * d  # вес = 1
        max_points += config.w_desc
        if config.is_debug:
            print(f"description cos range - {d}, range[0:]")

    # price
    if price_similarity[j] != -1:
        e = price_similarity[j]  # от 0 до 1
        points += config.w_price * e  # вес = 0.1
        max_points += config.w_price
        if config.is_debug:
            print(f"price similarity - {e}, range[0:1]")

    if config.is_debug:
        print("Current points - ", points)
        print("Max points of this item - ", max_points)
        print("TOTAL GRADE - ", points / max_points)
        print()

    if max_points != 0:
        return points / max_points
    else:
        logging.error(f"NO COMPARING WITH ADS_ID - {data['other_goods'][j]['ads_id']}")
        return -1


def check_url_local(url, orig=False):
    # проверить url на локальном компьютере или на сервере
    # загрузить на локальный, если на сервере
    if url[:5] == 'https':
        path = load_image_to_dir(url, orig=orig)
        del_jpgs.append(path)
        return path
    else:
        if os.path.exists(url):
            if url[-4:] == 'webp':
                shutil.copyfile(url, url + '.jpg')
                del_jpgs.append(url + '.jpg')
                return url + '.jpg'
            return url
        else:
            try:
                path = load_image_to_dir(url, orig=orig)
                if path != -1:
                    del_jpgs.append(path)
                return path
            except Exception as e:
                if config.is_debug:
                    print(e)
                logging.error(f"Can't find file with url - {url}")
                return -1


def load_image_to_dir(url, orig=False):
    # загрузка изображения на локальный компьютер
    try:
        img_data = requests.get(url).content
        if orig:
            with open(config.path_load_imgs_from_url + '/' + 'original.jpg', 'wb') as handler:
                handler.write(img_data)
            return str(Path.cwd()) + '/' + config.path_load_imgs_from_url + '/' + 'original.jpg'

        else:
            with open(config.path_load_imgs_from_url+'/'+str(k)+'.jpg', 'wb') as handler:
                handler.write(img_data)

            return str(Path.cwd()) + '/' + config.path_load_imgs_from_url+'/'+str(k)+'.jpg'

    except:
        logging.error(f"Can't download url - {url}")
        return -1


if __name__ == '__main__':
    """
    Загрузка всех основных библиотек, моделей.
    """
    with tf.device('/cpu:0'):
        # Create logs
        logs_file = str(datetime.datetime.now().strftime("%d.%m.%Y")) + '-log.txt'
        logging.basicConfig(filename=logs_file, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
        logging.info("Program was started\n\n")

        siamese_model = tf.keras.models.load_model(config.model_dir)  # модель
        logging.info("Siamese model loaded")

        # Load words
        stop_words = set(stopwords.words('russian'))
        stop_words.update(config.words_to_delete)

        # Создание папки для загрузки изображений с интернета
        if not os.path.exists(config.path_load_imgs_from_url):
            os.mkdir(config.path_load_imgs_from_url)

        # Добавить все обработанные json в переменную
        done_json = []
        if not os.path.exists(config.path_to_scan_json + config.json_done_db):
            open(config.path_to_scan_json + config.json_done_db, 'w')
        else:
            with open(config.path_to_scan_json + config.json_done_db, 'r') as file:
                for item in file:
                    done_json += item.split(';')

        print("\n\nProgram ready to use\n")
        while True:  # Каждый цикл выполняется
            try:
                del_jpgs = []  # Картинки на удаление

                json_path = find_new_json()
                if json_path == -1:  # если не найдено файлов для обработки
                    #print()
                    #print("Wait for json...")
                    time.sleep(config.scan_time)
                else:
                    print(f"Find new json - {json_path.split('/')[-1]}")
                    # Load Json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    logging.info(f"Json was load successfully - {json_path.split('/')[-1]}")

                    # Новый путь до картинок (url)
                    new_path_original = check_url_local(data["original_goods"]['image'], orig=True)
                    new_path_other = []
                    for k in range(len(data["other_goods"])):
                        new_path_other.append(check_url_local(data["other_goods"][k]['image']))

                    # Siamese Model fast
                    sm_grade_fast = CompareSiamese()
                    sm_grade = []
                    for k in range(len(sm_grade_fast)):
                        sm_grade.append(sm_grade_fast[k][0])

                    # Compare with turicreate
                    # ДОДЕЛАТЬ!!!
                    # path добавляется как абсолютный путь, а у меня он как относительный если сохраняются картинки
                    imgs_turi_data = tc.image_analysis.load_images(new_path_original)  # 1-я картинка
                    for i in range(len(new_path_other)):
                        try:
                            imgs_turi_data = imgs_turi_data.append(tc.image_analysis.load_images(new_path_other[i]))
                        except:
                            logging.error(f"Can't compare image tc with ads_id - {data['other_goods'][i]['ads_id']}")
                    imgs_turi_data = imgs_turi_data.add_row_number()  # добавляем id


                    model_turi = tc.image_similarity.create(imgs_turi_data)  # модель
                    # data картинок с рангом и схожестью
                    try:
                        distance_imgs_turi = model_turi.query(imgs_turi_data[imgs_turi_data['path'] == new_path_original],
                                                              k=len(imgs_turi_data))
                    except:
                        logging.error("NO ORIGINAL IMAGE IN FOLDER. OR CAN'T BE FIND!!!")

                    # расстояния
                    # для названия
                    original_name = data["original_goods"]["name"]
                    names_cos_ranges = []  # кос-ое расст-ие

                    # для описания
                    original_desc = data["original_goods"]["description"]
                    desc_cos_ranges = []  # кос-ое расст-ие

                    # для цены
                    original_price = data["original_goods"]["price"]
                    price_similarity = []

                    # для turi create
                    ranks_img = []

                    # для siamese model
                    img_original = read_img_from_dir(new_path_original, expand=True)

                    # Цикл по всем объектам
                    for i in range(len(data["other_goods"])):
                        # Turi create
                        try:
                            ref_label = imgs_turi_data[imgs_turi_data['path'] == new_path_other[i]]
                            r = str(ref_label['id'])
                            r = r.split("[")[1].split(",")[0]

                            rank = str(distance_imgs_turi[distance_imgs_turi['reference_label'] == int(r)]['rank'])
                            rank = int(rank.split("[")[1].split(",")[0])

                            # dist = str(distance_imgs_turi[distance_imgs_turi['reference_label'] == int(r)]['distance'])
                            # dist = float(dist.split("[")[1].split(",")[0])
                        except:
                            rank = -1
                            # dist = -1
                        ranks_img.append(rank)


                        # Siamese Model (slow)
                        if len(sm_grade_fast) == 0:  # если есть ошибка
                            try:
                                img_to_compare = read_img_from_dir(new_path_other[i], expand=True)
                                sm_grade.append(siamese_model.predict([img_original, img_to_compare])[0][0])  # оценка от 0 до 1 схожести
                            except:
                                sm_grade.append(-1)
                                logging.error(f"Can't compare image with ads_id - {data['other_goods'][i]['ads_id']}")

                        # Compare Names and Descriptions
                        name2 = data["other_goods"][i]["name"]
                        desc2 = data["other_goods"][i]["description"]

                        cos_range = find_cos_range_2_text(original_name, name2, config.lang_detect)
                        if cos_range == -1:
                            logging.error("No name with ads_id - {}".format(data["other_goods"][i]["ads_id"]))
                        names_cos_ranges.append(cos_range)

                        cos_range = find_cos_range_2_text(original_desc, desc2, config.lang_detect)
                        if cos_range == -1:
                            logging.error("No description with ads_id - {}".format(data["other_goods"][i]["ads_id"]))
                        desc_cos_ranges.append(cos_range)

                        # Compare price
                        price2 = data["other_goods"][i]["price"]

                        price_sim = compare_price(original_price, price2)
                        if price_sim == -1:
                            logging.error(
                                "Couldn't compare price with ads_id - {}".format(data["other_goods"][i]["ads_id"]))
                        price_similarity.append(price_sim)

                        #Calculate percent identity
                        p = calc_identity(i)
                        if p == -1:
                            logging.error(
                                "COULDN'T COMPARE ITEM WITH ADS_ID - {}".format(data["other_goods"][i]["ads_id"]))
                            p = 0.5

                        data["other_goods"][i]["identity"] = p

            except Exception as e:
                if config.is_debug:
                    print(e)
                print("ERROR WITH THIS JSON!!")
                logging.error(f"Can't start program!!! with json_path - {json_path}")


            if json_path != -1:

                # Save to Json
                with open(config.path_to_save_json+json_path.split('/')[-1], 'w') as file:
                    json.dump(data, file, indent=3)  # ensure_ascii = False (to windows tests)

                # удаление изображений, которые были загружены
                for i in range(len(del_jpgs)):
                    os.remove(del_jpgs[i])
                del_jpgs = []

                # добавление в файл обработанных json

                with open(config.path_to_scan_json + config.json_done_db, 'a') as f:
                    f.write(';' + json_path.split('/')[-1])
                done_json.append(json_path.split('/')[-1])

                logging.info(f"Program was end this JSON - {json_path.split('/')[-1]}\n")

                print("Done file has been saved at {}".format(config.path_to_save_json+json_path.split('/')[-1]))
                print("Logs file has been saved at {}".format(logs_file))
