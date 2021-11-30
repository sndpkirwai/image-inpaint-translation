import os

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from PIL import ImageFont, ImageDraw, Image
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import streamlit as st
import re
import string


def save_uploadedfile(uploadedfile, path):
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    print("Saved File:{} to upload".format(uploadedfile.name))


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def inpaint_easyocr(img_path):
    img = cv2.imread(img_path)
    img_copy = img.copy()
    plt.imshow(img)
    plt.show()
    result = reader.readtext(img)
    text = [result[i][1] for i in range(len(result))]
    # text=' '.join(map(str,text))

    kernel = np.ones((1, 1), np.uint8)
    # print(img_copy)
    mask = np.zeros(img.shape[:2], dtype="uint8")

    for i in range(len(result)):
        l = []
        for box in result[i]:
            l.append(box)
            x0, y0 = l[0][0]
            x1, y1 = l[0][1]
            x2, y2 = l[0][2]
            x3, y3 = l[0][3]
            x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
            x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
            thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

            rect = cv2.rectangle(img, (int(x0), int(y0)), ((int(x2), int(y2))),
                                 (0, 255, 0), 2)
            img_copy = cv2.inpaint(img_copy, mask, 7, cv2.INPAINT_NS)
            # img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)
            # img_copy=cv2.dilate(img_copy,kernel,iterations = 1)
    return (img_copy, rect, text, result)

def wrap_text(text, font, max_width):

    lines = []

        # If the text width is smaller than the image width, then no need to split
        # just add it to the line list and return
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        # split the line by spaces to get words
        words = text.split(' ')
        i = 0
            # append every word to a line while its width is shorter than the image width
        while i < len(words):
            line = ''
            while (i < len(words) and font.getsize(line + words[0] <=
                                                   max_width)):
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line)
    return lines


def preprocessing(text):
    if "com" in (text[-1]):
        # to remove .com if it is present at the end(Note: it is assumed that websites will be present at the end generally)
        com = text.pop(-1)
        # print(com)
    text2 = ' '.join(text)

    text1 = re.sub(r'[0-9]+', ' ', text2)
    text1 = re.sub(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff \n / > < '' ' ' "" " " }{][ ]',
        '', text1)

    # removing square brackets
    # Deletes particular pattern
    text1 = re.sub('\[.*?\]', '', text1)

    text1 = re.sub('<.*?>+', '', text1)
    # removing hyperlink
    text1 = re.sub('https?://\S+|www\.\S+', '', text1)

    # removing puncuation
    text1 = re.sub('[%s]' % re.escape(string.punctuation), '', text1)

    text1 = re.sub('\n', '', text1)

    # remove words containing numbers
    text1 = re.sub('\w*\d\w*', '', text1).split()

    text1 = " ".join(text1)
    return text1


def translated_img_other_lang(inpaint_img, result, fontpath,
                              translated_text_str, fontsize):
    img2 = inpaint_img.copy()
    b, g, r = 0, 0, 0
    a, b, c = inpaint_img.shape  # a=height,b=width
    gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    moment = cv2.moments(gray_img)
    x = []
    y = []
    for i in range(len(result)):
        x.append(result[i][0][0][0])  # gives starting x cordinate x1
        x.append(result[i][0][2][0])  # gives ending x cordinate x3
        y.append(result[i][0][0][1])  # gives starting y cordinate y1
        y.append(result[i][0][2][1])  # gives ending y cordinate y3

    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    X = min_x + 5
    Y = min_y + 5
    # X = int(moment ["m10"] / moment["m00"])/4.5
    # Y = int(moment ["m01"] / moment["m00"])/4
    # Y=10

    font = ImageFont.truetype(fontpath, fontsize)
    lines = wrap_text(translated_text_str, font, max_x - min_x)
    print(lines)
    img_pil = Image.fromarray(img2)
    for i in lines:
        draw = ImageDraw.Draw(img_pil)
        draw.text((X, Y), i, font=font, fill=(b, g, r))
        # X=X+len(translated_text_list[i])
        Y = Y + ((a / 10) + 10)  # gives space dynamically
    img2 = np.array(img_pil)

    return (img2)


st.title('Image-Inpaint')
st.subheader('Upload any image which needs to be translated')

# if the user chooses to upload the data
file = st.file_uploader('Image file')
# browsing and uploading the dataset (strictly in csv format)
# dataset = pd.DataFrame()


dict_language = {
    "Tamil": "ta",
    "Telugu": "te",
    "Chinese": "ch_sim",
    # "TChinese":"ch_tra",
    "Japanese": "ja",
    "Vietnamese": "vi",
    "Lithuanian": "lt",
    "French": "fr",
    "Thai": "th",
    "Czech": "cs",
    "Bengali": "bn",
    "Spanish": "es",
    "Russian": "ru",
    'Urdu': 'ur',
    'Latvian': 'lv',
    'Latin': 'la',
    'Italian': 'it',
    'Swahili': 'sw',
    'Ukrainian': 'uk',
    'Korean': 'ko',
    'English': 'en',
    'Romanian': 'ro',
    'Nepali': 'ne',
    'Indonesian': 'id',
    'Marathi': 'mr',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'German': 'de',
    'Persian': 'fa',
    'Estonian': 'et',
    'Polish': 'pl',
    'Sweden': 'sv',
    'Norwegian': 'no',
    "Portugese": "pt",
    "Turkish": "tr",
    "Afrikaans": "af",
    "Dutch": "nl",
    "Tagalog": "tl",
    "Slovenian": "sl"

}

target_lang = {
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovene": "sl_SI"
}

if file is not None:
    speech, _ = sf.read(file)

    option = st.selectbox('Select the source language',
                          tuple(dict_language.keys()))

    st.write('You selected:', option)
    model_mBart = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt")
    tokenizer_mBart = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt")
    path_dir = os.path.join(os.getcwd(), 'upload')
    upload_path = os.path.join(path_dir, file.name)
    save_uploadedfile(file, upload_path)

    reader = easyocr.Reader([lang], gpu=False)
    inpainted, rect, text, result = inpaint_easyocr(upload_path)
    print(f"The detected text is: {text}")
    st.image(plt.imshow(inpainted))
    st.image(plt.show())
    st.image(plt.imshow(rect))

    target_option = st.selectbox('Select the Target language',
                                 tuple(target_lang.keys()))

    st.write('You selected target language as :', target_option)

    text_formatted = preprocessing(text)

    tokenizer_mBart.src_lang = option
    encoded_mBart = tokenizer_mBart(text_formatted, return_tensors="pt")
    generated_tokens = model_mBart.generate(**encoded_mBart,
                                            forced_bos_token_id=
                                            tokenizer_mBart.lang_code_to_id[
                                                target_option])
    translated_text = tokenizer_mBart.batch_decode(generated_tokens,
                                                   skip_special_tokens=True)
    translated_text_str = " ".join(translated_text)
    st.write(translated_text_str)

    if tgt == 'en_XX':
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/times-new-roman.ttf"
        trans_img = translated_img_other_lang(inpainted, result, fontpath,
                                              translated_text_str, 15)

    elif tgt == "ta_IN" or "te_IN" or "ml_IN" or "si_LK" or "hi_IN":
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/Akshar Unicode.ttf"
        if (inpainted.shape[0]) <= 400:
            trans_img_lang = translated_img_other_lang(inpainted, result,
                                                       fontpath,
                                                       translated_text_str, 25)

        elif (inpainted.shape[0]) > 1000:
            trans_img_lang = translated_img_other_lang(inpainted, result,
                                                       fontpath,
                                                       translated_text_str,
                                                       100)

        elif (inpainted.shape[0]) > 400 and (inpainted.shape[0]) <= 600:
            trans_img_lang = translated_img_other_lang(inpainted, result,
                                                       fontpath,
                                                       translated_text_str, 40)

    elif tgt == "de_DE":
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/German.ttf"
        trans_img_lang = translated_img_other_lang(inpainted, result, fontpath,
                                                   translated_text_str, 20)


    elif tgt == "es_XX":
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/spanish.ttf"
        trans_img_lang = translated_img_other_lang(inpainted, result, fontpath,
                                                   translated_text_str, 0.02)
        plt.rcParams["figure.figsize"] = (16, 16)



    elif tgt == "fr_XX":
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/French.ttf"
        trans_img_lang = translated_img_other_lang(inpainted, result, fontpath,
                                                   translated_text_str, 20)


    elif tgt == "ja_XX":
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/arial-unicode-ms.ttf"
        trans_img_lang = translated_img_other_lang(inpainted, result, fontpath,
                                                   translated_text_str, 20)

    elif tgt == "ko_KR":
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/arial-unicode-ms.ttf"
        trans_img_lang = translated_img_other_lang(inpainted, result, fontpath,
                                                   translated_text_str, 20)


    elif tgt == "zh_CN":  # chinese
        fontpath = "gdrive/My Drive/TMLC_Project3/lang/arial-unicode-ms.ttf"
        trans_img_lang = translated_img_other_lang(inpainted, result, fontpath,
                                                   translated_text_str, 36)
    st.image(plt.imshow(trans_img_lang))