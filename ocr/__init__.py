from textwrap import wrap
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from pathlib import Path

from word_segmentation import checkout

import os
from os import listdir, path
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from pathlib import Path
from collections import Counter

import pickle
import nltk
from pathlib import Path
from collections import Counter
from PIL import Image

from fpdf import FPDF

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
# os.system('pip install -U git+https://github.com/albumentations-team/albumentations')

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

from keras.preprocessing.image import ImageDataGenerator
import albumentations as albu
from tensorflow.keras.applications import *

tf.__version__

# import urllib.request

# # from pythonRLSA import rlsa
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np

# import os
# from os import listdir, path


# from sklearn.model_selection import train_test_split

# import tensorflow as tf
# # tf.enable_eager_execution()
# from tensorflow import keras
# from tensorflow.keras import layers
# from keras.utils.vis_utils import plot_model

# from keras.preprocessing.image import ImageDataGenerator
# import albumentations as albu
# from tensorflow.keras.applications import *

# tf.__version__

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Resize the files into 200x50 and save them in the same directory

data_dir = 'ocr/static/words/'


def resize():
    dirs = os.listdir(data_dir)
#     print(dirs)
    #cnt = 0
    for item in dirs:
        if os.path.isfile(data_dir+item):
            #cnt = cnt+1
            im = Image.open(data_dir+item)
            f, e = os.path.splitext(data_dir+item)
            imResize = im.resize((200, 50), Image.ANTIALIAS)
            imResize = imResize.convert('RGB')
            imResize.save(f + '.jpg', 'JPEG', quality=100)

# --------------- Model ----------------


img_height = 50
img_width = 200

ldir = 'ocr/model'

model = tf.keras.models.load_model(ldir)
data_order = ["Train", "Validation", "Test"]
model.load_weights(os.path.join(ldir, 'best_weight.h5'))

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(
        name="dense2").output
)

mapping = {
    "KA": "क",
    "KH": "ख",
    "GA": "ग",
    "GH": "घ",
    "NG": "ङ",
    "CH": "च",
    "AL": "छ",
    "JA": "ज",
    "JH": "झ",
    "NY": "ञ",
    "TA": "ट",
    "TH": "ठ",
    "DA": "ड",
    "DH": "ढ",
    "AN": "ण",
    "AT": "त",
    "HT": "थ",
    "AD": "द",
    "HD": "ध",
    "NA": "न",
    "PA": "प",
    "FA": "फ",
    "BA": "ब",
    "BH": "भ",
    "MA": "म",
    "YA": "य",
    "RA": "र",
    "LA": "ल",
    "VA": "व",
    "SH": "श",
    "HS": "ष",
    "SA": "स",
    "HP": "ह",
    "KS": "क्ष",
    "TR": "त्र",
    "JY": "ज्ञ",
    "AA": "अ",
    "AK": "आ",
    "IE": "इ",
    "EE": "ई",
    "UU": "उ",
    "OO": "ऊ",
    "AE": "ए",
    "AI": "ऐ",
    "AO": "ओ",
    "AU": "औ",
    "RE": "ऋ",
    "GP": "अं",
    "AH": "अ:",
    "CO": "ो",
    "BO": "ौ",
    "AP": "ा",
    "CA": "े",
    "BB": "ै",
    "IZ": "ं",
    "NZ": "ँ",
    "UP": "ु",
    "UQ": "ू",
    "LZ": "्र",
    "GZ": "ृ",
    "EK": "ि",
    "EL": "ी",
    "HZ": "र्",
    "MZ": "ॅ",
    "JZ": "़",
    "PP": "ः",
    "KF": "क्",
    "EF": "ख्",
    "GF": "ग्",
    "IF": "घ्",
    "CF": "च्",
    "AF": "छ्",
    "JF": "ज्",
    "NF": "झ्",
    "TF": "ट्",
    "OF": "ठ्",
    "DF": "ड्",
    "QF": "ढ्",
    "UF": "त्",
    "HF": "थ्",
    "WF": "द्",
    "XF": "ध्",
    "KZ": "न्",
    "PF": "प्",
    "FF": "फ्",
    "BF": "ब्",
    "ZF": "भ्",
    "MF": "म्",
    "YF": "य्",
    "RF": "र्",
    "LF": "ल्",
    "VF": "व्",
    "SF": "श्",
    "AZ": "ष्",
    "BZ": "स्",
    "CZ": "ह्",
    "FZ": "ज्ञ्",
    "DZ": "क्ष्",
    "EZ": "त्र्"
}

# Mapping keys to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(mapping.keys()), mask_token=None
)

# Mapping integers back to original keys
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
        #self.acc = tf.keras.metrics.Accuracy()

    # def on_epoch_end(self):
    #    self.acc.reset_states()

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * \
            tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * \
            tf.ones(shape=(batch_len, 1), dtype="int64")

        # Adding loss
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

# A utility function to decode the output of the network


max_length = 20

# A utility function to decode the output of the network


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         print(res)
        res = wrap(res, 2)
        label = ""
        carry = ""
        for ch in res:
            try:
                cur = mapping[ch]
                if(ch == 'HZ' or ch == 'EK'):
                    carry = cur
                else:
                    label += cur
                    if(carry):
                        label += carry
                        carry = ""
            except:
                break
        output_text.append(label)
        # print(label)
    return output_text


# def decode_batch_predictions2(pred):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     # Use greedy search. For complex tasks, you can use beam search
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
#         :, :max_length
#     ]

#     return results
#     # Iterate over the results and get back the text
#     output_text = []
#     for res in results:
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
# #         print(res)
#         res = wrap(res, 2)
#         label = ""
#         carry = ""
#         for ch in res:
#             try:
#                 cur = mapping[ch]
#                 if(ch == 'LZ' or ch == 'HZ' or ch == 'EK'):
#                     carry = cur
#                 else:
#                     label += cur
#                     if(carry):
#                         label += carry
#                         carry = ""
#             except:
#                 break
#         output_text.append(label)
#         print(label)
#     return output_text


class DataGenTest(tf.keras.utils.Sequence):
    """
    X_dirs:         Contains image directories.
    y_labels:       Contains truth-label (text) of each image.
    batch_size:     The batch_size of data flow.
    augment:        A boolean param. If set to True, it would return augmented images.
    output_shape:   The output image shape.
    """

    def __init__(self, X_dirs, batch_size=1,
                 output_shape=(50, 200, 3), p=0.6):

        self.img_height = output_shape[0]
        self.img_width = output_shape[1]
        self.channels = output_shape[2]
        self.batch_size = batch_size
        self.indexes = np.arange(16)

        self.x_dirs = X_dirs

        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of data-transfers (batches) per epoch. """
        return int(self.x_dirs.shape[0] // self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        imgs = tf.map_fn(lambda id: tf.transpose(tf.image.resize(tf.io.decode_png(tf.io.read_file(self.x_dirs[id]),
                                                                                  channels=1),
                                                                 [self.img_height, self.img_width]), perm=[1, 0, 2]),
                         indexes, dtype=tf.float32)

        imgs = keras.backend.repeat_elements(
            x=imgs, rep=self.channels, axis=-1)
        imgs = (imgs.numpy()).astype(np.uint8)

        return tf.keras.applications.imagenet_utils.preprocess_input(imgs)

    def read_file(self, fileID):
        # 1. Read image
        img = tf.io.read_file(self.x_dirs[fileID])
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        return img

        return imgs


def get_data():
    # Path to the data directory
    data_dir_path = Path(data_dir)

    # Get list of all the images
    images = sorted(list(map(str, list(data_dir_path.glob("*.jpg")))))

    return np.array(images)


# def rlsa(image):

#     from pythonRLSA import rlsa

#     image_rlsa = image.copy()
#     image_rlsa_1 = rlsa.rlsa(image_rlsa, True, False, 70)

# #     cv2.imshow('test',image_rlsa_1)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

#     image_rlsa_1 = ~image_rlsa_1
#     contours = cv2.findContours(
#         image_rlsa_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#     num = 0

#     for f in os.listdir('ocr/static/img'):
#         os.remove(os.path.join('ocr', 'static', 'img', f))

#     for cnt in contours:
#         (x, y, w, h) = cv2.boundingRect(cnt)
#         ROI = image[y:y+h, x:x+w]
#         cv2.imwrite(f'ocr/static/img/{num}.jpg', ROI)
#         num = num+1


# def preprocess(img):
#     binary = binarization(img)
#     rlsa(binary)

# --------------- Model ----------------


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():

    for f in os.listdir('ocr/static/uploads'):
        os.remove(os.path.join('ocr', 'static', 'uploads', f))

    for f in os.listdir('ocr/static/lines'):
        os.remove(os.path.join('ocr', 'static', 'lines', f))

    for f in os.listdir('ocr/static/words'):
        os.remove(os.path.join('ocr', 'static', 'words', f))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # print(os.path.join('ocr', app.config['UPLOAD_FOLDER'], filename))

        filepath = os.path.join('ocr', app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        img = cv2.imread(filepath)

        print(filename)

        # main.main()
        checkout.word_segmentor(f'ocr/static/uploads/{filename}')

        print('line segmented')

        resize()

        data = get_data()

        # print(data)

        test_image_data = DataGenTest(data)

        label = ""

        for batch in test_image_data:
            preds = prediction_model.predict(batch)
            pred_texts = decode_batch_predictions(preds)
            print(pred_texts)

            label = label + pred_texts[0] + ' '

        # preprocess(img)

        with open('ocr/static/text/words.txt', 'w') as f:
            f.write(str(label))

            f.close()

        # for f in os.listdir('ocr/static/img'):
        #     img = encode_single_sample(os.path.join('ocr', 'static', 'img', f))
        #     img = tf.expand_dims(
        #         img, axis=0, name=None
        #     )
        #     preds = prediction_model.predict(img)

        #     label = decode_batch_predictions(preds)
        #     # print(label)
        return render_template('index.html', filename=filename, output=label)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/change_label", methods=['POST'])
def change_label():
    # Return the text you want the label to be
    global result
    # print(result)
    return "test"


# @app.route("/change_label", methods=['POST'])
# def download_pdf():
#     # Return the text you want the label to be
#     global result
#     path = "static/pdf/words.pdf"
#     return send_file(path, as_attachment=True)
