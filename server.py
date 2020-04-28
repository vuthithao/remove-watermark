from flask import Flask, jsonify
from flask import request
import json
from gevent.pywsgi import WSGIServer
import time
from io import BytesIO
import base64
import cv2
import numpy as np
try:
    from PIL import Image
except:
    from pil import Image

app = Flask(__name__)

def readb64(base64_string, rgb=True):
    """
    Đọc ảnh từ dạng base64 -> numpy array\n

    Input
    -------
    **base64_string**: Ảnh ở dạng base64\n
    **rgb**: True nếu ảnh là dạng RBG (đủ 3 channel), False nếu là ảnh xám (1 channel)

    Output:
    -------
    Ảnh dạng numpy array
    """
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    if rgb:
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2GRAY)

def image_arr_to_b64(arr):
    """
    Chuyển ảnh dạng numpy array sang base64 \n
    Dùng cho mục đích debug/test

    Input
    -------
    **arr**: Ảnh ở dạng numpy array (sau khi đọc bằng openCV)

    Output:
    -------
    Dạng base64 của ảnh
    """
    try:
        buff = BytesIO()
        pil_img = Image.fromarray(arr)
        pil_img.save(buff, format="JPEG")
        new_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        return new_b64
    finally:
        buff.close()


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def watermask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dis = np.amax(gray) - np.amin(gray)
    color = img.mean(axis=0).mean(axis=0)
    alpha = 2.0
    beta = -10
    if sum(color/3) < 250 and dis > 190:
        new = alpha * img + beta
        new = np.clip(new, 0, 255).astype(np.uint8)
    else:
        new = img
    return new


def watermask_(data_type):
    if request.method == "POST":
        dataDict = json.loads(request.data.decode('utf-8'))
        input = dataDict.get(data_type, None)
    img = readb64(input, rgb=True)
    start = time.time()

    new_img = watermask(img)
    new_img = bgr_to_rgb(new_img)
    result = image_arr_to_b64(new_img)
    img = readb64(result, rgb=True)
    cv2.imwrite('tmp.jpg', img)

    end = time.time() - start
    response = jsonify({"result": result, "time": end, "status_code": 200})
    response.status_code = 200
    response.status = 'OK'
    return response, 200


@app.route('/watermask', methods=['POST'])
def watermask__():
    return watermask_(data_type="img")

if __name__ == "__main__":
    http_server = WSGIServer(('', 4000), app)
    http_server.serve_forever()

