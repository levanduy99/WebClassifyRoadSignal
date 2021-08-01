from __future__ import division, print_function
import os
import numpy as np
# import tensorflow as tf
# tf.config.experimental.list_physical_devices('GPU')
# Keras
# from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'models/trainRoadSignals.h5'
# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

classNames = {0: 'Tốc độ tối đa cho phép (20km/h)',
 1: 'Tốc độ tối đa cho phép (30km/h)',
 2: 'Tốc độ tối đa cho phép (50km/h)',
 3: 'Tốc độ tối đa cho phép (60km/h)',
 4: 'Tốc độ tối đa cho phép (70km/h)',
 5: 'Tốc độ tối đa cho phép (80km/h)',
 6: 'Hạn chế tốc độ tối đa (80km/h)',
 7: 'Tốc độ tối đa cho phép (100km/h)',
 8: 'Tốc độ tối đa cho phép (120km/h)',
 9: 'Cấm vượt',
 10: 'Cấm ôtô tải vượt',
 11: 'Giao nhau với đường không ưu tiên',
 12: 'Bắt đầu đường ưu tiên',
 13: 'Nhường đường',
 14: 'Dừng lại',
 15: 'Đường cấm',
 16: 'Cấm xe ôtô tải',
 17: 'Cấm đi ngược chiều',
 18: 'Nguy hiểm khác',
 19: 'Chỗ ngoặt nguy hiểm vòng bên trái',
 20: 'Chỗ ngoặt nguy hiểm vòng bên phải',
 21: 'Nhiều chỗ ngoặt nguy hiểm liên tiếp',
 22: 'Đường có ổ gà, lồi lõm',
 23: 'Đường trơn',
 24: 'Đường bị thu hẹp về phía phải',
 25: 'Công trường',
 26: ' Giao nhau có tín hiệu đèn',
 27: 'Đường người đi bộ cắt ngang',
 28: 'Trẻ em',
 29: 'Đường người đi xe đạp cắt ngang',
 30: 'Cẩn thận với băng / tuyết',
 31: 'Động vật hoang dã băng qua',
 32: 'Hết tất cả các lệnh cấm',
 33: 'Các xe chỉ được rẽ phải',
 34: 'Các xe chỉ được rẽ trái',
 35: 'Các xe chỉ được đi thẳng',
 36: 'Đi thẳng hoặc sang phải',
 37: 'Đi thẳng hoặc sang trái',
 38: 'Hướng phải đi vòng chướng ngại vật',
 39: 'Hướng trái đi vòng chướng ngại vật',
 40: 'Đi vòng bắt buộc',
 41: 'Hết cấm vượt',
 42: 'Cấm các phương tiện trên 3,5 tấn đi qua'}

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32, 3))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255.0)
    x = np.expand_dims(x, axis=(0))
    #To predict the image
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for client
        pred_class = np.argmax(preds)            # Simple argmax
        if preds[0][pred_class] > 0.7:
            result = classNames[pred_class]      # Convert to string
            return result
    return "No Answer"

if __name__ == '__main__':
    app.run(host='169.254.166.27')
