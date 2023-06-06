import os
import tensorflow as tf
import numpy as np

acu = 0.9  # 0.5
label = ["取消動作", "選擇手勢", "呼叫選單"]
gpus = tf.config.experimental.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)

loaded_model = tf.lite.Interpreter(model_path='models/DNNGestureModel.tflite')
loaded_model.allocate_tensors()


def get_predictions(landmark, hasData=False):
    new_data = []
    if not hasData:
        for l in landmark:
            new_data.append(l.x)
            new_data.append(l.y)
    else:
        new_data = landmark
    # Get input and output tensors
    input_data = np.array([new_data], dtype=np.float32)
    input_details = loaded_model.get_input_details()
    output_details = loaded_model.get_output_details()
    loaded_model.set_tensor(input_details[0]['index'], input_data)
    loaded_model.invoke()
    # results = loaded_model.predict([new_data])[0]
    results = loaded_model.get_tensor(output_details[0]['index'])[0]
    count = 0
    for i in results:
        if count > 4:
            return "no gesture"
        if i > acu:
            return label[count]
        count += 1


label = ["cancel", "choose", "menu", "left", "right", "else"]
new_data = [0 for i in range(42)]

print(get_predictions(new_data, hasData=True))
