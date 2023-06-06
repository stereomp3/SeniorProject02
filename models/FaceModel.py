import numpy as np
import os
import tensorflow as tf

# Memory leak when repeatedly loading and deleting keras models: https://github.com/tensorflow/tensorflow/issues/40171
gpus = tf.config.experimental.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)

loaded_model = tf.lite.Interpreter(model_path='models/DNNFaceModel.tflite')
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
    return results


new_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


get_predictions(new_data, hasData=True)