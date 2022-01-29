from tflite_runtime.interpreter import Interpreter
import PIL.Image as Image
import numpy as np


interpreter = Interpreter(model_path='../models/mvnet_converted.tflite')

print("== Input details ==")
print("name:", interpreter.get_input_details()[0]['name'])
print("shape:", interpreter.get_input_details()[0]['shape'])
print("type:", interpreter.get_input_details()[0]['dtype'])

print("\n== Output details ==")
print("name:", interpreter.get_output_details()[0]['name'])
print("shape:", interpreter.get_output_details()[0]['shape'])
print("type:", interpreter.get_output_details()[0]['dtype'])

print("\nDUMP INPUT")
print(interpreter.get_input_details()[0])
print("\nDUMP OUTPUT")
print(interpreter.get_output_details()[0])


def preprocess_inference_image(img: str):
    IMAGE_SHAPE = (224, 224)
    img = Image.open(img).resize(IMAGE_SHAPE)
    img = np.array(img)/255.0
    #img = preprocess_inference_image('../sample_image.png')
    print('Old shape', img.shape)
    img_expanded = img[np.newaxis, ...]
    print('New shape', img_expanded.shape)
    return img_expanded

example_img_for_tflite = preprocess_inference_image('../image02.png')

interpreter.allocate_tensors()
print("Input data shape:", example_img_for_tflite.shape)
example_img_for_tflite = np.array(example_img_for_tflite, dtype=np.float32)
print("Input data type:", example_img_for_tflite.dtype)

input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], example_img_for_tflite)

interpreter.invoke()

labels_path = '../ImageNetLabels.txt'
imagenet_labels = np.array(open(labels_path).read().splitlines())

output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("\n\nPrediction results:", output_data)
print("Predicted value:", np.argmax(output_data))
print("Predicted label:", imagenet_labels[np.argmax(output_data)])