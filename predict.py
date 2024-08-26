import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import tensorflow_hub as hub



def load_model(model_path):
   
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

def process_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict(model, image_path, top_k):
    image = process_image(image_path)
    preds = model.predict(image)[0]
    top_k_indices = preds.argsort()[-top_k:][::-1]
    return top_k_indices, preds[top_k_indices]

def main():
    parser = argparse.ArgumentParser(description="Predict flower class from an image.")
    parser.add_argument('image_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str)
    args = parser.parse_args()

    

    model = load_model(args.model_path)

    top_k_indices, top_k_probs = predict(model, args.image_path, args.top_k)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    top_k_flowers = [class_names.get(str(i), "") for i in top_k_indices]
    top_k_flowers = [flower for flower in top_k_flowers if flower]

    for name_of_flower, probability in zip(top_k_flowers, top_k_probs):
        print(f"{name_of_flower}: {probability:.4f}")

if __name__ == "__main__":
    main()
