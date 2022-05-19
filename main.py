from PIL import Image
from urllib.request import urlopen
from flask import Flask, render_template, request, jsonify
from src.classification_of_mammography import classify_right_left_mlo_cc
from src.modeling.run_model_single import (
    load_model, load_inputs, process_augment_inputs, batch_to_tensor
)
from src.optimal_centers.get_optimal_center_single import get_optimal_center_single
from src.cropping.crop_single import crop_single_mammogram
import numpy as np

# Initializations for the model
shared_parameters = {
"device_type": "gpu",
"gpu_number": 0,
"max_crop_noise": (100, 100),
"max_crop_size_noise": 100,
"batch_size": 1,
"seed": 0,
"augmentation": True,
"use_hdf5": True,
}
random_number_generator = np.random.RandomState(shared_parameters["seed"])
image_only_parameters = shared_parameters.copy()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/mammo/image-view/', methods=['POST', 'GET'])
def classifyImageView():
    if request.method == 'GET':
        return render_template('index.html')
    data = request.get_json()
    imageURL = data['imageURL']
    print(imageURL)
    image = np.asarray(Image.open(urlopen(imageURL)))
    mn = image.min()
    mx = image.max()
    mx -= mn
    image = ((image - mn)/mx) * 255
    image = image.astype(np.uint8)
    final_result = classify_right_left_mlo_cc(image)
    print(final_result)
    
    return jsonify({'view': final_result})


@app.route('/mammo/predict/', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    data = request.get_json()
    imageURL = data['imageURL']
    viewData = data['view']
    # imageURL = request.args.get('imageURL')
    # token = request.args.get('token')
    # viewData = request.args.get('view')
    # Initializations for the model
    image_only_parameters["view"] = viewData[0]+'-'+viewData[2:]
    image_only_parameters["use_heatmaps"] = False
    image_only_parameters["model_path"] = "models/ImageOnly__ModeImage_weights.p"
    model, device = load_model(image_only_parameters)
    # File Paths
    cropped_img_path = 'sample_single_output/' + 'out_file.png'
    metadata_path = 'sample_single_output/' + 'metadata' + '.pkl'
    # Preprocessing
    crop_single_mammogram(imageURL, "NO", viewData, cropped_img_path, metadata_path, 100, 50)
    get_optimal_center_single(cropped_img_path, metadata_path)
    # Load Inputs
    model_input = load_inputs(
    image_path=imageURL,
    metadata_path=metadata_path,
    use_heatmaps=False,
    )
    batch = [
    process_augment_inputs(
        model_input=model_input,
        random_number_generator=random_number_generator,
        parameters=image_only_parameters,
    ),
    ]
    # Classification
    tensor_batch = batch_to_tensor(batch, device)
    y_hat = model(tensor_batch)
    predictions = np.exp(y_hat.cpu().detach().numpy())[:, :2, 1]
    predictions_dict = {
        "benign": float(predictions[0][0]),
        "malignant": float(predictions[0][1]),
    }

    predBen = round(predictions_dict['benign'], 3) * 100
    predMal = round(predictions_dict['malignant'], 3) * 100
    
    result = {
        'benign': predBen,
        'malignant': predMal
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)