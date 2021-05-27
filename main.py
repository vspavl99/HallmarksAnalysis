import time
from dataclasses import dataclass
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from torchvision import models


@dataclass
class Configuration:
    cities: tuple = ('Sheffield', 'London', 'Birmingham', 'Glasgow', 'Chester', 'Dublin', 'Edinburgh')
    letters: tuple = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')

    hallmark_image_shape: tuple = (265, 95)
    letter_image_shape: tuple = (70, 70)

    path_to_letter_model_weights: str = 'configuration/letter_classification.pth'
    path_to_city_model_weights: str = 'configuration/city_classification.pth'

    path_to_detection_model_weights: str = 'configuration/yolov4.weights'
    path_to_detection_config_file: str = 'configuration/yolov4-obj.cfg'

    # Name of the classes for detection
    class_names = ['Hallmark', 'Letter']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def init_classification_model(config: dataclass, path_to_weights: str) -> models.ResNet:
    """
    The function return object of model with uploaded weights for certain task of classification


    :param config: dataclass object with all necessary information
    :param path_to_weights: path to .pth file with weights of model
    :return:
    """

    device = config.device
    model_state = torch.load(path_to_weights, map_location=torch.device(device))
    output_number_of_classes, input_number_of_features = model_state['fc.weight'].shape

    # Model architecture initialization
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(input_number_of_features, output_number_of_classes)

    # Loading state of the model
    model.load_state_dict(model_state)

    # Transferring the model to the device and freezing the weights
    model.to(device)
    model.eval()

    return model


def init_detection_model(config: dataclass) -> cv2.dnn_DetectionModel:
    """
    The function return object of model with uploaded weights for detection task
    :param config: dataclass object with all necessary information
    :return:
    """
    path_to_weights = config.path_to_detection_model_weights
    path_to_config = config.path_to_detection_config_file

    net = cv2.dnn.readNet(path_to_weights, path_to_config)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    return model


class HallmarkAnalyser:
    def __init__(self, config: dataclass):
        """

        :param config: class with all configuration parameters
        """
        self.config = config
        self.device = config.device

        # Initialization of all necessary models
        self.letter_model = init_classification_model(config, config.path_to_letter_model_weights)
        self.city_model = init_classification_model(config, config.path_to_city_model_weights)
        self.detection_model = init_detection_model(config)

        # Detection parameters
        self.detection_threshold = 0.2
        self.nms_threshold = 0.4
        self.colors = ((0, 255, 255), (255, 0, 255))

    def make_prediction(self, image: np.ndarray, type_of_task: str) -> tuple:
        """
        Function provide classification of incoming image
        :param image: Cropped image with hallmark or letter
        :param type_of_task: type of classification task ('hallmark' or 'letter')
        :return: tuple with predicted label and confidence of prediction
        """

        if type_of_task == 'Letter':
            model = self.letter_model
            image_shape = self.config.letter_image_shape
            labels = self.config.letters
        else:
            model = self.city_model
            image_shape = self.config.hallmark_image_shape
            labels = self.config.cities

        image = cv2.resize(image, image_shape)
        image = torch.tensor(image / 255).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        start = time.time()
        prediction = model(image).squeeze().detach().cpu()
        end = time.time()

        print(f'Classification {type_of_task} inference time : {(end - start):.2f} seconds')

        predicted_class = prediction.argmax()
        predicted_label = labels[predicted_class]
        confidence = prediction.sigmoid()[predicted_class].item()

        return predicted_label, confidence

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Function process image of silverware. (detects the hallmark and classify the town and letter)
        :param image: image for hallmark analysis
        :return: dictionary with keys Detection classes and values tuple of classification classes
        """
        image = cv2.resize(image, (416, 416))

        start = time.time()
        classes, scores, boxes = self.detection_model.detect(image, self.detection_threshold, self.nms_threshold)
        end = time.time()

        print(f'Detection inference time : {(end - start):.2f} seconds')

        analysis_results = {}
        for (class_id, score, box) in zip(classes, scores, boxes):
            color = self.colors[int(class_id)]

            x_left, y_top, x_right, y_bottom = box[0], box[1], box[0] + box[2], box[1] + box[3]
            image_hallmark = image[y_top:y_bottom, x_left:x_right]

            classification_label, confidence = self.make_prediction(image_hallmark,
                                                                    self.config.class_names[class_id[0]])
            analysis_results[self.config.class_names[class_id[0]]] = (classification_label, confidence)

            label = f"{self.config.class_names[class_id[0]]}: {float(score):.2f}," \
                    f"{classification_label}: {confidence:.2f}"

            text_position = (box[0], box[1] - 10) if self.config.class_names[class_id[0]] == 'Hallmark' \
                else (box[0], box[1] + box[2] + 30)

            cv2.rectangle(image, box, color, 1)
            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image, analysis_results


if __name__ == '__main__':

    cfg = Configuration()
    analyser = HallmarkAnalyser(cfg)

    image_input = cv2.imread('Data/DatasetOriginal/images/5_original.jpg')
    image_output, results = analyser.process_image(image_input)

    print(results)

    url = "https://silvermakersmarks.co.uk/Dates/{}/Date%20Letters%20{}.html".format(
        results['Hallmark'][0], results['Letter'][0]
    )

    print(url)
    plt.imshow(image_output)
    plt.show()
