import cv2
import numpy as np
from pyzbar.pyzbar import decode
import sys
sys.path.append("./doctr/")
#from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os, glob

class QROCRProcessor:
    def __init__(self, input_image_path):
        self.input_image_path = input_image_path
        self.model = ocr_predictor(pretrained=True)
        self.points=[]
    def detect_qr_codes(self):
        frame = cv2.imread(self.input_image_path)
        qr_codes = decode(frame)
        qr_data_list = []
        points = []
        for qr_code in qr_codes:
            qr_data = qr_code.data.decode('utf-8')
            qr_data_list.append(qr_data)
            qr_points = qr_code.polygon
            points.append(qr_points)
            points = np.array(qr_code.polygon, dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        return frame, qr_data_list

    def overlay_text_and_save(self):
        qr_image, qr_data_list = self.detect_qr_codes()
        for qr_points in self.points:
            xmi = min(qr_points[:, 0])
            xma = max(qr_points[:, 0])
            ymi = min(qr_points[:, 1])
            yma = max(qr_points[:, 1])
            cv2.rectangle(qr_image, (xmi, yma), (xma, yma + 20), (255, 255, 255), -1)
            cv2.putText(qr_image, qr_data, (xmi, yma + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        img_doc = DocumentFile.from_images(self.input_image_path)
        result = self.model(img_doc)
        json_output = result.export()

        for i in range(len(json_output['pages'])):
            output_list = json_output['pages'][i]['blocks']
            for j in output_list:
                lines = j['lines']
                for k in lines:
                    words = k['words']
                    for l in words:
                        value = l['value']
                        geometry = l['geometry']
                        xmi = int(geometry[0][0] * json_output['pages'][i]['dimensions'][1])
                        xma = int(geometry[1][0] * json_output['pages'][i]['dimensions'][1])
                        ymi = int(geometry[0][1] * json_output['pages'][i]['dimensions'][0])
                        yma = int(geometry[1][1] * json_output['pages'][i]['dimensions'][0])

                        # Draw bounding boxes for OCR text
                        cv2.rectangle(qr_image, (xmi, ymi), (xma, yma), (0, 255, 0), 2)
                        cv2.putText(qr_image, value, (xmi, ymi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(os.getcwd()+"/output_image.png", qr_image)

if __name__ == "__main__":
    folder_path = "./input-folder" # Make changes for input folder path here
    all_files = os.listdir(folder_path)
    image_files = [file for file in all_files if file.endswith(('.png', '.jpg'))]
    for i in image_files:
	input_image_path = i
    	processor = QROCRProcessor(input_image_path)
    	processor.overlay_text_and_save()
