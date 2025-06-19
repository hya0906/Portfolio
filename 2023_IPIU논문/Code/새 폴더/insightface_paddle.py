import insightface_paddle as face
import cv2 as cv
import logging
from PIL import Image
logging.basicConfig(level=logging.INFO)

parser = face.parser()
help_info = parser.print_help()
print(help_info)
##########
parser = face.parser()
print('parser====',parser)
args = parser.parse_args()
print('args=====',args)
args.build_index = "D:/insightface_folder/lab_test/index.bin"#경로를 만들면 자동적으로 만들어짐
args.img_dir = "C:/Users/IPCG/Desktop/Cropped_images/train"
args.label = "D:/insightface_folder/lab_test/label.txt"
predictor = face.InsightFace(args)
predictor.build_index()
'''
parser = face.parser()
args = parser.parse_args()

args.det = True
args.rec = True
args.index = "./demo/friends/index.bin"
args.output = "./output"
input_path = "./demo/friends/query/friends1.jpg"

img = cv.imread(input_path)
predictor = face.InsightFace(args)
# res = predictor.predict(input_path, print_info=True)
# next(res)
# #cv.imshow('res', face.InsightFace.draw(input_path))
#
'''

# box, features = predictor.predict_np_img(img)
# predictor.init_rec(args)
# labels = predictor.rec_predictor.retrieval(features)
# a = predictor.draw(img,box, labels)
# pil_image=Image.fromarray(a)
# pil_image.show()