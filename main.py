import sys

from src.yolo_detector import YoloDetector
from src.faster_rcnn_detector import FasterRCNNDetector
from src.draw_utils import Draw

video_name = "crowd.mp4"

if __name__ == "__main__":

    while True:
        print("Выберете модель на клавиатуре")
        print("Выберете модель. \n 1 - Yolo, \n 2 - Faster R-CNN \n\n\n 42 - Для завершения программы ")
        target_model = input()
        match target_model:
            case "1":
                print("Вы выбрали Yolo")
                detector = YoloDetector()
                break
            case "2":
                detector = FasterRCNNDetector()
                print("Вы выбрали Faster R-CNN")
                break
            case "42":
                print("Хорошего дня")
                sys.exit()
            case _:
                print("Некорректный выбор. Попробуйте еще раз. ")

    detector.load_video(video_name)
    drawer = Draw()
    detector.run(drawer)
    print("Работа выполнена успешно. Для Выхода из программы нажмите клавишу Enter.")
    input()
