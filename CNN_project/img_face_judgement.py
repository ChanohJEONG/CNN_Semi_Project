import sys
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import settings

def detect_face(model, cascade_filepath, image):
    # 이미지를 BGR형식에서 RGB형식으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plt.imshow(image)
    # plt.show()
    # print(image.shape)

    # 그레이스케일 이미지로 변환
    image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 얼굴인식 실행
    cascade = cv2.CascadeClassifier(cascade_filepath)
    # 얼굴인식
    faces = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=2,minSize=(64,64))
    # 얼굴이 1개 이상 검출된 경우
    if len(faces) > 0:
        print(f"인식된 얼굴의 수 : {len(faces)}")
        for (x_pos, y_pos, width, height) in faces: # x_pos, y_pos, width, height
            face_image = image[y_pos: y_pos+height, x_pos:x_pos+width]
            print(f"인식한 얼굴의 사이즈 : {face_image.shape}")
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                print("인식한 얼굴의 사이즈가 너무 작습니다.")
                continue
            # 인식한 얼굴 사이즈 축소
            face_image = cv2.resize(face_image,(64,64))
            # 인식한 얼굴 주변에 붉은색 사각형을 표시
            cv2.rectangle(image, (x_pos, y_pos),(x_pos+width, y_pos+height), (255,0,0), thickness=2)
             # image, (사각형 시작 좌표), (사각형 종료 좌표), (색상), (선 굵기)thickness=

            # 인식한 얼굴을 1장의 사진으로 합치고 --> 배열로 변환
            face_image = np.expand_dims(face_image, axis=0)
            # 인식한 얼굴에 이름을 표기
            name = detect_who(model, face_image)
            cv2.putText(image, name, (x_pos,y_pos+height+20), cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255),2)


    # 얼굴이 검출되지 않은 경우
    else:
        print(" 이미지 파일에서 얼굴을 인식할 수 없습니다.")
    # TO-DO

    return image

def detect_who(model, face_image):
    # 예측
    name = ""
    result = model.predict(face_image)
    print(f"송혜교일 가능성 : {result[0][0]*100: .3f}%")
    print(f"전지현일 가능성 : {result[0][1]*100: .3f}%")
    name_number_label = np.argmax(result)
    if name_number_label == 0:
        name = "Heygo"
    elif name_number_label == 1:
        name = "Jihyun"
    return name

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Inpou Model Directory
INPUT_MODEL_PATH = "./model/model.h5"

def main():
    print("===================================================================")
    print("Keras를 이용한 얼굴인식")
    print("학습 모델과 지정한 이미지 파일을 기본으로 연예인 구분하기")
    print("===================================================================")

   # Arguments(Parameter) 인수 체크
    argvs = sys.argv
    if len(argvs) != 2 or not os.path.exists(argvs[1]):
        print("[ERROR] Select image!!")
        return RETURN_FAILURE
    image_file_path = argvs[1]

    # 이미지 파일 읽기
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"[ERROR] Image file read error!! {image_file_path}")
        return RETURN_FAILURE

    # 모델 파일 읽기
    if not os.path.exists(INPUT_MODEL_PATH):
        print("[ERROR] Model not found!!")
        return RETURN_FAILURE

    # 얼굴인식
    model = keras.models.load_model(INPUT_MODEL_PATH)
    cascade_filepath = settings.CASCADE_FILE_PATH
    result_image = detect_face(model, cascade_filepath, image)
    plt.imshow(result_image)
    plt.show()
    # TO-DO

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()