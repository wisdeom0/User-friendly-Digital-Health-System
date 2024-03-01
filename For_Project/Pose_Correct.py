import time
import csv
import cv2
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils     # Mediapipe의 시각화 기능을 사용하기 위한 모듈
mp_pose = mp.solutions.pose                 # Mediapipe의 포즈 모델을 사용하기 위한 모듈
csvp = r"/home/jeon/maker/csv"              # CSV 파일 경로
data_dir = r"/home/jeon/maker/json"         # json 디렉토리
output_dir = r"/home/jeon/maker/state"      # numpy 저장 디렉토리
exe_gui = '랫풀 다운'                        # 운동 선택

# '랫풀 다운'에 사용되는 keypoint
data_keypoint_names_2 = ['Nose', 'Left Shoulder', 'Right Shoulder',
                         'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist']  # json 파일 keypoint
mp_keypoint_names = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                     'LEFT_WRIST', 'RIGHT_WRIST']                                   # mediapipe kyepoint
data_keypoint_names = {'NOSE': 'Nose', 'LEFT_SHOULDER': 'Left Shoulder', 'RIGHT_SHOULDER': 'Right Shoulder',
                       'LEFT_ELBOW': 'Left Elbow', 'RIGHT_ELBOW': 'Right Elbow',
                       'LEFT_WRIST': 'Left Wrist', 'RIGHT_WRIST': 'Right Wrist'}




# 운동 종류에 해당하는 파일범위 반환
def load_exe(exe_type):
    exercises = {
        '스탠딩 사이드 크런치': (1, 33),
        '스탠딩 니업': (33, 49),
        '버피 테스트': (49, 81),
        '스텝 포워드 다이나믹 런지': (81, 113),
        '스텝 백워드 다이나믹 런지': (113, 145),
        '사이드 런지': (145, 177),
        '크로스 런지': (177, 185),
        '굿모닝': (185, 193),
        '프런트 레이즈': (193, 201),
        '업라이트로우': (201, 209),
        '바벨 스티프 데드리프트': (209, 217),
        '바벨 로우': (217, 249),
        '덤벨 벤트오버 로우': (249, 281),
        '바벨 데드리프트': (281, 313),
        '바벨 스쿼트': (313, 329),
        '바벨 런지': (329, 361),
        '오버 헤드 프레스': (361, 377),
        '사이드 레터럴 레이즈': (377, 409),
        '바벨 컬': (409, 441),
        '덤벨 컬': (441, 473),
        '라잉 레그 레이즈': (473, 489),
        '크런치': (489, 505),
        '바이시클 크런치': (505, 513),
        '시저크로스': (513, 545),
        '힙쓰러스트': (545, 553),
        '플랭크': (553, 561),
        '푸시업': (561, 593),
        '니푸시업': (593, 625),
        'Y - Exercise': (625, 633),
        '덤벨 체스트 플라이': (633, 641),
        '덤벨 인클라인 체스트 플라이': (641, 649),
        '덤벨 풀 오버': (649, 665),
        '라잉 트라이셉스 익스텐션': (665, 697),
        '딥스': (697, 713),
        '풀업': (713, 729),
        '행잉 레그 레이즈': (729, 737),
        '랫풀 다운': (737, 750),  ################# 원래 752
        '페이스 풀': (753, 761),
        '케이블 크런치': (761, 769),
        '케이블 푸시 다운': (769, 785),
        '로잉머신': (785, 817)
    }

    return exercises.get(exe_type)  # 해당하는 운동이 없는 경우는 넣지 않음

# data에서 특정 keypoint의 좌표를 추출하여 백터 형태로 반환
## data: 자세 데이터(dictionary), keypoint_name: keypoint 이름
def extract_vectors(data, keypoint_name):
    vectors = []
    for idx in data:
        if keypoint_name in data[idx]:
            coord = data[idx][keypoint_name]
            vector = [coord['x'], coord['y'], coord['z']]
            vectors.append(vector)
    return vectors

# 두 벡터 리스트 간의 코사인 유사도 계산
def calculate_cosine_similarity(vectors_a, vectors_b): # a vector input, b vector state
    sum_cosine_similarity = 0.0
    for i in range(0, 15): # 총 16 frame
        prevector_a = vectors_a[i]-vectors_a[i+1]#[(vectors_a[i][0] - vectors_a[i + 1][0]), (vectors_a[i][1]-vectors_a[i+1][1])]
        prevector_b = vectors_b[i]-vectors_b[i+1]#[(vectors_b[i][0] - vectors_b[i + 1][0]), (vectors_b[i][1]-vectors_b[i+1][1])]
        prevector_a[1] = (prevector_a[1]) * 3 # weight
        veca = normalize([prevector_a])
        vecb = normalize([prevector_b])
        cosine_similarities = cosine_similarity(veca,vecb)
        sum_cosine_similarity += cosine_similarities[0, 0]
    return sum_cosine_similarity


def pose_correction(exe_gui, csvp, pipe_conn):

    start, finish = load_exe(exe_gui)
    csv_path = os.path.join(csvp + '/jsonoutput.csv')               # CSV 파일 경로 설정

    # Data setup
    for i in range(start, finish+1):
        file_path = os.path.join(data_dir, f'D22-1-{i}-3d.json')    # JSON 파일 경로 설정
        output_file = os.path.join(output_dir, f'state_{i}.npy')    # 출력 파일 경로 설정

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            data_fra = data['frames']
            #print(data_fra)

        result = {}

        # 각 프레임의 자세 데이터 추출하여 dictionary에 저장
        for index, value in enumerate(data_fra):  # index == frames, value == pts 이하
            keypoints = value['pts']  # key가 신체부위, data가 좌표인 dictionary == keypoints
            keypoints_dict = {}
            for key, values in keypoints.items():
                keypoints_dict[key] = values  # 신체를 key로 하고 좌표를 data로 하는 dictionary to keypoints_dict
                result[index] = keypoints_dict  #

        keypoint_vectors = {}

        # 각 신체 부위의 좌표를 추출하여 벡터로 저장
        for keypoint_name in data_keypoint_names_2:
            vectors = extract_vectors(result, keypoint_name)
            keypoint_vectors[keypoint_name] = vectors

        np.save(output_file, keypoint_vectors)      # 추출한 벡터를 numpy 파일로 저장

    print('Data Setup')

    count = 0
    prev_time = 0
    input_appending = {}
    input_vector = {}
    stage_num = 1  # stages
    stage_check = 0

    # 자세 정확도 측정 loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            frame = pipe_conn.recv()    # 파이프를 통해 프레임 수신

            curr_time = time.time()     # 현재 시간 저장

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 이미지 형식 변환: BGR2RGB
            image.flags.writeable = False
            result = pose.process(image)                    # result: 이미지 자세 측정 프로세스 결과 저장(자세 정보 추출)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지 형식 변환: RGB2BGR

# if results true
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                input_dict = {}

                for keypoint_name in mp_keypoint_names:
                    x_past = landmarks[getattr(mp_pose.PoseLandmark, keypoint_name).value].x
                    y_past = landmarks[getattr(mp_pose.PoseLandmark, keypoint_name).value].y
                    z_past = landmarks[getattr(mp_pose.PoseLandmark, keypoint_name).value].z
                    x = x_past*(-26.001) - 11.0998
                    y = y_past*(-26.001) - 11.0998
                    z = z_past*(-26.001) - 11.0998

                    input_dict[data_keypoint_names[keypoint_name]] = [x, y, z]

                if curr_time - prev_time > 0.25:
                    # get input_dict, append to input_vector
                    prev_time = curr_time
                    stage_check += 1
                    #print (stage_check)
                    if stage_check > 15:
                        stage_check = 0

                    if len(input_appending) == 0:
                        for keypoint_name in data_keypoint_names_2:
                            input_appending[keypoint_name] = [input_dict[keypoint_name]]  # 130

                    else:
                        for keypoint_name in data_keypoint_names_2:
                            if keypoint_name not in input_dict:
                                print('화면에 동작이 모두 잡히지 않습니다.')
                                break
                            input_appending[keypoint_name].append(input_dict[keypoint_name])
                        count += 1

                    if count > 14:
                        input_vector = input_appending.copy()  # returning 16 frame input_vector
                        count = 0
                        input_appending.clear()

                    # 최대 유사도와 해당 state 파일의 숫자 부분을 저장할 변수 초기화
                        max_similarity = -1
                        most_similar_state_number = None

                        for i in range(start, finish):
                            state_file_path = os.path.join(output_dir, f'state_{i}.npy')
                            state_vectors = np.load(state_file_path, allow_pickle=True).item() #state vector@@@@@@@@@@@@@@@@@@@@@@@@@@@
                            # 이하 indent 추가
                            similarity_score = 0.0

                            for keypoint_name in data_keypoint_names_2:  # 소문자

                                input_vector_nparr = np.array(input_vector[keypoint_name])
                                #print (input_vector_nparr)
                                state_vectors_nparr = np.array(state_vectors[keypoint_name])  # key error

                                similarity_score += np.mean(
                                    calculate_cosine_similarity(input_vector_nparr, state_vectors_nparr))

                        # 최대 유사도와 해당 state 파일의 숫자 부분 업데이트
                            if similarity_score > max_similarity:
                                max_similarity = similarity_score
                                most_similar_state_number = i

                            if most_similar_state_number is None:  # 가장 유사한걸 못찾으면 이전 단계 사용
                                most_similar_state_number = i

                        # 동작 상태 및 개선 방향 출력
                        file_path_2d = os.path.join(data_dir, f'D22-1-{most_similar_state_number}.json')
                        print (file_path_2d)

                    # JSON 파일 로드
                        with open(file_path_2d, 'r', encoding='utf-8') as json_file:
                            data_state_2d = json.load(json_file)
                            data_type_info = data_state_2d['type_info']

                    # list 작성

                        result_list = []

                        print("동작 정확도:")
                        for condition_dict in data_type_info['conditions']:
                            condition = condition_dict['condition']
                            value = condition_dict['value']

                            value_state = 'O' if value else 'X'

                            print(f"{condition}: {value_state}")
                            result_list.append(condition)
                            result_list.append(value_state) # o, x 표시

                        print('\n')

                        print(f"수정할 동작: {data_type_info['description']}")
                        result_list.append(data_type_info['description']) # 보완할 부분 표시
                        result_list.append('0')
                        result_list.insert(8, '수정할 동작:')
                        #print(result_list)
                        if not os.path.isfile(csv_path) :
                            with open(csv_path, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(result_list)
                        else :
                            with open(csv_path, mode='a', newline='') as file:  # 'a' for append
                                writer = csv.writer(file)
                                writer.writerow(result_list)

            # cv print
#indent finished
            if cv2.waitKey(10) & 0xFF == ord('q'):
                os.remove(csv_path)
                break

        cap.release()
        cv2.destroyAllWindows()

    video_path = '/home/jeon/maker/video/latpulldown.mp4'
    video = cv2.VideoCapture(video_path)

    # 반복 재생을 위한 루프
    while True:
        # 동영상 한 프레임씩 읽기
        ret, frame = video.read()

        if not ret:
            # 동영상의 끝에 도달하면 처음부터 재생
            video.release()
            video = cv2.VideoCapture(video_path)
            continue

        # 여기에서 프레임을 result에 추가하거나 원하는 처리를 수행합니다.
        result = frame

        # 이미지 처리 및 표시 등의 추가 작업이 필요한 경우 여기서 수행합니다.

        # 'q' 키를 누르면 루프 탈출
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    # 창 닫기 및 리소스 해제
    video.release()
    cv2.destroyAllWindows()

    # 여기서 result를 반환할 수 있도록 코드를 수정합니다.
    return result