

ffmpeg - i /Users/fgtear/Downloads/8.mp4 -t 30 -c:v copy -c:a copy -y -loglevel error aaa.mp4


## MOSI:
resolution: 480 × 360
train: 1284
valid: 229
test: 686
total: 2199


## MOSEI:
resolution: 1280 × 720
train: 16326
valid: 1871
test: 4659
total: 22856


## CH-SIMS:
resolution: 1920 × 804
train: 1368
valid: 456
test: 457
total:  2281

M：multimodal，T：text，A：audio，V：visual


## CH-SIMSv2:
resolution: 1920 × 1080
train: 2722
valid: 647
test: 1034
total: 4403


| Item | Total | NEG | WNEG | NEU | WPOS | POS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #Train | 2722 | 921 | 433 | 232 | 318 | 818 |
| #Valid | 647 | 224 | 110 | 62 | 83 | 168 |
| #Test | 1034 | 291 | 211 | 93 | 183 | 256 |


| Item | CH-SIMS | CH-SIMS v2.0 (s) | CH-SIMS v2.0 (u) |
| :--- | :---: | :---: | :---: |
| Total videos number | 60 | 145 | 161 |
| Total segments number | 2281 | 4402 | 10161 |
| Min segments duration(s) | 1.2 | 0.6 | 0.2 |
| Max segments duration(s) | 16 | 19 | 28 |
| Average segments duration(s) | 3.67 | 3.63 | 4.71 |
| Max word count | 44 | 47 | 96 |
| Average word count | 15.8 | 17 | 19.3 |
| Standard deviation word count | 7.3 | 7 | 10 |



models = {
    "facial_recognition": {
        "VGG-Face": VGGFace.VggFaceClient,
        "OpenFace": OpenFace.OpenFaceClient,
        "Facenet": Facenet.FaceNet128dClient,
        "Facenet512": Facenet.FaceNet512dClient,
        "DeepFace": FbDeepFace.DeepFaceClient,
        "DeepID": DeepID.DeepIdClient,
        "Dlib": Dlib.DlibClient,
        "ArcFace": ArcFace.ArcFaceClient,
        "SFace": SFace.SFaceClient,
        "GhostFaceNet": GhostFaceNet.GhostFaceNetClient,
    },
    "spoofing": {
        "Fasnet": FasNet.Fasnet,
    },
    "facial_attribute": {
        "Emotion": Emotion.EmotionClient,
        "Age": Age.ApparentAgeClient,
        "Gender": Gender.GenderClient,
        "Race": Race.RaceClient,
    },
    "face_detector": {
        "opencv": OpenCv.OpenCvClient,
        "mtcnn": MtCnn.MtCnnClient,
        "ssd": Ssd.SsdClient,
        "dlib": DlibDetector.DlibClient,
        "retinaface": RetinaFace.RetinaFaceClient,
        "mediapipe": MediaPipe.MediaPipeClient,
        "yolov8": Yolo.YoloClient,
        "yunet": YuNet.YuNetClient,
        "fastmtcnn": FastMtCnn.FastMtCnnClient,
        "centerface": CenterFace.CenterFaceClient,
    },
}