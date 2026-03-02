import os
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    use_gpu=False
)

result = ocr.ocr(
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png",
    cls=False
)

for line in result:
    print(line)