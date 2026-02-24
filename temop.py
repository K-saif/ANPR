
from rfdetr import RFDETRNano

model = RFDETRNano(pretrain_weights="/home/medprime/Music/ANPR/rf-detr-nano.pth")

model.export()