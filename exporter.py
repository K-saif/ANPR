
from rfdetr import RFDETRNano

model = RFDETRNano(pretrain_weights="/home/medprime/Music/ANPR/licence_plate_checkpoint_best_total.pth")

model.export()


