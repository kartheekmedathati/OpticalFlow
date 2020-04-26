python main.py --batch_size 8 --total_epochs 50 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-5 \
--resume /home/SDD_Work/Models/Pretrained_Model/FlowNet2_checkpoint.pth.tar \
--training_dataset Wallach --training_dataset_root /home/SDD_Work/ColorVersion_CC1/ \
--validation_dataset MpiSintel --validation_dataset_root /home/MedathatiExt/OpticalFlow/Data/MPISintel/training \
--inference_dataset MpiSintel --inference_dataset_root /home/MedathatiExt/OpticalFlow/Data/MPISintel/training \
-s /home/SDD_Work/Models/Model_ColorVersion_from_UsingPretrained_Epoch_scratch/ \
--validation_frequency 5 --render_validation 


