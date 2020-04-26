python main.py --batch_size 16 --total_epochs 100 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-5 \
--training_dataset Wallach --training_dataset_root /home/SDD_Work/ColorVersion_FC1 \
--inference_dataset MpiSintel --inference_dataset_root /home/MedathatiExt/OpticalFlow/Data/MPISintel/training \
--validation_dataset MpiSintel --validation_dataset_root /home/MedathatiExt/OpticalFlow/Data/MPISintel/training \
--resume /home/SDD_Work/Models/Pretrained_Model/FlowNet2_checkpoint.pth.tar \
-s /home/SDD_Work/Models/Model_finetune_Wallach/ \
--validation_frequency 5 --render_validation 


