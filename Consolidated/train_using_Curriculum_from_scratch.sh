python main.py --batch_size 8 --total_epochs 5 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Wallach --training_dataset_root /home/SDD_Work/ColorVersion/ \
--validation_dataset MpiSintel --validation_dataset_root /home/MedathatiExt/OpticalFlow/Data/MPISintel/training \
--resume /home/SDD_Work/Models/Model_ColorVersion_from_2_Epoch_scratch/FlowNet2_checkpoint.pth.tar \
-s /home/SDD_Work/Models/Model_ColorVersion_from_7_Epoch_scratch/


