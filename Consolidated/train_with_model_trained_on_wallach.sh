python main.py --batch_size 8 --total_epochs 60 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-5 \
--resume /home/SDD_Work/Models/Model_ColorVersion_from_N3_Epoch_scratch/FlowNet2_checkpoint.pth.tar \
--training_dataset MpiSintel --training_dataset_root /home/MedathatiExt/OpticalFlow/Data/MPISintel/training \
--validation_dataset Wallach --validation_dataset_root /home/SDD_Work/ColorVersion_CC1 \
--inference_dataset Wallach --inference_dataset_root /home/SDD_Work/ColorVersion_CC1 \
-s /home/SDD_Work/Models/Model_pretrained_on_wallach_epoch_scratch/ \
--validation_frequency 5 --render_validation 


