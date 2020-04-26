python main.py --batch_size 8 --total_epochs 3 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Wallach --training_dataset_root /media/medathati/2910b1ae-0b9f-4c66-b5eb-357d01a713d7/medathati/Work/ColorVersion_CC1/ \
--validation_dataset MpiSintel --validation_dataset_root /media/medathati/MedathatiExt/OpticalFlow/Data/MPISintel/training \
--inference_dataset MpiSintel --inference_dataset_root /media/medathati/MedathatiExt/OpticalFlow/Data/MPISintel/training \
-s /media/medathati/2910b1ae-0b9f-4c66-b5eb-357d01a713d7/medathati/Work/Models/Model_ColorVersion_from_31_Epoch_scratch/ \
--validation_frequency 2 --render_validation 


