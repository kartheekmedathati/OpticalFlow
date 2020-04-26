# Training curriculum 

python main.py --batch_size 8 --total_epochs 2 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Wallach --training_dataset_root /home/MedathatiExt/OpticalFlow/Stimuli/Wallach_dataset/moving_circle_full_r42_c255_GT/  \
--validation_dataset Wallach --validation_dataset_root /home/MedathatiExt/OpticalFlow/Stimuli/Wallach_dataset/moving_circle_full_r21_c255_GT/ \
--resume /home/MedathatiExt/OpticalFlow/trainable/AlexHall/flownet2-pytorch/flownet_models/FlowNet2_checkpoint.pth.tar \
-s ./Model_Wallach_moving_circle_full_r42_c255_GT


python main.py --batch_size 8 --total_epochs 2 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Wallach --training_dataset_root /home/MedathatiExt/OpticalFlow/Stimuli/Wallach_dataset/moving_gratings_circ_aper_R_125_c255_GT/  \
--validation_dataset Wallach --validation_dataset_root /home/MedathatiExt/OpticalFlow/Stimuli/Wallach_dataset/moving_gratings_circ_aper_R_100_c255_GT/ \
--resume ./Model_Wallach_moving_circle_full_r42_c255_GT/FlowNet2_checkpoint.pth.tar \
-s ./Model_Wallach_moving_gratings_circ_aper_R_100_c255_GT/



python main.py --batch_size 8 --total_epochs 2 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Wallach --training_dataset_root /home/MedathatiExt/OpticalFlow/Stimuli/ColorVersion \
--validation_dataset Wallach --validation_dataset_root /home/MedathatiExt/OpticalFlow/Stimuli/ColorVersion/ \
--resume ./Model_Wallach_moving_circle_full_r42_c255_GT/FlowNet2_checkpoint.pth.tar \
-s ./Model_Wallach_ColorVersion/
