python main.py --batch_size 12 --total_epochs 10 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Wallach --training_dataset_root /home/SDD_Work/ColorVersion/ \
--validation_dataset MpiSintel --validation_dataset_root /home/SDD_Work/ColorVersion/ \
--resume ./Model_Wallach_moving_circle_full_r42_c255_GT/FlowNet2_checkpoint.pth.tar \
-s /home/SDD_Work/Models/Model_ColorVersion/


