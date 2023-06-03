CUDA_VISIBLE_DEVICES=0 python3 faster_rcnn_train_net.py --num-gpus 1 --dist-url auto \
	--config-file configs/fsod/single_branch_pretraining_vbtv_pvt_v2_b2_li.yaml 2>&1 | tee log/single_branch_pretraining_vbtv_pvt_v2_b2_li.txt
