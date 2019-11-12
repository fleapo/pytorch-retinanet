#nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made/csv/coco/coco/ --depth 18 --epochs 1 &
#nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made/train_val/revers_gauss_coco/coco --depth 18 --epochs 1 --save_path ./output_models/main_detect_v2_blur/ >./output_models/main_detect_v2_blur/nohup 2>&1 &
#nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made/all_made_coco/coco --depth 18 --epochs 1 --save_path ./output_models/main_detect_v3_mix/ >./output_models/main_detect_v3_mix/nohup 2>&1 &
#nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made_add_txt_coco/coco --depth 18 --epochs 2 --save_path ./output_models/main_detect_v4_txt/ >./output_models/main_detect_v4_txt/nohup 2>&1 &
#nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/txt_logo_coco/coco --depth 18 --epochs 2 --save_path ./output_models/main_detect_v5_logo/ >./output_models/main_detect_v5_logo/nohup 2>&1 &
#nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/txt_logo_coco/coco --depth 18 --epochs 3 --save_path ./output_models/main_detect_v6_big_iou/ >./output_models/main_detect_v6_big_iou/nohup 2>&1 &
#nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made/all_made_191111_coco/coco --depth 18 --epochs 3 --save_path ./output_models/main_detect_v7_video/ >./output_models/main_detect_v7_video/nohup 2>&1 &
#env CUDA_VISIBLE_DEVICE=0 nohup python -u train.py --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made/all_made_191111_coco/coco --depth 18 --epochs 3 --save_path ./output_models/main_detect_v8_small_im/ >./output_models/main_detect_v8_small_im/nohup 2>&1 &

#python visualize.py  --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made/csv/coco/coco/ --model model_final.pt
#python visualize.py  --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/x小视频封面图/made/train_val/revers_gauss_coco/coco --output_path ./output_imgs/main_detect_v2_blur/ --model ./output_models/main_detect_v2_blur/model_final.pt
#python visualize.py  --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/smart_coco_format/coco --output_path ./output_imgs/smart_res/ --model ./output_models/main_detect_v2_blur/model_final.pt
#python visualize.py  --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/smart_coco_format/coco --output_path ./output_imgs/smart_res_v3_mix/ --model ./output_models/main_detect_v3_mix/model_final.pt
#python visualize.py  --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/smart_coco_format/coco --output_path ./output_imgs/smart_res_v4_txt/ --model ./output_models/main_detect_v4_txt/model_final.pt
#python visualize.py  --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/smart_coco_format/coco --output_path ./output_imgs/smart_res_v6_big_iou/ --model ./output_models/main_detect_v6_big_iou/model_final.pt
#python visualize.py  --dataset coco --coco_path /home/hao.wyh/jupyter/黑边/smart_reverse_label/coco/ --output_path ./output_imgs/smart_res_v7_video/ --model ./output_models/main_detect_v7_video/coco_retinanet_1.pt
