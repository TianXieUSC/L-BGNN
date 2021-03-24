rm ./output/bgnn-adv/dblp/set*

python bgnn_main.py --dataset dblp --model adv --epoch 2 --layer_depth 2 --dis_hidden 128 --weight_decay 0.0005 --dropout 0.5 --batch_size 256

python node_classification.py