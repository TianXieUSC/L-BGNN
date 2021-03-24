rm ./output/bgnn-adv/movielens/set*

# epoch
for i in 2 4 6 8 10; do
  # layer_depth
  for j in 2 4 6 8 10; do
    # dis_hidden
    for k in 32 64 128 256; do
      python bgnn_main.py --dataset movielens --model adv --epoch $i --layer_depth $j --dis_hidden $k --weight_decay 0.0005 --dropout 0.5 --batch_size 256
      python recommendation.py
    done
  done
done
