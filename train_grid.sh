#!/bin/bash


for i in {1..10}
do
    export fold=`printf "%02d" $i`
    for angle in 55 65 75 85
    do
        for model in geinet no_pool_geinet no_pool
        do  
            export exp="$fold-$angle"
            echo "******************************"
            echo "$fold, $model, $angle"
            echo "******************************"
            python train.py --exp $exp --model $model --train_list OULP/CV$fold.txt_gallery_$angle --val_list OULP/CV$fold.txt_probe_$angle
        done
    done
done




