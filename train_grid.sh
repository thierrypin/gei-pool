
for opt in sgd # nadam adam
do
    for model in gru_ gru256 gru1024
    do
        for lr in 0.001 0.0001
        do
            echo "******************************"
            echo "$opt, $model, $lr"
            echo "******************************"
            python train_lstm.py --exp "$opt-$lr" --batch_size 16 --model $model --train_list CASIAB/features_mobilenet/list_train --val_list CASIAB/features_mobilenet/list_val --optimizer $opt --max_lr $lr --nb_epochs 30
        done
    done
done

