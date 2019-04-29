
date
rsync -azvP --exclude CASIAB --exclude OULP --exclude experiments --exclude good_models --exclude */__pycache__ --exclude __pycache__ . jesus9.recogna.tech:/home/thierry/workspace/gei-pool

# python train_vol.py --model volume --eager True --nb_frames 64 --max_lr 0.001 --batch_size 8 --nb_epochs 50

