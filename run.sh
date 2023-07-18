python main.py --model pfcl --dataset cifar10 --n_classes 10 --n_tasks 5 --lr 0.03 --n_epochs 50 --aux_dataset caltech256 --alpha 0.5 --device_id 0

python main.py --model pfcl --dataset cifar100 --n_classes 100 --n_tasks 5 --lr 0.03 --n_epochs 50 --aux_dataset caltech256 --alpha 0.5 --device_id 0

python main.py --model pfcl --dataset cifar100 --n_classes 100 --n_tasks 10 --lr 0.03 --n_epochs 50 --aux_dataset caltech256 --alpha 0.5 --device_id 0

python main.py --model pfcl --dataset cifar100 --n_classes 100 --n_tasks 20 --lr 0.03 --n_epochs 50 --aux_dataset caltech256 --alpha 0.5 --device_id 0

python main.py --model pfcl --dataset tinyimg --n_classes 200 --n_tasks 10 --lr 0.03 --n_epochs 100 --aux_dataset caltech256 --alpha 1.0 --device_id 0

python main.py --model pfcl --dataset rot-mnist --n_classes 10 --n_tasks 20 --lr 0.03 --n_epochs 1 --aux_dataset caltech256 --alpha 1.0 --batch_size 64 --device_id 1 