# SGAN-RSU

## train.py
```
python train.py --dataset_name bookstore0 --obs_len 8 --pred_len 8 --batch_size 32 --checkpoint_name checkpoint 
```

## evaluate_moduel.py
```
python evaluate_model.py --model_path ./checkpoint_with_model.pt --dataset_name bookstore0 
```
