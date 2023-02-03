example:
python evaluate.py --dset1 folder_1 --dset2 folder_2

notice the code uses torch dataloader ImageFolder, so inside of each folder, all images should be placed into a same-named subfolder


# CUDA_VISIBLE_DEVICES=0 python evaluate.py --dset1 dog --dset2 NADA_outputs_2/cat-id1 #nada cat to dog
