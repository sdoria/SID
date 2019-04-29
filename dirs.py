import glob
import os

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'


# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

in_files = [[glob.glob(input_dir + '%05d_00*.ARW' % train_id)] for train_id in train_ids]


#input_train_fns = glob.glob(input_dir + '0*.ARW')
#input_train_ids = [os.path.basename(train_fn) for train_fn in train_fns]

# val IDs
val_fns = glob.glob(gt_dir + '2*.ARW')
val_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in val_fns]
##input_val_fns = glob.glob(input_dir + '2*.ARW')
#input_val_ids = [os.path.basename(train_fn) for train_fn in val_fns]


# test IDs
test_fns = glob.glob(gt_dir + '1*.ARW')
test_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in test_fns]
#input_test_fns = glob.glob(input_dir + '1*.ARW')
#input_test_ids = [os.path.basename(train_fn) for train_fn in test_fns]

