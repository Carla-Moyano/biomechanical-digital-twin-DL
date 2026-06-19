import numpy as np

DATA_DIR = '/content/drive/MyDrive/TFM_DT/DIPIMUandOthers'
PATH_A   = DATA_DIR + '/imu_amass_bmlmovi_5s.npz'
PATH_B   = DATA_DIR + '/imu_amass_synthetic_5s.npz'
OUT      = DATA_DIR + '/imu_amass_combined.npz'
SEED     = 42

print('Cargando BMLmovi...')
da = np.load(PATH_A, allow_pickle=True)
ori_a  = list(da['orientation'])
acc_a  = list(da['acceleration'])
pose_a = list(da['smpl_pose'])
print('  %d secuencias' % len(ori_a))

print('Cargando CMU...')
db = np.load(PATH_B, allow_pickle=True)
ori_b  = list(db['orientation'])
acc_b  = list(db['acceleration'])
pose_b = list(db['smpl_pose'])
print('  %d secuencias' % len(ori_b))

ori_all  = ori_a  + ori_b
acc_all  = acc_a  + acc_b
pose_all = pose_a + pose_b
n = len(ori_all)
print('Total: %d secuencias' % n)

rng  = np.random.default_rng(SEED)
perm = rng.permutation(n)
ori_all  = [ori_all[i]  for i in perm]
acc_all  = [acc_all[i]  for i in perm]
pose_all = [pose_all[i] for i in perm]

ori_obj  = np.empty(n, dtype=object)
acc_obj  = np.empty(n, dtype=object)
pose_obj = np.empty(n, dtype=object)
for i in range(n):
    ori_obj[i]  = ori_all[i]
    acc_obj[i]  = acc_all[i]
    pose_obj[i] = pose_all[i]

print('Guardando...')
np.savez(OUT, orientation=ori_obj, acceleration=acc_obj, smpl_pose=pose_obj)
print('Listo: %s' % OUT)
