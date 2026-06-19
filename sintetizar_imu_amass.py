# sintetizar_imu_amass.py
# Extrae .npz de un tar.bz2 de AMASS, sintetiza IMU virtuales,
# guarda DOS archivos .npz compatibles con DIP_IMU_nn:
#   OUT_FILE   (6 sensores): orientation(T,54), acceleration(T,18)  -> IN_DIM=72
#   OUT_FILE_5 (5 sensores): orientation(T,45), acceleration(T,15)  -> IN_DIM=60
# OUT_FILE_5 es compatible con el modelo M2 entrenado con DIP_IMU_nn (60 features).
# El sensor descartado para la version de 5 es 'back' (spine3, indice 1 en IMU_JOINTS).

# --- SECCION 0: CONFIGURACION ---

DRIVE_ROOT  = '/content/drive/MyDrive'
AMASS_TAR   = DRIVE_ROOT + '/TFM_DT/BMLmovi.tar.bz2'
OUT_DIR     = DRIVE_ROOT + '/TFM_DT/DIPIMUandOthers'
OUT_FILE    = OUT_DIR + '/imu_amass_synthetic_6s.npz'   # 6 sensores, IN_DIM=72
OUT_FILE_5  = OUT_DIR + '/imu_amass_bmlmovi_5s.npz'     # 5 sensores, IN_DIM=60 (DIP-compatible)

# Windows local (descomentar si se usa localmente):
# AMASS_TAR  = r'C:\Users\34693\DIP_IMU_nn\AMASS_subset.tar.bz2'
# OUT_DIR    = r'C:\Users\34693\DIP_IMU_nn'
# OUT_FILE   = r'C:\Users\34693\DIP_IMU_nn\imu_amass_synthetic_6s.npz'
# OUT_FILE_5 = r'C:\Users\34693\DIP_IMU_nn\imu_amass_synthetic_5s.npz'

MAX_SEQS      = 500
TARGET_FPS    = 60
T_MIN         = 64
T_MAX         = 3000
ADD_NOISE     = True
ORI_NOISE_STD = 0.01
ACC_NOISE_STD = 0.15
N_IMUS        = 6

# --- SECCION 1: IMPORTACIONES ---

import os
import sys
import tarfile
import io
import warnings
import numpy as np
from scipy.spatial.transform import Rotation

warnings.filterwarnings('ignore')
np.random.seed(42)

print('NumPy', np.__version__)
print('Config: MAX_SEQS=%d, TARGET_FPS=%d, N_IMUS=%d' % (MAX_SEQS, TARGET_FPS, N_IMUS))
print('Entrada:', AMASS_TAR)
print('Salida :', OUT_FILE)

# --- SECCION 2: ESQUELETO SMPL ---

# Arbol cinematico SMPL (22 joints de cuerpo, sin dedos)
SMPL_PARENTS = np.array([
    -1, 0, 0, 0,    # 0:pelvis  1:L_hip    2:R_hip    3:spine1
     1, 2, 3, 4,    # 4:L_knee  5:R_knee   6:spine2   7:L_ankle
     5, 6, 7, 8,    # 8:R_ankle 9:spine3  10:L_foot  11:R_foot
     9, 9, 9, 12,   # 12:neck  13:L_collar 14:R_collar 15:head
    13, 14, 16, 17, # 16:L_shl  17:R_shl   18:L_elbow 19:R_elbow
    18, 19,         # 20:L_wrist 21:R_wrist
], dtype=np.int32)

N_JOINTS = len(SMPL_PARENTS)  # 22

# Offsets T-pose (LOCAL metros, relativo al padre)
SMPL_OFFSETS = np.array([
    [ 0.000,  0.000,  0.000],  # 0  pelvis
    [ 0.090, -0.090,  0.000],  # 1  L_hip
    [-0.090, -0.090,  0.000],  # 2  R_hip
    [ 0.000,  0.130,  0.000],  # 3  spine1
    [ 0.000, -0.420,  0.000],  # 4  L_knee
    [ 0.000, -0.420,  0.000],  # 5  R_knee
    [ 0.000,  0.130,  0.000],  # 6  spine2
    [ 0.000, -0.430,  0.000],  # 7  L_ankle
    [ 0.000, -0.430,  0.000],  # 8  R_ankle
    [ 0.000,  0.140,  0.000],  # 9  spine3
    [ 0.000, -0.070,  0.095],  # 10 L_foot
    [ 0.000, -0.070,  0.095],  # 11 R_foot
    [ 0.000,  0.140,  0.000],  # 12 neck
    [ 0.060,  0.130,  0.000],  # 13 L_collar
    [-0.060,  0.130,  0.000],  # 14 R_collar
    [ 0.000,  0.120,  0.000],  # 15 head
    [ 0.130,  0.000,  0.000],  # 16 L_shoulder
    [-0.130,  0.000,  0.000],  # 17 R_shoulder
    [ 0.270,  0.000,  0.000],  # 18 L_elbow
    [-0.270,  0.000,  0.000],  # 19 R_elbow
    [ 0.250,  0.000,  0.000],  # 20 L_wrist
    [-0.250,  0.000,  0.000],  # 21 R_wrist
], dtype=np.float64)

# Sensores IMU: (nombre, indice joint)
IMU_JOINTS = [
    ('head',         15),
    ('back',          9),
    ('left_wrist',   20),
    ('right_wrist',  21),
    ('left_ankle',    7),
    ('right_ankle',   8),
]
assert len(IMU_JOINTS) == N_IMUS

# Indice del sensor 'back' en IMU_JOINTS (se descarta en la version de 5 sensores)
BACK_SENSOR_IDX = 1   # IMU_JOINTS[1] = ('back', 9)

# 15 joints para smpl_pose (15 x 9 = 135)
POSE_JOINT_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19]
assert len(POSE_JOINT_IDX) == 15

GRAVITY = np.array([0.0, -9.81, 0.0], dtype=np.float64)

# --- SECCION 3: CINEMATICA DIRECTA ---

def aa_to_rotmat(aa):
    # axis-angle (..., 3) -> (..., 3, 3)
    shape = aa.shape[:-1]
    r = Rotation.from_rotvec(aa.reshape(-1, 3))
    return r.as_matrix().reshape(shape + (3, 3))


def forward_kinematics(poses_aa, trans):
    # Cinematica directa simplificada (T-pose canonica, sin forma corporal).
    # poses_aa: (T, >=66) SMPL+H axis-angle
    # trans   : (T, 3) traslacion raiz
    # Retorna : R_local (T,N,3,3), R_global (T,N,3,3), p_global (T,N,3)
    T = poses_aa.shape[0]
    aa = poses_aa[:, :N_JOINTS * 3].reshape(T, N_JOINTS, 3)
    R_local  = aa_to_rotmat(aa)
    R_global = np.zeros((T, N_JOINTS, 3, 3))
    p_global = np.zeros((T, N_JOINTS, 3))
    for j in range(N_JOINTS):
        parent = int(SMPL_PARENTS[j])
        if parent < 0:
            R_global[:, j] = R_local[:, j]
            p_global[:, j] = trans
        else:
            R_global[:, j] = np.einsum('tab,tbc->tac',
                                        R_global[:, parent], R_local[:, j])
            p_global[:, j] = (p_global[:, parent]
                               + np.einsum('tab,b->ta',
                                           R_global[:, parent],
                                           SMPL_OFFSETS[j]))
    return R_local, R_global, p_global


# --- SECCION 4: SINTESIS DE SENALES IMU ---

def synthesize_imu(R_global, p_global, fps, add_noise=False):
    # Sintetiza orientacion y aceleracion para todos los sensores.
    # Retorna: orientation (T, N_IMUS*9), acceleration (T, N_IMUS*3)
    T  = R_global.shape[0]
    dt = 1.0 / fps
    ori_list, acc_list = [], []
    for _, jidx in IMU_JOINTS:
        R_j = R_global[:, jidx]
        p_j = p_global[:, jidx]
        ori = R_j.reshape(T, 9).astype(np.float32)
        # 2a derivada numerica (diferencias centrales)
        a_world = np.zeros((T, 3), dtype=np.float64)
        if T > 2:
            a_world[1:-1] = (p_j[2:] - 2.0 * p_j[1:-1] + p_j[:-2]) / (dt * dt)
        if T > 1:
            a_world[0]  = a_world[1]
            a_world[-1] = a_world[-2]
        # Fuerza especifica: f_s = R^T (a_world - g_world)
        # En reposo (a_world=0): f_s = R^T [0,+9.81,0] apunta arriba
        a_sensor = np.einsum('tji,tj->ti', R_j, a_world - GRAVITY).astype(np.float32)
        if add_noise:
            ori      += np.random.normal(0, ORI_NOISE_STD, ori.shape).astype(np.float32)
            a_sensor += np.random.normal(0, ACC_NOISE_STD, a_sensor.shape).astype(np.float32)
        ori_list.append(ori)
        acc_list.append(a_sensor)
    return (np.concatenate(ori_list, axis=-1),
            np.concatenate(acc_list, axis=-1))


def extract_smpl_pose(R_local):
    # R_local (T, N_JOINTS, 3, 3) -> smpl_pose (T, 135)
    pose = R_local[:, POSE_JOINT_IDX, :, :]
    return pose.reshape(pose.shape[0], -1).astype(np.float32)


# --- SECCION 4b: RECORTE A 5 SENSORES (DIP-compatible, IN_DIM=60) ---

def trim_to_5sensors(ori, acc):
    # Elimina el sensor 'back' (indice BACK_SENSOR_IDX) de los arrays ya
    # calculados para los 6 sensores, produciendo shapes (T,45) y (T,15).
    #   ori : (T, N_IMUS*9)  ->  (T, 5*9=45)
    #   acc : (T, N_IMUS*3)  ->  (T, 5*3=15)
    idx = BACK_SENSOR_IDX
    # indices de columna a CONSERVAR para orientation (9 canales por sensor)
    keep_ori = list(range(0, idx * 9)) + list(range((idx + 1) * 9, N_IMUS * 9))
    # indices de columna a CONSERVAR para acceleration (3 canales por sensor)
    keep_acc = list(range(0, idx * 3)) + list(range((idx + 1) * 3, N_IMUS * 3))
    return ori[:, keep_ori], acc[:, keep_acc]


# --- SECCION 5: REMUESTREO ---

def resample_sequence(arr, src_fps, tgt_fps):
    # Interpola (T, ...) de src_fps a tgt_fps. Sin cambio si son iguales.
    if abs(src_fps - tgt_fps) < 0.5:
        return arr
    T_src  = arr.shape[0]
    T_tgt  = max(2, int(round(T_src * tgt_fps / src_fps)))
    t_src  = np.linspace(0.0, 1.0, T_src)
    t_tgt  = np.linspace(0.0, 1.0, T_tgt)
    flat   = arr.reshape(T_src, -1)
    result = np.zeros((T_tgt, flat.shape[1]), dtype=arr.dtype)
    for ch in range(flat.shape[1]):
        result[:, ch] = np.interp(t_tgt, t_src, flat[:, ch])
    return result.reshape((T_tgt,) + arr.shape[1:])


# --- SECCION 6: ESTADISTICAS ---

def compute_stats(sequences):
    # Estadisticas por canal sobre todas las secuencias de la lista.
    all_data  = np.concatenate(sequences, axis=0)
    seq_means = np.array([s.mean() for s in sequences])
    return {
        'mean_channel':  all_data.mean(axis=0).astype(np.float64),
        'std_channel':   all_data.std(axis=0).astype(np.float64),
        'mean_all':      float(all_data.mean()),
        'std_all':       float(all_data.std()),
        'max_channel':   all_data.max(axis=0).astype(np.float64),
        'min_channel':   all_data.min(axis=0).astype(np.float64),
        'max_all':       float(all_data.max()),
        'min_all':       float(all_data.min()),
        'mean_sequence': float(seq_means.mean()),
        'std_sequence':  float(seq_means.std()),
    }


# --- SECCION 7: PROCESADO DEL TAR.BZ2 ---

def process_tar(tar_path, max_seqs):
    # Abre el tar.bz2, extrae cada .npz y sintetiza IMU.
    orientations, accelerations, smpl_poses = [], [], []
    file_ids, data_ids = [], []
    skipped   = 0
    processed = 0
    print('Abriendo:', tar_path)
    with tarfile.open(tar_path, 'r:bz2') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith('.npz')]
        print('Archivos .npz encontrados:', len(members))
        for member in members:
            if processed >= max_seqs:
                break
            try:
                fobj = tar.extractfile(member)
                if fobj is None:
                    skipped += 1
                    continue
                npz = np.load(io.BytesIO(fobj.read()), allow_pickle=True)
            except Exception as e:
                print('  [SKIP]', member.name, str(e))
                skipped += 1
                continue
            if 'poses' not in npz.files or 'trans' not in npz.files:
                skipped += 1
                continue
            poses = npz['poses'].astype(np.float64)
            trans = npz['trans'].astype(np.float64)
            if poses.ndim != 2 or poses.shape[1] < N_JOINTS * 3:
                skipped += 1
                continue
            src_fps = float(TARGET_FPS)
            for k in ['mocap_framerate', 'mocap_frame_rate', 'frame_rate', 'fps']:
                if k in npz.files:
                    src_fps = float(npz[k])
                    break
            if abs(src_fps - TARGET_FPS) > 0.5:
                poses = resample_sequence(poses, src_fps, TARGET_FPS)
                trans = resample_sequence(trans, src_fps, TARGET_FPS)
            T = poses.shape[0]
            if T < T_MIN:
                skipped += 1
                continue
            if T > T_MAX:
                poses, trans = poses[:T_MAX], trans[:T_MAX]
                T = T_MAX
            try:
                R_local, R_global, p_global = forward_kinematics(poses, trans)
            except Exception as e:
                print('  [SKIP FK]', member.name, str(e))
                skipped += 1
                continue
            ori, acc = synthesize_imu(R_global, p_global, TARGET_FPS, ADD_NOISE)
            pose_out = extract_smpl_pose(R_local)
            if not (np.isfinite(ori).all() and np.isfinite(acc).all()
                    and np.isfinite(pose_out).all()):
                skipped += 1
                continue
            orientations.append(ori)
            accelerations.append(acc)
            smpl_poses.append(pose_out)
            file_ids.append(os.path.basename(member.name)[:64])
            data_ids.append('A%04d' % processed)
            processed += 1
            if processed % 10 == 0 or processed <= 5:
                print('  [%3d/%d] %s  T=%d  src_fps=%.0f' % (
                    processed, max_seqs, member.name, T, src_fps))
    print('Procesadas: %d | Descartadas: %d' % (processed, skipped))
    return orientations, accelerations, smpl_poses, file_ids, data_ids


# --- SECCION 8: GUARDADO EN NPZ ---

def save_dataset(orientations, accelerations, smpl_poses,
                 file_ids, data_ids, out_path):
    # Guarda en .npz con el mismo formato que DIP_IMU_nn (object arrays).
    n = len(orientations)
    print('Guardando %d secuencias en %s ...' % (n, out_path))
    ori_arr  = np.empty(n, dtype=object)
    acc_arr  = np.empty(n, dtype=object)
    pose_arr = np.empty(n, dtype=object)
    for i in range(n):
        ori_arr[i]  = orientations[i]
        acc_arr[i]  = accelerations[i]
        pose_arr[i] = smpl_poses[i]
    stats = {
        'orientation':  compute_stats(orientations),
        'acceleration': compute_stats(accelerations),
        'smpl_pose':    compute_stats(smpl_poses),
    }
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        out_path,
        orientation   = ori_arr,
        acceleration  = acc_arr,
        smpl_pose     = pose_arr,
        file_id       = np.array(file_ids, dtype='U64'),
        data_id       = np.array(data_ids, dtype='U8'),
        statistics    = np.array(stats, dtype=object),
        preprocessing = np.array(['normalized', 'synthetic'], dtype=str),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print('Guardado: %s  (%.1f MB)' % (out_path, size_mb))


# --- SECCION 9: VERIFICACION ---

def verify_output(out_path, label=''):
    print('\n--- VERIFICACION %s ---' % label)
    data = np.load(out_path, allow_pickle=True)
    n    = len(data['orientation'])
    print('Archivo   :', out_path)
    print('Secuencias:', n)
    for key in ['orientation', 'acceleration', 'smpl_pose']:
        arr    = data[key]
        sample = [arr[i].shape for i in range(min(3, n))]
        print('  %s: %s' % (key, sample))
    ori_c = data['orientation'][0].shape[1]
    acc_c = data['acceleration'][0].shape[1]
    print('  IN_DIM (ori+acc): %d + %d = %d' % (ori_c, acc_c, ori_c + acc_c))


# --- SECCION 10: MAIN ---

if __name__ == '__main__':
    print('SINTESIS IMU DESDE AMASS')
    print('Sensores IMU:')
    for name, jidx in IMU_JOINTS:
        print('  %-14s -> joint %d' % (name, jidx))
    print('Joints smpl_pose:', POSE_JOINT_IDX)
    if not os.path.isfile(AMASS_TAR):
        print('[ERROR] No se encontro:', AMASS_TAR)
        sys.exit(1)
    orientations, accelerations, smpl_poses, file_ids, data_ids = process_tar(
        AMASS_TAR, MAX_SEQS)
    if not orientations:
        print('[ERROR] Ninguna secuencia valida procesada.')
        sys.exit(1)

    # --- version 6 sensores (IN_DIM=72) ---
    save_dataset(orientations, accelerations, smpl_poses,
                 file_ids, data_ids, OUT_FILE)
    verify_output(OUT_FILE, label='6 sensores (IN_DIM=72)')

    # --- version 5 sensores (IN_DIM=60, DIP-compatible) ---
    # Recorta descartando el sensor 'back' (BACK_SENSOR_IDX=1)
    ori5_list = []
    acc5_list = []
    for ori, acc in zip(orientations, accelerations):
        o5, a5 = trim_to_5sensors(ori, acc)
        ori5_list.append(o5)
        acc5_list.append(a5)

    save_dataset(ori5_list, acc5_list, smpl_poses,
                 file_ids, data_ids, OUT_FILE_5)
    verify_output(OUT_FILE_5, label='5 sensores (IN_DIM=60, DIP-compatible)')

    print('\n[DONE]')
    print('Archivos generados:')
    print('  6 sensores: %s' % OUT_FILE)
    print('  5 sensores: %s' % OUT_FILE_5)
