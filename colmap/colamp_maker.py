import glob
import os

import numpy as np
from database import COLMAPDatabase, array_to_blob, blob_to_array


def find_files(directory, exts):
    """Find all files in a directory that have a certain file extension.

    Parameters
    ----------
    directory : str
        The directory to search for files.
    exts : list of str
        A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns
    -------
    list of str
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    return []


def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename, encoding="UTF-8").read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)


def get_camera_params(data_path, scene, split):
    split = "validation" if split == "val" else split
    split_dir = f"{data_path}/{scene}/final/{split}"

    intrinsics_files = find_files(f"{split_dir}/intrinsics", exts=["*.txt"])
    pose_files = find_files(f"{split_dir}/pose", exts=["*.txt"])
    img_names = find_files(f"{split_dir}/rgb", exts=["*.JPG"])

    num_cams = len(pose_files)

    intrinsics = []
    camera_to_worlds = []
    world_to_cameras = []
    for i in range(num_cams):
        intrinsics.append(parse_txt(intrinsics_files[i]))

        c2w = parse_txt(pose_files[i])
        w2c = np.linalg.inv(c2w)

        camera_to_worlds.append(c2w)
        world_to_cameras.append(w2c)

    intrinsics = np.stack(intrinsics).astype(np.float64)  # [N, 4, 4]
    camera_to_worlds = np.stack(camera_to_worlds).astype(np.float64)  # [N, 4, 4]
    world_to_cameras = np.stack(world_to_cameras).astype(np.float64)  # [N, 4, 4]

    return intrinsics, camera_to_worlds, world_to_cameras, img_names, num_cams


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


scene = "stjacob"

# Path to the data folder
data_folder = "/workspaces/sdfstudio/data/NeRF-OSR/Data"

# Path to the manually created sparse model folder
base_scene_folder = f"/workspaces/sdfstudio/colmap/{scene}"
sparse_model_folder = f"{base_scene_folder}/sparse/manual/"

# get all split cam params
intrinsics_train, camera_to_worlds_train, world_to_cameras_train, img_names_train, n_train = get_camera_params(
    data_folder, scene, "train"
)
intrinsics_val, camera_to_worlds_val, world_to_cameras_val, img_names_val, n_val = get_camera_params(
    data_folder, scene, "val"
)
intrinsics_test, camera_to_worlds_test, world_to_cameras_test, img_names_test, n_test = get_camera_params(
    data_folder, scene, "test"
)

# stack all cam params
intrinsics = np.concatenate([intrinsics_train, intrinsics_val, intrinsics_test], axis=0)
camera_to_worlds = np.concatenate([camera_to_worlds_train, camera_to_worlds_val, camera_to_worlds_test], axis=0)
world_to_cameras = np.concatenate([world_to_cameras_train, world_to_cameras_val, world_to_cameras_test], axis=0)
# combine image name lists into one
img_names = img_names_train + img_names_val + img_names_test

# make sparse.txt file and leave blank
sparse_file = os.path.join(sparse_model_folder, "points3D.txt")
with open(sparse_file, "w") as f:
    f.write("")

for i in range(len(img_names)):
    # copy image into base model folder /images
    os.system(f"cp {img_names[i]} {os.path.join(base_scene_folder, 'images', f'image{str(i).zfill(4)}.jpg')}")

cmd = f"colmap feature_extractor --database_path {base_scene_folder}/{scene}.db \
                                --image_path {os.path.join(base_scene_folder, 'images')} \
                                --ImageReader.single_camera 0 \
                                --ImageReader.camera_model PINHOLE \
                                --SiftExtraction.max_image_size 1280  \
                                --SiftExtraction.estimate_affine_shape 0 \
                                --SiftExtraction.domain_size_pooling 1 \
                                --SiftExtraction.use_gpu 1 \
                                --SiftExtraction.max_num_features 16384 \
                                --SiftExtraction.gpu_index {-1}"

os.system(cmd)

# load the database
db = COLMAPDatabase.connect(f"{base_scene_folder}/{scene}.db")

db_image_order = []
images = db.execute("SELECT * FROM images ORDER BY image_id")
images = images.fetchall()
for i in range(len(images)):
    db_image_order.append(images[i][1])

# Write the cameras.txt file and update cameras in database
cameras_file = os.path.join(sparse_model_folder, "cameras.txt")
with open(cameras_file, "w") as f:
    f.write(f"# Camera list with one line of data per camera:\n")
    f.write(f"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write(f"# Number of cameras: {len(intrinsics)}\n")
    for i, intrinsic_matrix in enumerate(intrinsics):
        f.write(
            f"{i + 1} PINHOLE {round(intrinsic_matrix[0,2] * 2, 6):.6f} {round(intrinsic_matrix[1,2] * 2, 6):.6f} {round(intrinsic_matrix[0,0], 6):.6f} {round(intrinsic_matrix[1,1], 6):.6f} {round(intrinsic_matrix[0,2], 6):.6f} {round(intrinsic_matrix[1,2], 6):.6f}"
        )
        f.write("\n")

        # get params
        model, width, height, params = (
            1,
            intrinsic_matrix[0, 2] * 2,
            intrinsic_matrix[1, 2] * 2,
            np.array(
                (intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]),
                dtype=np.float64,
            ),
        )

        # update database
        db.execute(
            "UPDATE cameras SET model = ?, width = ?, height = ?, params = ?, prior_focal_length = ? WHERE camera_id = ?",
            (model, width, height, array_to_blob(params), 0, i + 1),
        )
        db.commit()

# Write the images.txt file
images_file = os.path.join(sparse_model_folder, "images.txt")
with open(images_file, "w") as f:
    f.write(f"# Image list with two lines of data per image:\n")
    f.write(f"#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write(f"#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f.write(f"# Number of images: {len(img_names)}, mean observations per image: 0\n")
    for i, pose_matrix in enumerate(camera_to_worlds):
        # add to database
        qvec = rotmat2qvec(pose_matrix[:3, :3])
        tvec = pose_matrix[:3, 3]

        f.write(
            f"{i + 1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {i + 1} {db_image_order[i]}"
        )
        f.write("\n\n")

        db.execute(
            "UPDATE images SET camera_id = ?, prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?, prior_tx = ?, prior_ty = ?, prior_tz = ? WHERE image_id = ?",
            (i + 1, qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2], i + 1),
        )
        db.commit()

db.close()

# feature matching
cmd = f"colmap exhaustive_matcher --database_path {base_scene_folder}/{scene}.db \
                                  --SiftMatching.guided_matching 1 \
                                  --SiftMatching.use_gpu 1 \
                                  --SiftMatching.max_num_matches 65536 \
                                  --SiftMatching.max_error 3 \
                                  --SiftMatching.gpu_index {-1}"

os.system(cmd)

cmd = f"colmap point_triangulator --database_path {base_scene_folder}/{scene}.db \
                                  --image_path {os.path.join(base_scene_folder, 'images')} \
                                  --input_path {sparse_model_folder} \
                                  --output_path {sparse_model_folder}"

os.system(cmd)
# # %%
# import glob
# import os

# import numpy as np
# from database import COLMAPDatabase, array_to_blob, blob_to_array


# def find_files(directory, exts):
#     """Find all files in a directory that have a certain file extension.

#     Parameters
#     ----------
#     directory : str
#         The directory to search for files.
#     exts : list of str
#         A list of file extensions to search for. Each file extension should be in the form '*.ext'.

#     Returns
#     -------
#     list of str
#         A list of file paths for all the files that were found. The list is sorted alphabetically.
#     """
#     if os.path.isdir(directory):
#         # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
#         files_grabbed = []
#         for ext in exts:
#             files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
#         if len(files_grabbed) > 0:
#             files_grabbed = sorted(files_grabbed)
#         return files_grabbed
#     return []


# def parse_txt(filename):
#     assert os.path.isfile(filename)
#     nums = open(filename, encoding="UTF-8").read().split()
#     return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)


# def get_camera_params(data_path, scene, split):
#     split = "validation" if split == "val" else split
#     split_dir = f"{data_path}/{scene}/final/{split}"

#     intrinsics_files = find_files(f"{split_dir}/intrinsics", exts=["*.txt"])
#     pose_files = find_files(f"{split_dir}/pose", exts=["*.txt"])
#     img_names = find_files(f"{split_dir}/rgb", exts=["*.JPG"])

#     num_cams = len(pose_files)

#     intrinsics = []
#     camera_to_worlds = []
#     for i in range(num_cams):
#         intrinsics.append(parse_txt(intrinsics_files[i]))

#         pose = parse_txt(pose_files[i])

#         camera_to_worlds.append(pose)

#     intrinsics = np.stack(intrinsics).astype(np.float64)  # [N, 4, 4]
#     camera_to_worlds = np.stack(camera_to_worlds).astype(np.float64)  # [N, 4, 4]

#     return intrinsics, camera_to_worlds, img_names, num_cams


# def rotmat2qvec(R):
#     Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
#     K = (
#         np.array(
#             [
#                 [Rxx - Ryy - Rzz, 0, 0, 0],
#                 [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
#                 [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
#                 [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
#             ]
#         )
#         / 3.0
#     )
#     eigvals, eigvecs = np.linalg.eigh(K)
#     qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
#     if qvec[0] < 0:
#         qvec *= -1
#     return qvec


# scene = "stjacob"

# # Path to the data folder
# data_folder = "/workspaces/sdfstudio/data/NeRF-OSR/Data"

# # Path to the manually created sparse model folder
# base_scene_folder = f"/workspaces/sdfstudio/colmap/{scene}"
# sparse_model_folder = f"{base_scene_folder}/sparse/manual/"

# # get all split cam params
# intrinsics_train, camera_to_worlds_train, img_names_train, n_train = get_camera_params(data_folder, scene, "train")
# intrinsics_val, camera_to_worlds_val, img_names_val, n_val = get_camera_params(data_folder, scene, "val")
# intrinsics_test, camera_to_worlds_test, img_names_test, n_test = get_camera_params(data_folder, scene, "test")

# # stack all cam params
# intrinsics = np.concatenate([intrinsics_train, intrinsics_val, intrinsics_test], axis=0)
# camera_to_worlds = np.concatenate([camera_to_worlds_train, camera_to_worlds_val, camera_to_worlds_test], axis=0)
# # combine image name lists into one
# img_names = img_names_train + img_names_val + img_names_test

# # # make sparse.txt file and leave blank
# # sparse_file = os.path.join(sparse_model_folder, "points3D.txt")
# # with open(sparse_file, "w") as f:
# #     f.write("")

# # for i in range(len(img_names)):
# #     # copy image into base model folder /images
# #     os.system(f"cp {img_names[i]} {os.path.join(base_scene_folder, 'images', f'image{str(i).zfill(4)}.jpg')}")

# # cmd = f"colmap feature_extractor --database_path {base_scene_folder}/{scene}.db \
# #                                 --image_path {os.path.join(base_scene_folder, 'images')} \
# #                                 --ImageReader.single_camera 0 \
# #                                 --ImageReader.camera_model PINHOLE"

# # os.system(cmd)

# # load the database
# db = COLMAPDatabase.connect(f"{base_scene_folder}/{scene}.db")

# #%%
# cameras = db.execute("SELECT * FROM cameras ORDER BY camera_id")
# cameras = cameras.fetchall()
# # print(blob_to_array(cameras[0][4], dtype=np.float64))
# # cameras[0]
# for i in range(len(cameras)):
#     print(cameras[i])
# # %%
# # get column names
# cameras = db.execute("SELECT * FROM cameras")
# names = list(map(lambda x: x[0], cameras.description))
# names
# # %%
# new_image_names = []
# images = db.execute("SELECT * FROM images ORDER BY image_id")
# images = images.fetchall()
# for i in range(len(images)):
#     new_image_names.append(images[i][1])
# # %%
# # get column names
# images = db.execute("SELECT * FROM images ORDER BY image_id")
# names = list(map(lambda x: x[0], images.description))
# names
# # %%
# new_image_names = []
# for i in range(len(img_names)):
#     new_image_names.append(f"image{str(i).zfill(4)}.jpg")
# # %%
# new_image_names
# # %%
# # sort images by name
# images = sorted(images, key=lambda x: x[1])
# # %%
# new_image_names
# # %%
