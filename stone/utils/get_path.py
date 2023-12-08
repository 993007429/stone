import os

import setting


def get_slice_dir(slice_key: str):
    return os.path.join(setting.DATA_DIR, slice_key)


def get_slice_path(slice_key: str, slice_name: str):
    return os.path.join(get_slice_dir(slice_key), slice_name)


def get_tile_dir(slice_key: str):
    return os.path.join(get_slice_dir(slice_key), 'tiles')


def get_tile_path(slice_key: str, x: int, y: int, z: int):
    return os.path.join(get_tile_dir(slice_key), f'{z}_{x}_{y}.jpeg')


def get_db_dir(slice_key: str, analysis_key: str):
    return os.path.join(get_slice_dir(slice_key), 'analyses', analysis_key)


def get_db_path(slice_key: str, analysis_key: str, ai_model: str, model_version: str):
    return os.path.join(get_slice_dir(slice_key), 'analyses', analysis_key, f'analysis_{ai_model}_{model_version}.db')


def get_roi_dir(slice_key: str, analysis_key: str):
    return os.path.join(get_slice_dir(slice_key), 'analyses', analysis_key, 'rois')
