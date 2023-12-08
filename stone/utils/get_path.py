import os

import setting


def get_dir_with_key(key: str):
    return os.path.join(setting.DATA_DIR, key)


def get_db_path(slice_key: str, analysis_key: str, ai_model: str, model_version: str):
    return os.path.join(get_dir_with_key(slice_key), 'analyses', analysis_key, f'analysis_{ai_model}_{model_version}.db')
