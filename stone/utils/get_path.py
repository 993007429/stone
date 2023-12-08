import os

import setting


def get_dir_with_key(key: str):
    return os.path.join(setting.DATA_DIR, key)
