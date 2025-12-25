def str_to_bool(value):
    """文字列をboolに変換する関数。'True'/'true'/'1'/'yes'などはTrue、それ以外はFalse"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 't')
    return bool(value)
