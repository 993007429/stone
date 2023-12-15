from .value_objects import AIType

AI_MARK_TABLE_START_ID = 100000000

# 不同算法对应的人工mark表, 值为表名的后缀
AI_TYPE_MANUAL_MARK_TABLE_MAPPING = {
    AIType.tct1: 'human_tl',
    AIType.tct2: 'human_tl',
    AIType.lct1: 'human_tl',
    AIType.lct2: 'human_tl',
    AIType.tbs_dna: 'human_tl',
    AIType.human_tl: 'human_tl',
    AIType.human_bm: 'human_bm',
    AIType.bm: 'human_bm'
}

DEFAULT_MPP = 0.242042

HUMAN_TL_CELL_TYPES = [
    'HSIL', 'ASC-H', 'LSIL', 'ASC-US', 'AGC', '阴性', '无', '异物',
    '不确定', '滴虫', '霉菌', '线索', '放线菌', '疱疹', 'AI假阳', 'AI假阴', '炎性细胞', '乳酸杆菌'
]
