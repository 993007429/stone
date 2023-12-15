import enum

from stone.seedwork.domain.value_objects import BaseEnum


class AnalysisStat(BaseEnum):
    success = 1  # 已处理
    failed = 2  # 处理异常


@enum.unique
class AIFolder(BaseEnum):
    tct1 = 'TCTAnalysis_v2_2'
    tct2 = 'TCTAnalysis_v3_1'
    bm = 'BM'
    dna1 = 'DNA1'
    dna2 = 'DNA2'
    fish = 'FISH_deployment'
    her2 = 'Her2New_'
    ki67 = 'Ki67'
    ki67_new = 'Ki67New'
    np = 'np'
    pdl1 = 'PDL1'


@enum.unique
class AIModel(BaseEnum):
    tct1 = 'tct1'
    tct2 = 'tct2'
    lct1 = 'lct1'
    lct2 = 'lct2'
    tbs_dna = 'tbs_dna'
    dna_ploidy = 'dna_ploidy'
    her2 = 'her2'
    ki67 = 'ki67'
    pdl1 = 'pdl1'
    np = 'np'
    er = 'er'
    pr = 'pr'
    bm = 'bm'
    cd30 = 'cd30'
    ki67hot = 'ki67hot'
    celldet = 'celldet'
    cellseg = 'cellseg'
    fish_tissue = 'fishTissue'
    model_calibrate_tct = 'model_calibrate_tct'
    model_calibrate_lct = 'model_calibrate_lct'
