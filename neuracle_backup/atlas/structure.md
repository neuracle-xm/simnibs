# 项目结构说明

```text
atlas/
├─ atlas/                                               atlas 数据主目录
│  ├─ BN/                                               Brainnetome Atlas 相关文件
│  │  ├─ BN_abbreviation_mapping.csv                    BN 缩写、英文全称与中文全称对照表
│  │  ├─ BN_Atlas_246_1mm.nii.gz                        BN 246 区 1mm 离散图谱 NIfTI
│  │  ├─ BN_Atlas_246_labels_zh.csv                     BN 中文标签表，含统一输出字段
│  │  ├─ BN_Atlas_246_LUT.txt                           BN 原始 LUT 文本表，含编号、名称与颜色
│  │  ├─ BNA_subregions.xlsx                            BN 原始分区说明表
│  │  └─ MNI152_T1_1mm.nii.gz                           BN 配套使用的 1mm MNI 模板
│  ├─ DiFuMo/                                           DiFuMo 图谱、标签表与配套模板
│  │  ├─ DiFuMo_translation_mapping.csv                 DiFuMo 统一中英翻译映射表
│  │  ├─ tpl-MNI152NLin6Asym_res-02_T1w.nii.gz          DiFuMo 配套的原始 2mm MNI152NLin6Asym 模板
│  │  ├─ tpl-MNI152NLin6Asym_res-02_T1w_difumo-grid.nii.gz  重采样到 DiFuMo 网格的配套模板
│  │  ├─ DiFuMo_64/                                     64 成分 DiFuMo 数据
│  │  │  ├─ labels_64_dictionary.csv                    64 成分原始英文标签表
│  │  │  ├─ labels_64_dictionary_zh.csv                 64 成分中文标签表
│  │  │  ├─ 2mm/                                        64 成分 2mm 图谱目录
│  │  │  │  ├─ DiFuMo64.nii.gz                          64 成分 2mm 离散 winner-take-all 图谱
│  │  │  │  └─ maps.nii.gz                              64 成分 2mm 原始 4D 概率图
│  │  │  └─ 3mm/                                        64 成分 3mm 图谱目录
│  │  │     └─ maps.nii.gz                              64 成分 3mm 原始 4D 概率图
│  │  ├─ DiFuMo_128/                                    128 成分 DiFuMo 数据
│  │  │  ├─ labels_128_dictionary.csv                   128 成分原始英文标签表
│  │  │  ├─ labels_128_dictionary_zh.csv                128 成分中文标签表
│  │  │  ├─ 2mm/                                        128 成分 2mm 图谱目录
│  │  │  │  ├─ DiFuMo128.nii.gz                         128 成分 2mm 离散 winner-take-all 图谱
│  │  │  │  └─ maps.nii.gz                              128 成分 2mm 原始 4D 概率图
│  │  │  └─ 3mm/                                        128 成分 3mm 图谱目录
│  │  │     └─ maps.nii.gz                              128 成分 3mm 原始 4D 概率图
│  │  ├─ DiFuMo_256/                                    256 成分 DiFuMo 数据
│  │  │  ├─ labels_256_dictionary.csv                   256 成分原始英文标签表
│  │  │  ├─ labels_256_dictionary_zh.csv                256 成分中文标签表
│  │  │  ├─ 2mm/                                        256 成分 2mm 图谱目录
│  │  │  │  ├─ DiFuMo256.nii.gz                         256 成分 2mm 离散 winner-take-all 图谱
│  │  │  │  └─ maps.nii.gz                              256 成分 2mm 原始 4D 概率图
│  │  │  └─ 3mm/                                        256 成分 3mm 图谱目录
│  │  │     └─ maps.nii.gz                              256 成分 3mm 原始 4D 概率图
│  │  ├─ DiFuMo_512/                                    512 成分 DiFuMo 数据
│  │  │  ├─ labels_512_dictionary.csv                   512 成分原始英文标签表
│  │  │  ├─ labels_512_dictionary_zh.csv                512 成分中文标签表
│  │  │  ├─ 2mm/                                        512 成分 2mm 图谱目录
│  │  │  │  ├─ DiFuMo512.nii.gz                         512 成分 2mm 离散 winner-take-all 图谱
│  │  │  │  └─ maps.nii.gz                              512 成分 2mm 原始 4D 概率图
│  │  │  └─ 3mm/                                        512 成分 3mm 图谱目录
│  │  │     └─ maps.nii.gz                              512 成分 3mm 原始 4D 概率图
│  │  └─ DiFuMo_1024/                                   1024 成分 DiFuMo 数据
│  │     ├─ labels_1024_dictionary.csv                  1024 成分原始英文标签表
│  │     ├─ labels_1024_dictionary_zh.csv               1024 成分中文标签表
│  │     ├─ 2mm/                                        1024 成分 2mm 图谱目录
│  │     │  ├─ DiFuMo1024.nii.gz                        1024 成分 2mm 离散 winner-take-all 图谱
│  │     │  ├─ maps.nii.gz                              1024 成分 2mm 原始 4D 概率图
│  │     │  └─ resampled_maps.nii.gz                    1024 成分处理中间重采样文件
│  │     └─ 3mm/                                        1024 成分 3mm 图谱目录
│  │        └─ maps.nii.gz                              1024 成分 3mm 原始 4D 概率图
│  ├─ JulichBrainAtlas/                                 Julich 图谱、标签与模板
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv  Julich 双侧合并后的中文标签表
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz  Julich 双侧合并后的离散图谱
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.nii.gz         Julich 左半球原始图谱
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.xml            Julich 左半球原始标签 XML
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.nii.gz         Julich 右半球原始图谱
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.xml            Julich 右半球原始标签 XML
│  │  ├─ JulichBrainAtlas_translation_mapping.csv                   Julich 中英翻译映射表
│  │  └─ mni_icbm152_t1_tal_nlin_asym_09c.nii                       Julich 目录中的附带 MNI 模板
│  └─ 颜色查找表/                                      汇总后的颜色查找表目录
│     ├─ BN_Atlas_246_labels_zh.csv                    BN 颜色查找表副本
│     ├─ JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv  Julich 颜色查找表副本
│     ├─ labels_64_dictionary_zh.csv                   DiFuMo 64 成分颜色查找表副本
│     ├─ labels_128_dictionary_zh.csv                  DiFuMo 128 成分颜色查找表副本
│     ├─ labels_256_dictionary_zh.csv                  DiFuMo 256 成分颜色查找表副本
│     ├─ labels_512_dictionary_zh.csv                  DiFuMo 512 成分颜色查找表副本
│     └─ labels_1024_dictionary_zh.csv                 DiFuMo 1024 成分颜色查找表副本
├─ scripts/                                             项目处理脚本目录
│  ├─ build_bn_abbreviation_mapping.py                  生成 BN 缩写映射表
│  ├─ build_difumo_translation_mapping.py               生成 DiFuMo 翻译映射表
│  ├─ build_julich_translation_mapping.py               生成 Julich 翻译映射表
│  ├─ download_difumo_MNI_template.py                   下载并整理 DiFuMo 配套 MNI 模板
│  ├─ generate_label_tables.py                          批量生成中文标签表与颜色字段
│  └─ __pycache__/                                      Python 运行生成的缓存目录
├─ atlas_processing_plan.md                             atlas 整理实施方案文档
└─ structure.md                                         当前项目结构说明文档
```
