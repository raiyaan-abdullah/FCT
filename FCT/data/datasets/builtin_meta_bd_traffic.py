#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 02:56:39 2023

@author: rmedu
"""

VBTV_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "pedestrian"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "people"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bicycle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "car"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "van"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "truck"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "tricycle"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "awning-tricycle"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "bus"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "motor"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "others"},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "rickshaw"},
    {"color": [175, 116, 175],"isthing": 1,"id": 13,"name": "leguna",},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "cng"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "manual-van"},
 
]

# 1: "pedestrian",
#         2: "people",
#         3: "bicycle",
#         4: "car",
#         5: "van",
#         6: "truck",
#         7: "tricycle",
#         8: "awning-tricycle",
#         9: "bus",
#         10: "motor",
#         11: "others",
#         12: "rickshaw",
#         13: "leguna",
#         14: "cng",
#         15: "manual-van",


VBTV_BASE_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "pedestrian"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "people"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bicycle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "car"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "van"},

    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "tricycle"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "awning-tricycle"},

    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "motor"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "others"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "bus"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "truck"},
    {"color": [175, 116, 175],"isthing": 1,"id": 13,"name": "leguna",},
    
 
]

VBTV_NOVEL_CATEGORIES = [
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "rickshaw"},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "cng"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "manual-van"},
    

]

def _get_vbtv_fewshot_instances_meta():
    ret = {
        "thing_classes": VBTV_CATEGORIES,
        "novel_classes": VBTV_NOVEL_CATEGORIES,
        "base_classes": VBTV_BASE_CATEGORIES,
    }
    return ret

def _get_vbtv_instances_meta():
    thing_ids = [k["id"] for k in VBTV_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VBTV_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 15, len(thing_ids)
    # Mapping from the incontiguous VBTV category id to an id in [0, 14]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VBTV_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_vbtv_fewshot_instances_meta():
    ret = _get_vbtv_instances_meta()
    novel_ids = [k["id"] for k in VBTV_NOVEL_CATEGORIES if k["isthing"] == 1]
    novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
    novel_classes = [
        k["name"] for k in VBTV_NOVEL_CATEGORIES if k["isthing"] == 1
    ]
    base_categories = [
        k
        for k in VBTV_CATEGORIES
        if k["isthing"] == 1 and k["name"] not in novel_classes
    ]
    base_ids = [k["id"] for k in base_categories]
    base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
    base_classes = [k["name"] for k in base_categories]
    ret[
        "novel_dataset_id_to_contiguous_id"
    ] = novel_dataset_id_to_contiguous_id
    ret["novel_classes"] = novel_classes
    ret["base_dataset_id_to_contiguous_id"] = base_dataset_id_to_contiguous_id
    ret["base_classes"] = base_classes
    return ret

def _get_builtin_metadata_bd_traffic(dataset_name):
    if dataset_name == "vhbt":
        return _get_vbtv_instances_meta()
    elif dataset_name == "vhbt_fewshot":
        return _get_vbtv_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))