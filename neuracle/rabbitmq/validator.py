"""
RabbitMQ 消息参数验证器

在接收消息时进行参数合法性检查，确保满足以下条件：
1. 必填字段存在
2. 字段类型正确
3. 值范围有效
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """验证失败异常，仅用于 raise 和 try...except 捕获"""

    pass


def validate_model_params(params: dict[str, Any]) -> None:
    """
    验证头模生成参数

    验证规则：
    - T1_file_path: 非空字符串
    - dir_path: 非空字符串
    - T2_file_path: 可选
    - DTI_file_path: 可选

    Note:
        id 在消息顶层验证，params 内不验证 id

    Raises:
        ValidationError: 验证失败时抛出
    """
    # 验证 T1_file_path
    t1_path = params.get("T1_file_path")
    if not t1_path or not isinstance(t1_path, str):
        logger.error("[T1_file_path] T1_file_path 必须是非空字符串")
        raise ValidationError("T1_file_path 验证失败")

    # 验证 dir_path
    dir_path = params.get("dir_path")
    if not dir_path or not isinstance(dir_path, str):
        logger.error("[dir_path] dir_path 必须是字符串")
        raise ValidationError("dir_path 验证失败")

    # 验证 T2_file_path（可选）
    t2_path = params.get("T2_file_path")
    if t2_path is not None and not isinstance(t2_path, str):
        logger.error("[T2_file_path] T2_file_path 必须是字符串")
        raise ValidationError("T2_file_path 必须是字符串")

    # 验证 DTI_file_path（可选）
    dti_path = params.get("DTI_file_path")
    if dti_path is not None and not isinstance(dti_path, str):
        logger.error("[DTI_file_path] DTI_file_path 必须是字符串")
        raise ValidationError("DTI_file_path 必须是字符串")


def validate_forward_params(params: dict[str, Any]) -> None:
    """
    验证正向仿真参数

    验证规则：
    - dir_path: 非空字符串
    - msh_file_path: 非空字符串
    - montage: 非空字符串
    - electrode_A: 非空列表，元素为字符串
    - electrode_B: 非空列表，元素为字符串
    - current_A: 非空列表，元素为浮点数，总和必须为 0
    - current_B: 非空列表，元素为浮点数，总和必须为 0
    - electrode_A 和 current_A 长度必须一致
    - electrode_B 和 current_B 长度必须一致
    - cond: 非空字典，值为浮点数
    - anisotropy: 布尔值
    - DTI_file_path: 可选

    Note:
        id 在消息顶层验证，params 内不验证 id

    Raises:
        ValidationError: 验证失败时抛出
    """
    # 验证 dir_path
    dir_path = params.get("dir_path")
    if not dir_path or not isinstance(dir_path, str):
        logger.error("[dir_path] dir_path 必须是字符串")
        raise ValidationError("dir_path 验证失败")

    # 验证 msh_file_path
    msh_path = params.get("msh_file_path")
    if not msh_path or not isinstance(msh_path, str):
        logger.error("[msh_file_path] msh_file_path 必须是字符串")
        raise ValidationError("msh_file_path 验证失败")

    # 验证 montage
    montage = params.get("montage")
    if not montage or not isinstance(montage, str):
        logger.error("[montage] montage 必须是字符串")
        raise ValidationError("montage 验证失败")

    # 验证 electrode_A
    electrode_A = params.get("electrode_A")
    if not electrode_A or not isinstance(electrode_A, list):
        logger.error("[electrode_A] electrode_A 必须是列表")
        raise ValidationError("electrode_A 验证失败")
    if not all(isinstance(e, str) for e in electrode_A):
        logger.error("[electrode_A] electrode_A 元素必须是字符串")
        raise ValidationError("electrode_A 元素必须是字符串")

    # 验证 electrode_B
    electrode_B = params.get("electrode_B")
    if not electrode_B or not isinstance(electrode_B, list):
        logger.error("[electrode_B] electrode_B 必须是列表")
        raise ValidationError("electrode_B 验证失败")
    if not all(isinstance(e, str) for e in electrode_B):
        logger.error("[electrode_B] electrode_B 元素必须是字符串")
        raise ValidationError("electrode_B 元素必须是字符串")

    # 验证 current_A
    current_A = params.get("current_A")
    if not current_A or not isinstance(current_A, list):
        logger.error("[current_A] current_A 必须是列表")
        raise ValidationError("current_A 验证失败")
    if not all(isinstance(c, (int, float)) for c in current_A):
        logger.error("[current_A] current_A 元素必须是数字")
        raise ValidationError("current_A 元素必须是数字")

    # 验证 current_B
    current_B = params.get("current_B")
    if not current_B or not isinstance(current_B, list):
        logger.error("[current_B] current_B 必须是列表")
        raise ValidationError("current_B 验证失败")
    if not all(isinstance(c, (int, float)) for c in current_B):
        logger.error("[current_B] current_B 元素必须是数字")
        raise ValidationError("current_B 元素必须是数字")

    # 验证 electrode_A 和 current_A 长度一致
    if electrode_A and current_A and len(electrode_A) != len(current_A):
        logger.error(
            f"[electrode_A/current_A] 长度不一致: {len(electrode_A)} vs {len(current_A)}"
        )
        raise ValidationError("electrode_A/current_A 长度不一致")

    # 验证 electrode_B 和 current_B 长度一致
    if electrode_B and current_B and len(electrode_B) != len(current_B):
        logger.error(
            f"[electrode_B/current_B] 长度不一致: {len(electrode_B)} vs {len(current_B)}"
        )
        raise ValidationError("electrode_B/current_B 长度不一致")

    # 验证 current_A 总和为 0
    if current_A and abs(sum(current_A)) > 1e-6:
        logger.error(f"[current_A] 电流总和必须为 0，当前为: {sum(current_A)}")
        raise ValidationError("current_A 电流总和必须为 0")

    # 验证 current_B 总和为 0
    if current_B and abs(sum(current_B)) > 1e-6:
        logger.error(f"[current_B] 电流总和必须为 0，当前为: {sum(current_B)}")
        raise ValidationError("current_B 电流总和必须为 0")

    # 验证 cond
    cond = params.get("cond")
    if not cond or not isinstance(cond, dict):
        logger.error("[cond] cond 必须是字典")
        raise ValidationError("cond 验证失败")
    if not all(isinstance(v, (int, float)) for v in cond.values()):
        logger.error("[cond] cond 值必须是数字")
        raise ValidationError("cond 值必须是数字")

    # 验证 anisotropy
    anisotropy = params.get("anisotropy")
    if not isinstance(anisotropy, bool):
        logger.error("[anisotropy] anisotropy 必须是布尔值")
        raise ValidationError("anisotropy 必须是布尔值")

    # 验证 DTI_file_path（可选）
    dti_path = params.get("DTI_file_path")
    if dti_path is not None and not isinstance(dti_path, str):
        logger.error("[DTI_file_path] DTI_file_path 必须是字符串")
        raise ValidationError("DTI_file_path 必须是字符串")


def validate_inverse_params(params: dict[str, Any]) -> None:
    """
    验证逆向仿真参数

    验证规则：
    - dir_path: 非空字符串
    - msh_file_path: 非空字符串
    - montage: 非空字符串
    - current_A: 非空列表，元素为浮点数，总和必须为 0
    - current_B: 非空列表，元素为浮点数，总和必须为 0
    - cond: 非空字典，值为浮点数
    - anisotropy: 布尔值
    - roi_type: 字符串，必须为 "atlas" 或 "mni_pos"
    - roi_param: 字典，根据 roi_type 验证对应参数
    - roi_type=atlas 时，atlas_param 必填（name, area）
    - roi_type=mni_pos 时，mni_param 必填（center, radius）
    - target_threshold: 浮点数，>= 0
    - DTI_file_path: 可选

    Note:
        逆向仿真与正向仿真不同，没有 electrode_A 和 electrode_B 参数

    Raises:
        ValidationError: 验证失败时抛出
    """
    # 验证 dir_path
    dir_path = params.get("dir_path")
    if not dir_path or not isinstance(dir_path, str):
        logger.error("[dir_path] dir_path 必须是字符串")
        raise ValidationError("dir_path 验证失败")

    # 验证 msh_file_path
    msh_path = params.get("msh_file_path")
    if not msh_path or not isinstance(msh_path, str):
        logger.error("[msh_file_path] msh_file_path 必须是字符串")
        raise ValidationError("msh_file_path 验证失败")

    # 验证 montage
    montage = params.get("montage")
    if not montage or not isinstance(montage, str):
        logger.error("[montage] montage 必须是字符串")
        raise ValidationError("montage 验证失败")

    # 验证 current_A
    current_A = params.get("current_A")
    if not current_A or not isinstance(current_A, list):
        logger.error("[current_A] current_A 必须是列表")
        raise ValidationError("current_A 验证失败")
    if not all(isinstance(c, (int, float)) for c in current_A):
        logger.error("[current_A] current_A 元素必须是数字")
        raise ValidationError("current_A 元素必须是数字")

    # 验证 current_B
    current_B = params.get("current_B")
    if not current_B or not isinstance(current_B, list):
        logger.error("[current_B] current_B 必须是列表")
        raise ValidationError("current_B 验证失败")
    if not all(isinstance(c, (int, float)) for c in current_B):
        logger.error("[current_B] current_B 元素必须是数字")
        raise ValidationError("current_B 元素必须是数字")

    # 验证 current_A 总和为 0
    if current_A and abs(sum(current_A)) > 1e-6:
        logger.error(f"[current_A] 电流总和必须为 0，当前为: {sum(current_A)}")
        raise ValidationError("current_A 电流总和必须为 0")

    # 验证 current_B 总和为 0
    if current_B and abs(sum(current_B)) > 1e-6:
        logger.error(f"[current_B] 电流总和必须为 0，当前为: {sum(current_B)}")
        raise ValidationError("current_B 电流总和必须为 0")

    # 验证 cond
    cond = params.get("cond")
    if not cond or not isinstance(cond, dict):
        logger.error("[cond] cond 必须是字典")
        raise ValidationError("cond 验证失败")
    if not all(isinstance(v, (int, float)) for v in cond.values()):
        logger.error("[cond] cond 值必须是数字")
        raise ValidationError("cond 值必须是数字")

    # 验证 anisotropy
    anisotropy = params.get("anisotropy")
    if not isinstance(anisotropy, bool):
        logger.error("[anisotropy] anisotropy 必须是布尔值")
        raise ValidationError("anisotropy 必须是布尔值")

    # 验证 roi_type
    roi_type = params.get("roi_type")
    if not roi_type or roi_type not in ("atlas", "mni_pos"):
        logger.error('[roi_type] roi_type 必须是 "atlas" 或 "mni_pos"')
        raise ValidationError('roi_type 必须是 "atlas" 或 "mni_pos"')

    # 验证 roi_param
    roi_param = params.get("roi_param", {})
    if not isinstance(roi_param, dict):
        logger.error("[roi_param] roi_param 必须是字典")
        raise ValidationError("roi_param 必须是字典")

    if roi_type == "atlas":
        atlas_param = roi_param.get("atlas_param")
        if not atlas_param or not isinstance(atlas_param, dict):
            logger.error("[roi_param.atlas_param] atlas_param 必须是字典")
            raise ValidationError("atlas_param 必须是字典")
        if not atlas_param.get("name"):
            logger.error("[roi_param.atlas_param.name] name 不能为空")
            raise ValidationError("atlas_param.name 不能为空")
        if not atlas_param.get("area"):
            logger.error("[roi_param.atlas_param.area] area 不能为空")
            raise ValidationError("atlas_param.area 不能为空")

    elif roi_type == "mni_pos":
        mni_param = roi_param.get("mni_param")
        if not mni_param or not isinstance(mni_param, dict):
            logger.error("[roi_param.mni_param] mni_param 必须是字典")
            raise ValidationError("mni_param 必须是字典")
        center = mni_param.get("center")
        if not center or not isinstance(center, list) or len(center) != 3:
            logger.error(
                "[roi_param.mni_param.center] center 必须是 [x, y, z] 格式的列表"
            )
            raise ValidationError("center 必须是 [x, y, z] 格式的列表")
        if not all(isinstance(c, (int, float)) for c in center):
            logger.error("[roi_param.mni_param.center] center 元素必须是数字")
            raise ValidationError("center 元素必须是数字")
        radius = mni_param.get("radius")
        if not isinstance(radius, (int, float)) or radius <= 0:
            logger.error("[roi_param.mni_param.radius] radius 必须是正数")
            raise ValidationError("radius 必须是正数")

    # 验证 target_threshold
    threshold = params.get("target_threshold")
    if not isinstance(threshold, (int, float)):
        logger.error("[target_threshold] target_threshold 必须是数字")
        raise ValidationError("target_threshold 必须是数字")
    if threshold < 0:
        logger.error("[target_threshold] target_threshold 必须 >= 0")
        raise ValidationError("target_threshold 必须 >= 0")

    # 验证 DTI_file_path（可选）
    dti_path = params.get("DTI_file_path")
    if dti_path is not None and not isinstance(dti_path, str):
        logger.error("[DTI_file_path] DTI_file_path 必须是字符串")
        raise ValidationError("DTI_file_path 必须是字符串")


def validate_ack_test_params(params: dict[str, Any]) -> None:
    """
    验证 ack 时机测试参数

    验证规则：
    - sleep_seconds: 数字且 > 0
    """
    sleep_seconds = params.get("sleep_seconds", 30.0)
    if not isinstance(sleep_seconds, (int, float)):
        logger.error("[sleep_seconds] sleep_seconds 必须是数字")
        raise ValidationError("sleep_seconds 必须是数字")
    if sleep_seconds <= 0:
        logger.error("[sleep_seconds] sleep_seconds 必须大于 0")
        raise ValidationError("sleep_seconds 必须大于 0")


def validate_message(message: dict[str, Any]) -> None:
    """
    验证消息完整性

    验证规则：
    - message 必须是字典
    - id: 非空字符串
    - type: 非空字符串，必须为 "model", "forward", "inverse" 或 "ack_test"
    - params: 字典，根据 type 验证对应参数

    Args:
        message: 消息字典

    Raises:
        ValidationError: 验证失败时抛出
    """
    if not isinstance(message, dict):
        logger.error("[message] 消息必须是字典")
        raise ValidationError("消息必须是字典")

    # 验证 id
    msg_id = message.get("id")
    if not msg_id or not isinstance(msg_id, str):
        logger.error("[id] id 必须是字符串")
        raise ValidationError("id 必须是字符串")

    # 验证 type
    msg_type = message.get("type")
    if not msg_type or msg_type not in ("model", "forward", "inverse", "ack_test"):
        logger.error(
            '[type] type 必须是 "model", "forward", "inverse" 或 "ack_test"'
        )
        raise ValidationError(
            'type 必须是 "model", "forward", "inverse" 或 "ack_test"'
        )

    # 验证 params
    params = message.get("params")
    if not params or not isinstance(params, dict):
        logger.error("[params] params 必须是字典")
        raise ValidationError("params 必须是字典")

    # 根据 type 验证 params
    if msg_type == "model":
        validate_model_params(params)
    elif msg_type == "forward":
        validate_forward_params(params)
    elif msg_type == "inverse":
        validate_inverse_params(params)
    elif msg_type == "ack_test":
        validate_ack_test_params(params)
