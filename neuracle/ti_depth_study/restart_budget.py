"""固定电流与有效评价预算的 TI 搜索，仅供离线研究。"""

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from neuracle.parameters.schemas import AtlasParam, ROIParam
from neuracle.ti_depth_study.baseline import PHYSICS, content_id, inverse_config
from neuracle.ti_leadfield_optimization.fitness import LeadfieldFitnessEvaluator
from neuracle.ti_leadfield_optimization.leadfield import ensure_leadfield, load_leadfield
from neuracle.ti_leadfield_optimization.models import GASettings, LeadfieldConfig
from neuracle.ti_leadfield_optimization.optimizer import _GAObjective, _current_tick_bounds, _require_genetic_algorithm
from neuracle.ti_leadfield_optimization.regions import prepare_region_masks
from neuracle.ti_optimization import setup_electrodes_and_roi, setup_goal
from neuracle.utils import cond_dict_to_list, find_montage_file
from neuracle.utils.constants import N_WORKERS
from neuracle.utils.find_nifty import find_optional_nifti_file
from neuracle.utils.optimization_roi import build_optimization_rois, resolve_roi_settings
from simnibs import opt_struct


FloatArray = NDArray[np.float64]
METHODS = ("leadfield_based", "leadfield_free")
CURRENT_MA = 1.0
THRESHOLD = 0.10
BASED_SETTINGS = GASettings(current_min_ma=CURRENT_MA, current_max_ma=CURRENT_MA,
                            non_roi_threshold_v_per_m=THRESHOLD, roi_threshold_v_per_m=THRESHOLD)
BASED_INITIAL_REQUESTS = BASED_SETTINGS.population_size
FREE_N_DIMENSIONS = 8
FREE_POPSIZE = 13
FREE_INITIAL_REQUESTS = FREE_N_DIMENSIONS * FREE_POPSIZE
INITIAL_REQUESTS = dict(zip(METHODS, (BASED_INITIAL_REQUESTS, FREE_INITIAL_REQUESTS)))


class BudgetReached(Exception):
    """优化器请求额外评价时结束本次预算内搜索。"""


class CachedFreeObjective(Exception):
    """当前有效电极节点与之前已求解的右侧项相同。"""


@dataclass
class Trace:
    """按独立起点记录每次请求，并允许同一臂共享已评价配置。"""

    budget: int | None
    minimum_requests: int
    scores: dict[str, float] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    new_evals: int = 0
    cache_hits: int = 0
    invalid: int = 0
    best_score: float = np.inf
    best_parameters: list[float] | None = None

    def check_budget(self) -> None:
        """请求下一次评价前检查预算，允许优化器完整初始化。"""
        if self.budget is not None and self.new_evals >= self.budget:
            if len(self.rows) < self.minimum_requests:
                raise ValueError("研究预算不足以完成初始化")
            raise BudgetReached

    def record(self, parameters: FloatArray, key: str | None, score: float, start: float) -> float:
        """保存一个有效新评价、缓存命中或无效请求及当前最优目标。

        Parameters
        ----------
        parameters : ndarray
            入参向量的原始值，可复现原布局。
        key : str or None
            有效布局的规范键。None 表示不可行。
        score, start : float
            搜索器的最小化值与本次请求起始时钟。

        Returns
        -------
        float
            未调整的搜索值或命中值，传回原搜索器。
        """
        if not np.isfinite(score):
            raise ValueError("本次搜索返回非有限目标")
        if key is None:
            state = "invalid"
            self.invalid += 1
        elif key in self.scores:
            state = "cache_hit"
            self.cache_hits += 1
        else:
            state = "new"
            self.scores[key] = score
            self.new_evals += 1
        if key is not None and score < self.best_score:
            self.best_score = score
            self.best_parameters = parameters.tolist()
        self.rows.append(dict(request=len(self.rows) + 1, status=state, configuration_id=key or "NA",
                              n_valid_unique=self.new_evals, arm_n_valid_unique=len(self.scores),
                              objective=score, best_objective=self.best_score if self.best_parameters is not None else "NA",
                              parameters=json.dumps(parameters.tolist()), wall_s=perf_counter() - start))
        return score


class FreeOptimization(opt_struct.TesFlexOptimization):
    """在在线 FEM 前识别同一求解配置，复用原生几何、场与 ROC0。"""

    trace: Trace | None = None
    configuration_id: str | None = None

    def get_nodes_electrode(self, electrode_pos: list[list[FloatArray]], plot: bool = False) -> Any:
        """保留原生可行域与电流分配，并在评价时识别相同节点激励。

        Parameters
        ----------
        electrode_pos : list of lists
            各载波阵列的原生椭球参数。
        plot : bool
            保留原生节点导出开关。

        Returns
        -------
        list or tuple
            原生节点索引或几何无效标记。
        """
        nodes = super().get_nodes_electrode(electrode_pos, plot)
        if self.trace is None or isinstance(nodes[0], str):
            return nodes
        channels = []
        for layout in self.electrode:
            electrodes = []
            for array in layout._electrode_arrays:
                for electrode in array.electrodes:
                    order = np.argsort(electrode.node_idx)
                    indices = np.asarray(electrode.node_idx[order], dtype="<i8")
                    currents = np.asarray(np.abs(electrode.node_current[order]), dtype="<f8")
                    electrodes.append(sha256(indices.tobytes() + currents.tobytes()).hexdigest())
            # 固定等幅正负极及 max_TI 目标下，换极和换载波不改变适应度。
            channels.append(sorted(electrodes))
        self.configuration_id = content_id({"channels": sorted(channels)})
        if self.configuration_id in self.trace.scores:
            raise CachedFreeObjective
        return nodes

    def traced_objective(self, parameters: FloatArray) -> float:
        """通过原生目标计算一次请求，并按节点激励区分命中和新评价。"""
        trace = self.trace
        if trace is None:
            raise RuntimeError("free 搜索缺少评价记录器")
        trace.check_budget()
        start = perf_counter()
        self.configuration_id = None
        previous = self.n_sim
        try:
            score = self.goal_fun(parameters)
        except CachedFreeObjective:
            score = trace.scores[self.configuration_id]
        else:
            if (self.n_sim != previous) != (self.configuration_id is not None):
                raise RuntimeError("free 几何判定与 FEM 原生评价计数不一致")
        return trace.record(parameters, self.configuration_id, float(score), start)


def prepare_case(study: Path, subject: str, target: dict[str, str], method: str, output: Path) -> dict[str, Any]:
    """解析 02 协议靶区及头模，准备可以跨各起点复用的设置。

    Parameters
    ----------
    study, subject, target : Path, str and dict
        研究根、开发头模 ID 及其协议靶区。
    method, output : str and Path
        研究路线和本病例缓存或求解目录。

    Returns
    -------
    dict
        头模、帽位、mesh、ROI 设置和 based 适应度计算器。
    """
    head = study / "head_models" / f"m2m_{subject}"
    mesh = head / f"{subject}.msh"
    name = PHYSICS["montage"]
    cap = Path(find_montage_file(str(head), name))
    config = inverse_config(target, method)
    roi = resolve_roi_settings(config["roi_type"], ROIParam(atlas_param=AtlasParam(**config["roi_param"]["atlas_param"])))
    tensor = find_optional_nifti_file(head, ("DTI_coregT1_tensor.nii.gz", "DTI_coregT1_tensor.nii"))
    prepared = dict(head=head, mesh=mesh, cap=cap, roi=roi)
    if method == "leadfield_based":
        settings = LeadfieldConfig(head_model_dir=head, mesh_path=mesh, montage_path=cap,
                                   leadfield_dir=study / "leadfields" / subject,
                                   conductivity_config=PHYSICS["conductivity_config"], anisotropy_type="scalar",
                                   n_workers=N_WORKERS, dti_path=Path(tensor) if tensor is not None else None)
        leadfield = load_leadfield(ensure_leadfield(settings, output))
        regions = build_optimization_rois(str(head), leadfield.mesh, **roi)
        prepared["evaluator"] = LeadfieldFitnessEvaluator(leadfield, prepare_region_masks(leadfield.mesh, regions),
                                                         THRESHOLD, THRESHOLD)
    return prepared


def search_based(prepared: dict[str, Any], trace: Trace, seed: int) -> list[str]:
    """用既有 GA 与 leadfield ROC0 搜索，并从预算内记录选出具名布局。

    Parameters
    ----------
    prepared, trace, seed : dict, Trace and int
        固定 leadfield 和 ROI、评价预算记录器及预登记种子。

    Returns
    -------
    list of str
        评价中最优的帽位，顺序为 A+、A−、B+、B−。
    """
    evaluator = prepared["evaluator"]
    settings = replace(BASED_SETTINGS, random_seed=seed)
    if trace.budget is not None:
        settings = replace(settings, max_num_iteration=1000, max_iteration_without_improv=None)
    algorithm = {name: value for name, value in asdict(settings).items() if name in (
        "max_num_iteration", "population_size", "mutation_probability", "elit_ratio", "crossover_probability",
        "parents_portion", "crossover_type", "max_iteration_without_improv")}
    electrodes = evaluator.leadfield.electrode_names
    tick = list(_current_tick_bounds(settings))
    bounds = np.asarray([[0, len(electrodes) - 1]] * 4 + [tick, tick], dtype=np.int64)
    goal = _GAObjective(evaluator, settings)

    def objective(parameters: FloatArray) -> float:
        """在原生整数染色体评价外记录有效配置及缓存命中。"""
        trace.check_budget()
        start = perf_counter()
        genes = tuple(int(value) for value in np.rint(parameters))
        key = content_id({"genes": goal._canonical_key(genes)}) if len(set(genes[:4])) == 4 else None
        score = trace.scores[key] if key in trace.scores else goal(parameters)
        return trace.record(parameters, key, score, start)

    random_state = np.random.get_state()
    np.random.seed(seed)
    try:
        runner = _require_genetic_algorithm()(function=objective, dimension=6, variable_type="int",
                                              variable_boundaries=bounds, algorithm_parameters=algorithm,
                                              function_timeout=settings.function_timeout_seconds,
                                              convergence_curve=False, progress_bar=False)
        runner.run()
    except BudgetReached:
        pass
    finally:
        np.random.set_state(random_state)
    if trace.best_parameters is None:
        raise RuntimeError("based 搜索无有效布局")
    return [str(electrodes[round(index)]) for index in trace.best_parameters[:4]]


def search_free(prepared: dict[str, Any], trace: Trace, seed: int, output: Path) -> list[str]:
    """用既有连续差分进化设置与 OnlineFEM 选出并映射预算内最优布局。

    Parameters
    ----------
    prepared, trace, seed, output : dict, Trace, int and Path
        固定头模及 ROI、评价预算记录器、种子、映射结果目录。

    Returns
    -------
    list of str
        使用原生唯一指派映射得到的四个帽位。
    """
    opt = FreeOptimization()
    opt.subpath = str(prepared["head"])
    opt.fn_mesh = str(prepared["mesh"])
    opt.output_folder = str(output)
    opt.open_in_gmsh = False
    opt.anisotropy_type = PHYSICS["anisotropy_type"]
    opt.cond = cond_dict_to_list(PHYSICS["conductivity_config"])
    opt.seed = seed
    opt._n_cpu = N_WORKERS
    setup_goal(opt, goal="focality", focality_threshold=[THRESHOLD, THRESHOLD], net_electrode_file=str(prepared["cap"]))
    pair = [CURRENT_MA / 1000, -CURRENT_MA / 1000]
    setup_electrodes_and_roi(opt, goal="focality", mesh_file_path=str(prepared["mesh"]),
                             electrode_current1=pair, electrode_current2=pair, electrode_radius=[PHYSICS["electrode_radius"]],
                             **prepared["roi"])
    random_state = np.random.get_state()
    try:
        opt._prepare()
    finally:
        np.random.set_state(random_state)
    options = opt._optimizer_options_std
    if len(options["init_vals"]) != FREE_N_DIMENSIONS or options["popsize"] != FREE_POPSIZE:
        raise RuntimeError("free 原生初始化种群与已登记预算不一致")
    if any(layout.dirichlet_correction or layout._current_estimator is not None or layout.current_outlier_correction
           for layout in opt.electrode):
        raise ValueError("节点激励去重仅支持无电流修正及估计的固定电流搜索")
    opt.trace = trace
    try:
        differential_evolution(opt.traced_objective, x0=options["init_vals"], strategy="best1bin",
                               recombination=options["recombination"], mutation=tuple(options["mutation"]),
                               tol=options["tol"] if trace.budget is None else 0.0,
                               atol=0.0 if trace.budget is None else -1.0, maxiter=options["maxiter"],
                               popsize=options["popsize"], bounds=options["bounds"], disp=False, polish=False, seed=seed)
    except BudgetReached:
        pass
    finally:
        opt.trace = None
    if opt.n_sim != trace.new_evals:
        raise RuntimeError("free 原生 FEM 计数与记录不一致")
    if trace.best_parameters is None:
        raise RuntimeError("free 搜索无有效布局")
    best = opt.get_electrode_pos_from_array(np.array(trace.best_parameters))
    opt.get_nodes_electrode(best)
    return opt.map_to_nearest_net_electrodes(str(prepared["cap"]))["mapped_labels"]
