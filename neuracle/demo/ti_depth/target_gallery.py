"""发布按头模和指定靶点查看的三视图与结果清单。"""

from collections import defaultdict
import csv
from datetime import datetime
import json
from math import isclose
from pathlib import Path
from typing import Any


SCREENING = {f"{code:02d}" for code in range(4, 13)} | {"20"}
SEARCH = {"15", "16"}
DECISIONS = {"improved": "通过改善门槛", "maintained": "保持", "failed": "未通过", "indeterminate": "不可判定"}
COMPARISON_16 = Path("results/16/run-002/comparison")


def read_rows(path: Path) -> list[dict[str, str]]:
    """读取带字段名的研究结果表。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def selected_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留指定靶点的选中布局、同靶参照和各项诊断场。"""
    selected = []
    for entry in entries:
        code = entry["experiment"]
        if code == "pool" or (code == "04" and entry["title"].startswith("定位线索")):
            continue
        if code == "20" and not entry["title"].endswith("shift00 · 排名 1"):
            continue
        group = {"03": "baseline"}.get(code, "screening" if code in SCREENING else "search" if code in SEARCH else "diagnostic")
        selected.append(entry | {"group": group, "id": len(selected), "pairs": []})
    return selected


def add_pair(
    candidate: dict[str, Any], reference: dict[str, Any], raw: dict[str, str], path: Path, root: Path, label: str,
) -> None:
    """登记原表配对，核对两端图像与原表的目标及指标相同。"""
    if (candidate["subject"], candidate["target"]) != (reference["subject"], reference["target"]):
        raise ValueError(f"配对跨越头模或靶区 {path}")
    for entry, prefix in ((candidate, ""), (reference, "reference_")):
        for name, column in (("distance", "D_hot_mm"), ("coverage", "Cov010"), ("spill", "Vout010")):
            source_value = raw.get(prefix + column, "NA")
            if source_value in ("", "NA"):
                continue
            expected = float(source_value)
            if entry[name] is None:
                entry[name] = expected
            elif not isclose(entry[name], expected, abs_tol=1e-7, rel_tol=1e-8):
                raise ValueError(f"图项指标与配对来源不一致 {path} {entry['title']} {column}")
    candidate["pairs"].append({"reference_id": reference["id"], "reference_title": reference["title"],
                              "label": label, "table": path.relative_to(root).as_posix(),
                              "delta_distance": candidate["distance"] - reference["distance"],
                              "delta_coverage_pp": 100 * (candidate["coverage"] - reference["coverage"]),
                              "delta_spill_cm3": (candidate["spill"] - reference["spill"]) / 1000,
                              "decision": DECISIONS.get(raw["decision"], raw["decision"]),
                              "reason": "" if raw["reason"] == "NA" else raw["reason"]})


def attach_pairs(selected: list[dict[str, Any]], root: Path, arms: dict[str, str]) -> None:
    """用原配对表给各靶的首选和搜索胜者添加同条件参照。"""
    by_case = defaultdict(list)
    for entry in selected:
        by_case[entry["experiment"], entry["subject"], entry["target"]].append(entry)
    for code in sorted(SCREENING):
        path = root / "results" / code / "run-001/paired.csv"
        for raw in read_rows(path):
            group = by_case[code, raw["subject"], raw["target"]]
            matched = []
            for prefix in ("", "reference_"):
                name = raw[prefix + "arm"]
                title = arms.get(name, name)
                if code == "20":
                    title += " · shift00 · 排名 1"
                token = raw[prefix + "case_id"].rsplit(":", 1)[-1]
                entry, = (item for item in group if item["title"] == title and item["token"] == token)
                matched.append(entry)
            matched[1]["reference"] = True
            add_pair(matched[0], matched[1], raw, path, root, "固定候选库配对")
    path = root / "results/15/run-001/paired.csv"
    for raw in read_rows(path):
        group = by_case["15", raw["subject"], raw["target"]]
        prefix = f"{arms[raw['method']]} · 种子组 {raw['seed_group']} · "
        candidate, = (item for item in group if item["title"] == prefix + arms[raw["arm"]])
        reference, = (item for item in group if item["title"] == prefix + arms[raw["reference_arm"]])
        add_pair(candidate, reference, raw, path, root, "同路线、同种子组")
    path = root / COMPARISON_16 / "paired.csv"
    if not path.is_file():
        return
    for raw in read_rows(path):
        key = raw["case_id"].split("_", 2)
        group = by_case["16", key[0], key[1]]
        method = arms[raw["receiver"]]
        candidate_title = f"{method} · {arms[raw['arm']]}"
        reference_title = f"{method} · {arms['random']}"
        label = "同有效评价预算"
        if raw["comparison"] == "from_zero_time_boundary":
            reference_title = candidate_title + " · 从零计时对照"
            label = "从零计时对照"
        elif raw["comparison"] != "cached_donor":
            raise ValueError(f"未登记的 16 配对类型 {raw['comparison']}")
        candidate = next((item for item in group if item["title"] == candidate_title), None)
        reference = next((item for item in group if item["title"] == reference_title), None)
        if candidate and reference and raw["D_hot_mm"] not in ("", "NA") and raw["reference_D_hot_mm"] not in ("", "NA"):
            add_pair(candidate, reference, raw, path, root, label)


def comparison_status(root: Path, arms: dict[str, str]) -> dict[str, Any]:
    """汇总 16 的运行状态与缺少场图的失败臂。"""
    folder = root / COMPARISON_16
    run = json.loads((folder / "run.json").read_text(encoding="utf-8-sig"))
    failures = []
    for row in read_rows(folder / "warm_start.csv"):
        if row["status"] != "failed":
            continue
        name = "未修复跨路供体" if row["arm"] == "cross" else arms[row["arm"]]
        reason = row["reason"]
        if "partly overlaps with invalid skin region" in reason:
            reason = "电极部分落在有效头皮区域之外，放置检查失败"
        failures.append({"name": f"{arms[row['receiver']]} · {name}", "reason": reason})
    controls = read_rows(folder / "cold_restart.csv")
    return {"finished": bool(run["finished_at"]), "passed": run["passed"], "failed": run["failed"],
            "pending": run["pending"], "controls": sum(row["status"] == "ok" for row in controls),
            "failures": failures, "report": (COMPARISON_16 / "report.md").as_posix()}


def reference_montages(selected: list[dict[str, Any]], root: Path, roc0_title: str) -> dict[int, dict[str, Any]]:
    """按场来源读取原生基线和 ROC0 实际使用的有符号电流。"""
    montages = {}
    by_source = {}
    for row in selected:
        if row["group"] != "baseline" and row["title"] != roc0_title:
            continue
        source = row["source"]
        if source not in by_source:
            receipt_path = (root / source).with_suffix(".json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            config_path = root / "simulations" / Path(receipt["source"]).name / "params.json"
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            by_source[source] = {"A": config["electrode_A"], "B": config["electrode_B"],
                                 "config": config_path.relative_to(root).as_posix()}
        montages[row["id"]] = by_source[source]
    return montages


def publish_targeted(
    entries: list[dict[str, Any]], root: Path, out: Path, targets: dict[str, str],
    experiments: dict[str, str], arms: dict[str, str],
) -> None:
    """写出主入口及含同靶配对读数的 CSV，沿用原场图。"""
    selected = selected_entries(entries)
    attach_pairs(selected, root, arms)
    data = {"entries": selected, "targets": targets, "experiments": experiments, "roc0_title": arms["roc0"],
            "montages": reference_montages(selected, root, arms["roc0"]),
            "comparison16": comparison_status(root, arms),
            "snapshot": datetime.now().astimezone().isoformat(timespec="minutes")}
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = out / "index.html"
    temp_html = out / "index.html.tmp"
    temp_html.write_text(HTML.replace("@@DATA@@", encoded), encoding="utf-8")
    temp_csv = out / "manifest.csv.tmp"
    with temp_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = tuple(key for key in selected[0] if key != "pairs")
        comparison = ("reference_id", "reference_title", "label", "table", "delta_distance", "delta_coverage_pp",
                      "delta_spill_cm3", "decision", "reason")
        writer = csv.DictWriter(stream, fieldnames=fields + tuple(f"pair_{key}" for key in comparison))
        writer.writeheader()
        for entry in selected:
            record = {key: entry[key] for key in fields}
            for pair in entry["pairs"] or [{}]:
                writer.writerow(record | {f"pair_{key}": value for key, value in pair.items()})
    temp_csv.replace(out / "manifest.csv")
    temp_html.replace(html)
    print(f"已发布同靶汇总 {len(selected)} 项图文，{sum(len(row['pairs']) for row in selected)} 项配对。", flush=True)


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TI 同靶场强对照</title>
<style>
:root{color-scheme:light;font:16px/1.55 "Microsoft YaHei",system-ui,sans-serif;color:#18334a;background:#ecf1f6}
*{box-sizing:border-box}body{margin:0}main{width:min(1420px,100%);margin:auto;padding:24px 26px 50px}
header{margin-bottom:20px}h1{font-size:clamp(1.8rem,4vw,2.5rem);margin:0 0 5px}h2{font-size:1.15rem;margin:0}h3{font-size:1rem;margin:0}
p{margin:.45em 0}small,.subtle{font-size:.87rem;color:#516378}a{color:#065b70}a:hover{text-decoration-thickness:2px}
.controls,.table-wrap,.panel{background:#fff;border:1px solid #d2dfe9;border-radius:12px}.controls{padding:14px 18px;display:flex;flex-wrap:wrap;gap:12px 22px;margin-bottom:22px}
.study-links:not([hidden]){display:flex;flex:1 0 100%;flex-wrap:wrap;gap:4px 16px}
label{display:grid;gap:4px;font-size:.88rem}select{font:inherit;min-width:130px;padding:6px 10px;border:1px solid #a9bacb;border-radius:6px;background:#fff;color:#18334a}
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:5px 16px;flex-wrap:wrap;margin:20px 0 8px}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.panel{padding:11px 13px 8px;scroll-margin-top:12px;min-width:0}
.panel:target{outline:3px solid #168e9d}.panel img{display:block;width:100%;height:auto;margin:8px 0 4px;border-radius:3px}
.metrics{display:flex;flex-wrap:wrap;gap:2px 14px;font-size:.88rem}.panel b,td{font-variant-numeric:tabular-nums}
.montage{display:flex;flex-wrap:wrap;gap:2px 16px;padding:3px 0;font-size:.87rem;font-variant-numeric:tabular-nums;color:#324d64}
.montage span{white-space:nowrap}.montage b{margin-right:.45em;color:#18334a}
.roc0-note{margin:8px 0;padding:4px 9px;border-left:3px solid #168e9d;background:#f2f7fa;font-size:.87rem;color:#324d64}
.score-explainer{margin-top:10px;padding:10px 12px;background:#f2f7fa;border:1px solid #d2dfe9;border-radius:8px;font-size:.9rem}
.score-explainer dl{margin:.65em 0}.score-explainer dt{font-weight:700;margin-top:.5em}.score-explainer dd{margin:.15em 0 0}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;text-align:left;font-size:.9rem}th,td{padding:8px 10px;border-bottom:1px solid #e4ebf1;vertical-align:top}
th{background:#f2f6f9;white-space:nowrap}tbody tr:last-child td{border-bottom:0}.good{color:#006b59}.bad{color:#a33d30}
details{background:#f8fbfc;border:1px solid #d2dfe9;border-radius:10px;margin:10px 0;padding:9px 12px}summary{cursor:pointer;font-weight:600}details .cards{margin-top:10px}
footer{margin-top:22px;font-size:.87rem;color:#516378}footer a{margin-right:14px}.pairs{white-space:nowrap}.comparison{min-width:175px}
@media(max-width:820px){main{padding:15px 10px 35px}.cards{grid-template-columns:minmax(0,1fr)}th,td{padding:6px 8px}}
</style></head><body><main>
<header><h1>TI 同靶场强对照</h1><p>固定头模和靶点，对照原生基线与各方案选中的布局。</p>
<p class="subtle">同一靶点使用相同切面和 0–0.40 V/m 色标。青色轮廓为靶区，白色 × 为主热点投影。覆盖与外溢按 0.10 V/m 计算。</p></header>
<section class="controls" aria-label="选择对照范围">
<label>头模<select id="subject" aria-label="头模"></select></label>
<label>靶点<select id="target" aria-label="靶点"></select></label>
<label>实验<select id="experiment" aria-label="实验"></select></label>
<div id="study-links" class="study-links subtle" hidden></div>
</section>
<section aria-labelledby="baseline-title"><div class="section-head"><h2 id="baseline-title">03 原生基线</h2>
<small>实际原生电流，作为同靶浏览对照</small></div><div id="baselines" class="cards"></div></section>
<section aria-labelledby="results-title"><div class="section-head"><h2 id="results-title">所选靶点的结果</h2>
<small id="summary" role="status"></small></div>
<div id="table-wrap" class="table-wrap"><table><thead><tr><th>方案与布局</th><th>热点距靶</th><th>靶内覆盖</th><th>靶外超阈</th><th>实验内配对差值及判定</th></tr></thead><tbody id="results"></tbody></table></div>
<p class="subtle" id="pair-note">差值相对该实验登记的同条件参照。候选库重排与逐靶搜索各用自己的参照。</p>
<div id="figures"></div></section>
<footer><a href="manifest.csv">下载同靶结果清单</a><a href="full_audit.html">查看全量场图</a><span id="snapshot"></span></footer>
</main>
<template id="score-04"><section id="objective-04" class="score-explainer" aria-labelledby="objective-title-04">
<h3 id="objective-title-04">04 的 A、D、P 评分</h3>
<p>三种评分分别最小化 <b>A</b>、<b>A + D</b>、<b>A + D + P</b>。这里的场强阈值 τ 为 0.10 V/m。</p>
<dl>
<dt>A · 靶内场强不足</dt>
<dd>靶内每个单元的不足量为 max(0, 1 − 场强 / τ)，再按单元体积取平均。低于阈值越多，惩罚越大。</dd>
<dt>D · 全脑强场离靶</dt>
<dd>取脑实质中场强最高的 100 mm³。每个入选单元的 d 是其重心到靶区的最短距离，靶内 d 为 0。对 d / (d + 10 mm) 按入选体积取平均。结果表的「热点距靶」另取主热点簇中心到靶区的距离。</dd>
<dt>P · 靶外皮层热点</dt>
<dd>H<sub>C</sub> 是靶外皮层内场强最高的 100 mm³ 的平均场强，m<sub>T</sub> 是靶内场强的体积加权中位数。P = max(0, H<sub>C</sub> − m<sub>T</sub>) / τ，高于靶内中位数的部分计入惩罚。</dd>
</dl>
</section></template>
<template id="score-05"><section id="objective-05" class="score-explainer" aria-labelledby="objective-title-05">
<h3 id="objective-title-05">05 的覆盖评分和外溢预算</h3>
<p>固定头模和靶点后，三种方案给同一批布局计算同一个覆盖分 <b>S = (K + F) / 2</b>，越高越好。靶内和靶外统一用 0.10 V/m 作为阈值。</p>
<dl>
<dt>K · 硬覆盖</dt>
<dd>靶内达到阈值的体积占整个靶区体积的比例，对应结果表的「靶内覆盖」。</dd>
<dt>F · 软覆盖</dt>
<dd>靶内每个单元按场强 ÷ 0.10 V/m 得分，最高为 1，再按单元体积取平均。</dd>
</dl>
<p>外溢预算以同头模、同靶点的 ROC0 参照为基准。全脑靶外和靶外皮层分别统计超阈体积，各有一个上限。</p>
<dl>
<dt>仅靶区覆盖</dt>
<dd>直接从全部候选中选 S 最高的布局，不设外溢上限。</dd>
<dt>原外溢预算</dt>
<dd>两项靶外超阈体积都须小于或等于 ROC0 各自的基准值，再选 S 最高的布局。</dd>
<dt>75% 外溢预算</dt>
<dd>两个上限分别设为 ROC0 对应基准值的 75%，满足两项约束的布局按 S 排名。</dd>
</dl>
</section></template>
<script type="application/json" id="gallery-data">@@DATA@@</script>
<script>
const data=JSON.parse(document.getElementById('gallery-data').textContent);
const q=id=>document.getElementById(id);
const subject=q('subject'),target=q('target'),experiment=q('experiment');
const categories={screening:'候选库重排',search:'逐靶搜索',diagnostic:'诊断'};
const scoreNotes={'04':'A、D、P 评分说明','05':'覆盖评分与外溢预算说明'};
const params=new URLSearchParams(location.search);
const allSubjects=[...new Set(data.entries.map(row=>row.subject))].sort();
const codes=Object.keys(data.experiments).sort();
const currents=new Intl.NumberFormat('zh-CN',{signDisplay:'always',minimumFractionDigits:1,maximumFractionDigits:12});
function choose(node,options,preferred){
  node.replaceChildren(...options.map(([key,label])=>new Option(label,key)));
  node.value=options.some(([key])=>key===preferred)?preferred:options[0][0];
}
function filterTarget(){return data.entries.filter(row=>row.subject===subject.value&&row.target===target.value);}
function filters(wantedTarget,wantedExperiment){
  const targets=new Set(data.entries.filter(row=>row.subject===subject.value).map(row=>row.target));
  choose(target,Object.entries(data.targets).filter(([id])=>targets.has(id)),wantedTarget);
  experimentChoices(wantedExperiment);
}
function experimentChoices(preferred){
  const rows=filterTarget();
  const options=[['all','全部方案结果']];
  for(const code of codes.filter(code=>rows.some(row=>row.experiment===code&&row.group!=='baseline'&&(row.group!=='diagnostic'||code==='14')))){
    options.push([code,code+' '+data.experiments[code]]);
  }
  if(rows.some(row=>row.group==='diagnostic'))options.push(['diagnostic','同靶诊断']);
  choose(experiment,options,preferred);
}
function add(parent,tag,text,cls){
  const node=document.createElement(tag);
  if(text!==undefined)node.textContent=text;
  if(cls)node.className=cls;
  parent.append(node);
  return node;
}
function link(parent,label,href){const node=add(parent,'a',label);node.href=href;return node;}
function source(path){return '../../'+path.split('/').map(encodeURIComponent).join('/');}
function metric(value,unit,scale=1){return value===null?'—':(value*scale).toFixed(2)+' '+unit;}
function signed(value,unit){const rounded=Number(value.toFixed(2));return (rounded>0?'+':'')+rounded.toFixed(2)+' '+unit;}
function showCard(row){
  const card=q('figure-'+row.id);
  if(!card)return;
  const group=card.closest('details');
  if(group)group.open=true;
  card.scrollIntoView({block:'start'});
}
function jump(parent,label,row){
  const anchor=link(parent,label,'#figure-'+row.id);
  anchor.onclick=()=>showCard(row);
  return anchor;
}
function card(parent,row){
  const panel=add(parent,'article',undefined,'panel');
  panel.id='figure-'+row.id;
  add(panel,'h3',row.title);
  add(panel,'small',row.kind+(row.reference&&row.group!=='baseline'?' · 参照':''));
  const stats=add(panel,'p',undefined,'metrics');
  add(stats,'b','热点距靶 '+metric(row.distance,'mm'));
  add(stats,'b','靶内覆盖 '+metric(row.coverage,'%',100));
  add(stats,'b','靶外超阈 '+metric(row.spill,'cm³',.001));
  if(row.title===data.roc0_title){
    const note=add(panel,'div',undefined,'roc0-note');
    add(note,'p','同一头模上，两条原生路线找到的电极布局及其单电极邻近变体组成候选库。每组布局用两路各 1 mA 复算，再针对当前靶点逐一打分，选损失最低的一组。');
    add(note,'p','打分沿用 leadfield-based 和 leadfield-free 共用的原逆向优化目标函数。以 0.10 V/m 为阈值，按四面体个数计算靶内未达标率和靶外超阈率，损失是这两个比例平方和的平方根。');
  }
  const montage=data.montages[row.id];
  if(montage){
    const currentRows=add(panel,'div',undefined,'montage');
    for(const channel of ['A','B']){
      const channelRow=add(currentRows,'span');
      add(channelRow,'b',channel+' 路');
      const electrodes=montage[channel].map(e=>e.name+' '+currents.format(e.current_mA).replace('-','−')+' mA');
      add(channelRow,'span',electrodes.join(' · '));
    }
  }
  const image=add(link(panel,undefined,row.figure),'img');
  image.src=row.figure;image.width=928;image.height=303;image.loading='lazy';
  image.alt=row.subject+' '+data.targets[row.target]+' '+row.title+'的彩色场强三视图';
  const provenance=add(panel,'p',undefined,'subtle');
  link(provenance,'原始结果',source(row.table));
  if(montage){
    add(provenance,'span',' · ');
    link(provenance,'电极与电流参数',source(montage.config));
  }
  if(scoreNotes[row.experiment]&&row.title!==data.roc0_title){
    add(provenance,'span',' · ');
    link(provenance,scoreNotes[row.experiment],'#objective-'+row.experiment);
  }
  if(row.token)add(provenance,'span',' · '+row.token);
}
function tableRow(parent,row){
  const line=add(parent,'tr');
  jump(add(line,'td'),row.experiment+' · '+row.title,row);
  add(line,'td',metric(row.distance,'mm'));
  add(line,'td',metric(row.coverage,'%',100));
  add(line,'td',metric(row.spill,'cm³',.001));
  const column=add(line,'td',undefined,'comparison');
  for(const pair of row.pairs){
    const ref=data.entries[pair.reference_id];
    const heading=add(column,'div');
    jump(heading,pair.reference_title,ref);
    add(heading,'small',' · '+pair.label);
    add(column,'div',signed(pair.delta_distance,'mm')+' · '+signed(pair.delta_coverage_pp,'百分点')+' · '+signed(pair.delta_spill_cm3,'cm³'),'pairs');
    const reason=add(column,'div',undefined,'subtle');
    add(reason,'span',pair.decision,pair.decision==='未通过'?'bad':pair.decision==='通过改善门槛'?'good':undefined);
    if(pair.reason)add(reason,'span',' · '+pair.reason);
    add(reason,'span',' · ');
    link(reason,'配对表',source(pair.table));
  }
  if(!row.pairs.length)column.textContent=row.reference?'实验参照':'—';
}
function showComparisonStatus(parent){
  const run=data.comparison16;
  const block=add(parent,'section',undefined,'score-explainer');
  add(block,'h3',run.finished?'16 已结束':'16 进行中');
  add(block,'p','等预算对照 '+run.passed+' 项已出场、'+run.failed+' 项失败、'+run.pending+' 项待处理。已完成从零计时对照 '+run.controls+' 项。');
  for(const item of run.failures)add(block,'p',item.name+' · '+item.reason);
  link(block,'16 实验报告',source(run.report));
}
function display(){
  const rows=filterTarget();
  const baselines=q('baselines'),results=q('results'),figures=q('figures');
  baselines.replaceChildren();results.replaceChildren();figures.replaceChildren();
  for(const row of rows.filter(row=>row.group==='baseline'))card(baselines,row);
  const selection=experiment.value;
  const visible=rows.filter(row=>selection==='all'?['screening','search'].includes(row.group):selection==='diagnostic'?row.group==='diagnostic':row.experiment===selection);
  q('summary').textContent=visible.length+' 项 · '+subject.value+' · '+data.targets[target.value];
  q('table-wrap').hidden=!visible.length;
  q('pair-note').hidden=!visible.length||selection==='diagnostic'||selection==='14';
  if(!visible.length)add(figures,'p','该头模和靶点目前有原生基线。');
  for(const row of visible)tableRow(results,row);
  for(const code of codes.filter(code=>visible.some(row=>row.experiment===code))){
    const experimentRows=visible.filter(row=>row.experiment===code);
    const group=add(figures,'details');
    group.open=selection!=='all'&&selection!=='diagnostic'||code==='04';
    add(group,'summary',code+' '+data.experiments[code]+' · '+categories[experimentRows[0].group]);
    if(scoreNotes[code])group.append(q('score-'+code).content.cloneNode(true));
    if(code==='16')showComparisonStatus(group);
    const cards=add(group,'div',undefined,'cards');
    for(const row of [...experimentRows].sort((a,b)=>Number(b.reference)-Number(a.reference)))card(cards,row);
  }
}
const linkedCases=new Map(data.entries.filter(row=>['14','16'].includes(row.experiment))
  .map(row=>[row.experiment+'/'+row.subject+'/'+row.target,row]));
if(linkedCases.size){
  const navigation=q('study-links');
  navigation.hidden=false;
  for(const {experiment,subject,target} of linkedCases.values()){
    const query=new URLSearchParams({subject,target,experiment});
    link(navigation,experiment+' '+data.experiments[experiment]+' · '+subject+' · '+data.targets[target],'?'+query.toString());
  }
}
choose(subject,allSubjects.map(id=>[id,id]),params.get('subject')||'sub2');
filters(params.get('target')||'gp222',params.get('experiment')||'all');
subject.onchange=()=>{filters(target.value,experiment.value);display();};
target.onchange=()=>{experimentChoices(experiment.value);display();};
experiment.onchange=display;
q('snapshot').textContent='结果快照 '+data.snapshot;
display();
</script></body></html>
"""
