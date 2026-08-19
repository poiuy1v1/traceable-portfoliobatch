from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .io_utils import atomic_write_bytes

PALETTE = {
    "blue": "#4C78A8",
    "dark_blue": "#2F5B85",
    "orange": "#F2A541",
    "green": "#4E9F6D",
    "red": "#C75B5B",
    "grey": "#A7ADB6",
    "light": "#EEF2F6",
    "grid": "#D9DEE5",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.linewidth": 0.75,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "svg.hashsalt": "traceable-portfoliobatch-v45A3",
    })


def _panel(ax, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, f"{label})", transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")


def _save(fig, root: Path, stem: str) -> list[str]:
    out = root / "figures"
    out.mkdir(exist_ok=True)
    paths = []
    metadata_by_ext = {
        "png": {"Software": "Traceable PortfolioBatch"},
        "pdf": {
            "Title": stem,
            "Author": "Jinnan Wei; Andrew E. H. Wheatley",
            "Subject": "Traceable PortfolioBatch figure",
            "Keywords": "MOF; reproducibility; batch selection",
            "Creator": "Traceable PortfolioBatch",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
        "svg": {
            "Title": stem,
            "Creator": "Traceable PortfolioBatch",
            "Date": "2026-08-09",
            "Description": "Deterministic figure artifact",
        },
    }
    for ext in ["png", "pdf", "svg"]:
        p = out / f"{stem}.{ext}"
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format=ext,
            bbox_inches="tight",
            metadata=metadata_by_ext[ext],
        )
        payload = buffer.getvalue()
        if ext == "svg":
            # Matplotlib emits presentation-only spaces before newlines in SVG
            # path data.  Normalize them so generated text artifacts satisfy
            # the repository whitespace gate without changing the rendering.
            svg = payload.decode("utf-8")
            payload = (
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
            ).encode("utf-8")
        atomic_write_bytes(p, payload)
        paths.append(p.relative_to(root).as_posix())
    plt.close(fig)
    return paths


def _box(ax, xy, wh, text, face, fontsize=6.2) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=face, edgecolor="#7D8794", lw=.7))
    ax.text(x + w/2, y + h/2, text, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize)


def build_figure1(root: Path) -> list[str]:
    _style()
    locked = json.loads(
        (
            root
            / "results/reproduction/tuning/v49_locked_parameters_before_test.json"
        ).read_text()
    )
    fig = plt.figure(figsize=(8.2, 3.95))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=.30, left=.035, right=.985, top=.98, bottom=.05)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    labels = [
        ("Train split", "source-group CV\n+ fit score model"),
        ("Calibration split", "select policy\nparameters"),
        ("Lock protocol", "freeze primary\ncost contract"),
        ("Test split", "final selector\ncomparison"),
        ("Trace + replay", "archive stepwise\ndecisions"),
    ]
    xs = [0.015, 0.215, 0.435, 0.64, 0.82]
    widths = [0.16, 0.18, 0.17, 0.145, 0.165]
    colors = ["#EAF1F8", "#FFF2DB", "#E9F5EA", "#F3EAF8", "#EAF4F4"]
    for i, ((title, sub), x, w, c) in enumerate(zip(labels, xs, widths, colors)):
        _box(ax, (x, .25), (w, .48), f"{title}\n{sub}", c, fontsize=9.2)
        if i < len(labels)-1:
            ax.annotate("", xy=(xs[i+1]-.008, .49), xytext=(x+w+.008, .49), xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", lw=1.05, color="#4D5966"))
    ax.text(.5, .08, f"Locked activation C = {locked['activation_C']:g}; test data are excluded from tuning",
            transform=ax.transAxes, ha="center", fontsize=9.0)
    _panel(ax, "a", x=-.005, y=1.02)

    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    _box(ax, (.015, .27), (.18, .48), "Current batch\n$B_{t-1}$", "#EAF1F8", fontsize=9.2)
    _box(ax, (.245, .27), (.23, .48), "Primary feasibility filter\ncost budget only", "#FFF2DB", fontsize=9.2)
    _box(ax, (.525, .27), (.25, .48), "Greedy marginal priority\nscore + uncertainty\n- normalized cost + diversity", "#E9F5EA", fontsize=8.8)
    _box(ax, (.825, .27), (.16, .48), "Select candidate\nappend step record", "#F3EAF8", fontsize=9.0)
    for a, b in [(.195, .245), (.475, .525), (.775, .825)]:
        ax.annotate("", xy=(b-.008, .51), xytext=(a+.008, .51), xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=1.05, color="#4D5966"))
    ax.text(.5, .08, "Legacy risk cap is audited separately as a stress test; no global batch-optimality claim",
            transform=ax.transAxes, ha="center", fontsize=9.0)
    _panel(ax, "b", x=-.005, y=1.02)
    return _save(fig, root, "Figure1_workflow")

def build_figure2(root: Path) -> list[str]:
    _style()
    leak=json.loads((root/"results/reproduction/leakage_audit.json").read_text())
    sg=json.loads((root/"results/reproduction/source_group_audit.json").read_text())
    replay=json.loads((root/"results/reproduction/replay_summary.json").read_text())
    act=json.loads((root/"results/reproduction/activation_model_metrics.json").read_text())
    fig=plt.figure(figsize=(8.2,4.65)); gs=fig.add_gridspec(2,3,height_ratios=[.92,1.28],wspace=.28,hspace=.30,left=.04,right=.99,top=.98,bottom=.055)
    ax=fig.add_subplot(gs[0,:]); ax.axis("off")
    items=[("Training",f"n = {act['n_train']}\nsource-group CV"),("Final calibration",f"n = {act['n_calibration']}\ncost-only policy lock"),("Final test",f"n = {act['n_test']}\nlocked evaluation"),("Excluded overlap",f"n = {sg['excluded_train_source_overlap_row_count']}\nprovenance only"),("Primary contract","cost-only +\ngroup resampling")]
    xs=[.01,.20,.39,.575,.78]; ws=[.15,.16,.15,.17,.205]
    for i,((t,sub),x,w) in enumerate(zip(items,xs,ws)):
        _box(ax,(x,.20),(w,.56),f"{t}\n{sub}",["#EAF1F8","#FFF2DB","#F3EAF8","#E9F5EA","#EAF4F4"][i],fontsize=8.5)
        if i<len(items)-1: ax.annotate("",xy=(xs[i+1]-.006,.48),xytext=(x+w+.006,.48),xycoords=ax.transAxes,textcoords=ax.transAxes,arrowprops=dict(arrowstyle="-|>",lw=1,color="#4D5966"))
    _panel(ax,"a",x=-.005,y=1.03)
    ax=fig.add_subplot(gs[1,0]); ax.axis("off"); ax.set_title("Included evidence",fontsize=10.2,pad=5)
    for i,t in enumerate(["activation label","model score","entropy proxy","descriptor cost","legacy risk stress audit"]): _box(ax,(.045,.80-i*.165),(.91,.12),t,"#EAF1F8",fontsize=8.6)
    _panel(ax,"b",x=-.10,y=1.08)
    ax=fig.add_subplot(gs[1,1]); ax.axis("off"); ax.set_title("Out of scope",fontsize=10.2,pad=5)
    for i,t in enumerate(["water stability","hydrolytic durability","measured synthesis cost","calibrated uncertainty","closed-loop discovery"]): _box(ax,(.035,.80-i*.165),(.93,.12),t,"#FBEAEA",fontsize=8.5)
    _panel(ax,"c",x=-.10,y=1.08)
    ax=fig.add_subplot(gs[1,2]); ax.axis("off"); ax.set_title("Leakage and replay guardrails",fontsize=9.7,pad=5)
    rows=[("train-calibration DOI",sg["train_calibration_source_doi_overlap_count"]),("train-test DOI",sg["train_test_source_doi_overlap_count"]),("calibration-test DOI",sg["policy_calibration_test_source_doi_overlap_count"]),("training DOI coverage",f"{100*sg['active_training_doi_coverage']:.0f}%"),("excluded overlap rows",sg["excluded_train_source_overlap_row_count"]),("exact replay",f"{replay['passed_count']} of {replay['trace_count']} passed")]
    for i,(k,v) in enumerate(rows):
        y=.90-i*.14; ax.text(.02,y,k,transform=ax.transAxes,fontsize=7.7); ax.text(.98,y,str(v),transform=ax.transAxes,ha="right",fontsize=7.7,fontweight="bold"); ax.plot([.02,.98],[y-.045,y-.045],transform=ax.transAxes,color=PALETTE["grid"],lw=.7)
    _panel(ax,"d",x=-.10,y=1.08)
    return _save(fig,root,"Figure2_benchmark_boundary")

def build_figure3(root: Path) -> list[str]:
    _style()
    pol=pd.read_csv(root/"source_data/figure3_test_policy_primary_cost_source.csv")
    rnd=pd.read_csv(root/"source_data/figure3_random_baseline_primary_cost_raw_points.csv")
    pool=pd.read_csv(root/"source_data/figure3_pool_resampling_primary_cost_raw_points.csv")
    names={"top_score":"top-score","uncertainty":"uncertainty","cost_aware":"cost-aware","portfolio_batch":"PortfolioBatch","random_baseline":"random"}
    order=["top_score","uncertainty","cost_aware","portfolio_batch","random_baseline"]
    pol=pol.set_index("method").loc[order].reset_index()
    fig,axs=plt.subplots(1,3,figsize=(8.6,3.55),gridspec_kw={"wspace":.42,"left":.055,"right":.99,"top":.94,"bottom":.25})
    ax=axs[0]; x=np.arange(len(pol)); ax.bar(x,pol.selected_count,color=PALETTE["grey"],width=.72,label="selected"); ax.bar(x,pol.hits,color=PALETTE["blue"],width=.46,label="stable hits"); ax.set_xticks(x); ax.set_xticklabels([names[m] for m in pol.method],rotation=25,ha="right",fontsize=6.9); ax.set_ylim(0,5.7); ax.set_ylabel("Candidates",fontsize=8.8); ax.set_title("Locked final DOI-group test\nprimary cost-only contract",fontsize=9.2); ax.legend(fontsize=7.3); ax.grid(axis="y",color=PALETTE["grid"],lw=.55); _panel(ax,"a",x=-.16,y=1.08)
    ax=axs[1]; batches=[5,10,20]; data=[]
    for k in batches:
        g=rnd[rnd.batch_size.eq(k)]; data.append((g.hits/g.batch_size).to_numpy())
    bp=ax.boxplot(data,positions=batches,widths=1.9,patch_artist=True,showfliers=False,medianprops={"color":"#222"})
    for b in bp["boxes"]: b.set_facecolor(PALETTE["light"]); b.set_edgecolor(PALETTE["blue"])
    rng=np.random.default_rng(45)
    for k,vals in zip(batches,data): ax.scatter(k+rng.normal(0,.25,len(vals)),vals,s=7,alpha=.35,color=PALETTE["blue"],lw=0)
    ax.set_xticks(batches); ax.set_ylim(0,1.05); ax.set_xlabel("Requested batch size",fontsize=8.8); ax.set_ylabel("Stable hits / requested slot",fontsize=8.5); ax.set_title("100 random full-pool runs\nprimary cost-only contract",fontsize=9.2); ax.grid(axis="y",color=PALETTE["grid"],lw=.55); _panel(ax,"b",x=-.16,y=1.08)
    ax=axs[2]; methods=order; xpos=np.arange(len(methods)); rng=np.random.default_rng(46)
    for i,m in enumerate(methods):
        g=pool[pool.method.eq(m)].copy(); vals=(g.hits/g.batch_size).to_numpy(); jitter=rng.normal(0,.07,len(vals)); ax.scatter(np.full(len(vals),i)+jitter,vals,s=7,alpha=.25 if m=="random_baseline" else .55,color=PALETTE["orange"] if m=="random_baseline" else PALETTE["blue"],lw=0)
        ax.plot([i-.18,i+.18],[np.mean(vals),np.mean(vals)],color="#222",lw=1.4)
    ax.set_xticks(xpos); ax.set_xticklabels([names[m] for m in methods],rotation=27,ha="right",fontsize=6.6); ax.set_ylim(0,1.05); ax.set_ylabel("Stable hits / requested slot",fontsize=8.5); ax.set_title("20 DOI-group pool resamples\nprimary cost-only contract",fontsize=9.2); ax.grid(axis="y",color=PALETTE["grid"],lw=.55); _panel(ax,"c",x=-.16,y=1.08)
    return _save(fig,root,"Figure3_policy_comparison")

def build_figure4(root: Path) -> list[str]:
    _style()
    pca = pd.read_csv(root / "source_data/figure4_pca_coordinates.csv")
    variance = json.loads((root / "results/reproduction/descriptor_pca_variance.json").read_text())
    manifest = pd.read_csv(root / "data/processed/full_public_stability_manifest_scored.csv")
    split_column = "final_evaluation_split" if "final_evaluation_split" in manifest else "model_split"
    manifest = manifest[manifest[split_column] == "test"].reset_index(drop=True)
    trace = json.loads((root / "traces/action_traces/portfolio_batch_test_primary_cost_contract_trace.json").read_text())
    selected = pd.DataFrame(trace["selected_candidates"])
    alternatives = pd.DataFrame(trace["logged_alternative_candidates"])

    fig = plt.figure(figsize=(8.2, 3.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.05], wspace=.38, left=.075, right=.96, top=.95, bottom=.22)
    ax1 = fig.add_subplot(gs[0, 0])
    stable = pca[pca.outcome_binary.astype(float) > 0]; unstable = pca[pca.outcome_binary.astype(float) <= 0]
    ax1.scatter(unstable.pc1, unstable.pc2, s=12, c=PALETTE["grey"], alpha=.65, lw=0, label="unstable")
    ax1.scatter(stable.pc1, stable.pc2, s=12, c=PALETTE["blue"], alpha=.72, lw=0, label="stable")
    ax1.set_xlabel(f"PC1 ({variance['pc1_percent']:.2f}%)", fontsize=9.2); ax1.set_ylabel(f"PC2 ({variance['pc2_percent']:.2f}%)", fontsize=9.2)
    ax1.set_title("Final active test descriptor PCA", fontsize=10.2, pad=7); ax1.tick_params(labelsize=8.2)
    ax1.legend(title="Activation label", loc="upper right", fontsize=8.0, title_fontsize=8.4)
    _panel(ax1, "a", x=-.15, y=1.08)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(manifest.proxy_total_cost, manifest.predicted_success, s=12, c="#CBD1DC", alpha=.40, lw=0)
    if len(alternatives):
        ax2.scatter(alternatives.proxy_total_cost, alternatives.predicted_success, s=38, c=PALETTE["red"], marker="x", lw=1.0)
    ax2.scatter(selected.proxy_total_cost, selected.predicted_success, s=42, c=PALETTE["dark_blue"], edgecolors="white", lw=.6)
    ax2.set_xlabel("Proxy total cost", fontsize=9.2); ax2.set_ylabel("Predicted success", fontsize=9.2)
    ax2.set_title("Locked PortfolioBatch test decision", fontsize=10.0, pad=7); ax2.tick_params(labelsize=8.2)
    ax2.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CBD1DC", markersize=5, label="test pool"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["dark_blue"], markersize=5, label=f"selected (n={len(selected)})"),
        Line2D([0], [0], marker="x", color=PALETTE["red"], markersize=5, label="logged alternatives"),
    ], title="Trace role", loc="lower right", fontsize=7.9, title_fontsize=8.3)
    _panel(ax2, "b", x=-.15, y=1.08)
    return _save(fig, root, "Figure4_descriptor_materials_analysis")


def build_supplementary_figure2(root: Path) -> list[str]:
    _style()
    contrast = pd.read_csv(root / "source_data/figure4_descriptor_contrasts.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.0), gridspec_kw={"left": .30, "right": .97, "top": .93, "bottom": .18})
    desc = contrast.descriptor.tolist(); vals = contrast.stable_minus_unstable_smd.astype(float).tolist(); y = np.arange(len(desc))[::-1]
    ax.barh(y, vals, color=[PALETTE["blue"] if v >= 0 else "#D98B8A" for v in vals], edgecolor="#333", lw=.45, height=.65)
    ax.axvline(0, color="#333", lw=.85); ax.set_yticks(y); ax.set_yticklabels(desc, fontsize=9.0)
    ax.set_xlabel("Stable - unstable standardized mean difference", fontsize=9.8)
    ax.set_title("Final active test descriptor contrasts", fontsize=10.8, pad=8)
    ax.grid(axis="x", color=PALETTE["grid"], lw=.55); ax.tick_params(axis="x", labelsize=8.6)
    return _save(fig, root, "Supplementary_Figure2_descriptor_contrasts")

def build_figure5(root: Path) -> list[str]:
    _style()
    replay = json.loads((root / "results/reproduction/replay_summary.json").read_text())
    activation = json.loads((root / "results/reproduction/activation_model_metrics.json").read_text())
    leak = json.loads((root / "results/reproduction/leakage_audit.json").read_text())
    source_group = json.loads((root / "results/reproduction/source_group_audit.json").read_text())
    trace = json.loads((root / "traces/action_traces/portfolio_batch_test_primary_cost_contract_trace.json").read_text())
    fig = plt.figure(figsize=(8.2, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=.32, left=.045, right=.98, top=.95, bottom=.08)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_title("Stepwise decision record", fontsize=10.4, pad=7)
    items = ["input, model, code and environment hashes", "locked primary cost-only configuration", "cost-feasibility counts; risk retained descriptively", "priority components and selected candidate", "top feasible alternatives and replay specification"]
    for i, t in enumerate(items):
        y = .91 - i*.18
        ax.add_patch(plt.Rectangle((.04, y-.08), .92, .12, transform=ax.transAxes, facecolor=PALETTE["light"], edgecolor="#8A96A6", lw=.75))
        ax.text(.5, y-.02, t, transform=ax.transAxes, ha="center", va="center", fontsize=8.7)
        if i < len(items)-1:
            ax.annotate("", xy=(.5, y-.14), xytext=(.5, y-.08), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", lw=.9))
    selection_events=sum(x.get("status")=="selected" for x in trace["stepwise_decisions"]); stop_events=sum(x.get("status","").startswith("stopped") for x in trace["stepwise_decisions"])
    ax.text(.5, .02, f"{selection_events} candidate selections + {stop_events} stop event", transform=ax.transAxes, ha="center", fontsize=8.4)
    _panel(ax, "a", x=-.08, y=1.08)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_title("Exact replay and integrity audit", fontsize=10.4, pad=7)
    rows = [
        ("archived traces replayed", f"{replay['passed_count']} / {replay['trace_count']}"),
        ("score-artifact reload", "< 1e-12 contract"),
        ("train-calibration DOI overlap", source_group["train_calibration_source_doi_overlap_count"]),
        ("train-test DOI overlap", source_group["train_test_source_doi_overlap_count"]),
        ("calibration-test DOI overlap", source_group["policy_calibration_test_source_doi_overlap_count"]),
        ("excluded overlap rows", source_group["excluded_train_source_overlap_row_count"]),
    ]
    for i, (k, v) in enumerate(rows):
        y = .88 - i*.14
        ax.text(.02, y, k, transform=ax.transAxes, ha="left", fontsize=8.5)
        ax.text(.98, y, str(v), transform=ax.transAxes, ha="right", fontsize=8.5, fontweight="bold")
        ax.plot([.02, .98], [y-.045, y-.045], transform=ax.transAxes, color=PALETTE["grid"], lw=.7)
    ax.text(.5, .04, "Archived-implementation consistency; not independent software validation", transform=ax.transAxes, ha="center", fontsize=8.0)
    _panel(ax, "b", x=-.10, y=1.08)
    return _save(fig, root, "Figure5_trace_replay_audit")

def build_supplementary_figure1(root: Path) -> list[str]:
    _style()
    tm = json.loads((root / "results/reproduction/thermal_regression_metrics.json").read_text())
    fig = plt.figure(figsize=(8.0, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[.95, 1.25], wspace=.35, left=.07, right=.97, top=.94, bottom=.18)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_title("Secondary thermal contract", fontsize=10.5, pad=7)
    rows = [("training rows", tm["n_train"]), ("validation rows", tm["n_validation"]), ("alpha selection", "5-fold training CV"), ("selected alpha", f"{tm['selected_alpha']:g}"), ("temperature unit", "°C")]
    for i, (k, v) in enumerate(rows):
        y = .88 - i*.16
        ax.text(.04, y, k, transform=ax.transAxes, fontsize=8.7)
        ax.text(.96, y, str(v), transform=ax.transAxes, ha="right", fontsize=8.7, fontweight="bold")
        ax.plot([.04, .96], [y-.05, y-.05], transform=ax.transAxes, color=PALETTE["grid"], lw=.7)
    _panel(ax, "a", x=-.12, y=1.08)

    ax = fig.add_subplot(gs[0, 1])
    vals = [tm["mae_C"], tm["rmse_C"], tm["training_mean_baseline_rmse_C"]]
    labels = ["MAE", "RMSE", "training-mean\nbaseline RMSE"]
    bars = ax.barh(np.arange(3), vals, color=[PALETTE["blue"], PALETTE["dark_blue"], PALETTE["grey"]])
    ax.set_yticks(np.arange(3)); ax.set_yticklabels(labels, fontsize=8.7); ax.invert_yaxis(); ax.set_xlabel("Error (°C)", fontsize=8.4)
    ax.set_title(f"Validation metrics; model-level R² = {tm['r2']:.3f}", fontsize=10.0, pad=7)
    for b, v in zip(bars, vals): ax.text(v+1, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", fontsize=8.6)
    ax.grid(axis="x", color=PALETTE["grid"], lw=.55); ax.tick_params(axis="x", labelsize=8.3)
    _panel(ax, "b", x=-.13, y=1.08)
    return _save(fig, root, "Supplementary_Figure1_secondary_thermal_repository_context")

def build_all(root: Path) -> list[str]:
    generated: list[str] = []
    generated += build_figure1(root)
    generated += build_figure2(root)
    generated += build_figure3(root)
    generated += build_figure4(root)
    generated += build_figure5(root)
    generated += build_supplementary_figure1(root)
    generated += build_supplementary_figure2(root)
    return generated
