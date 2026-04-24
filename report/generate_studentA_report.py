#!/usr/bin/env python3
"""Generate the Group 8 final report DOCX with Student A results only."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
SUMMARY_CSV = ROOT / "metrics" / "analysis" / "selected_scope_summary.csv"
COVERAGE_CSV = ROOT / "metrics" / "analysis" / "config_coverage_audit.csv"
OUTPUT_PATH = REPORT_DIR / "group8_final_report_studentA_results.docx"

TEAM_MEMBERS = [
    ("Student A", "Mohammad", "fmoha077@uottawa.ca"),
    ("Student B", "Anthony Nasr", "anasr014@uottawa.ca"),
    ("Student C", "Md Mosarraf", "mhoss048@uottawa.ca"),
    ("Student D", "Mariana Chavez Flores", "mchaz097@uottawa.ca"),
]

GITHUB_URL = "https://github.com/mohammad1774/elg5214_group_8_project"

BODY_FIGURES = [
    (
        ROOT / "visualizations" / "analysis" / "cartpole" / "cartpole_dense_compare_by_algorithm.png",
        "Figure 1. CartPole dense mean training reward aggregated across all included "
        "Student A DQN entropy and DQN RND configurations. The cohort average favors "
        "entropy early, even though the best-config table shows that both methods can "
        "perform strongly on this easier dense-reward task.",
    ),
    (
        ROOT / "visualizations" / "analysis" / "cartpole" / "cartpole_sparse_eval_success_compare_by_seed.png",
        "Figure 2. CartPole sparse checkpoint evaluation success aggregated by seed "
        "across the selected Student A runs. Sparse training rewards alone look strong, "
        "but greedy checkpoint success remains near zero for most seeds.",
    ),
    (
        ROOT / "visualizations" / "analysis" / "mountaincar_mc8000" / "mountaincar_mc8000_dense_compare_by_algorithm.png",
        "Figure 3. MountainCar dense 8000-episode follow-up reruns aggregated across "
        "all included Student A entropy and RND configurations. Both methods remain "
        "stuck at the failure floor, indicating that longer training alone does not "
        "solve the exploration problem.",
    ),
    (
        ROOT / "visualizations" / "analysis" / "mountaincar_mc8000" / "mountaincar_mc8000_sparse_compare_by_algorithm.png",
        "Figure 4. MountainCar sparse 8000-episode follow-up reruns aggregated across "
        "all included Student A entropy and RND configurations. RND produces occasional "
        "improvements, but the aggregate still sits close to failure.",
    ),
    (
        ROOT / "visualizations" / "analysis" / "mountaincar_mc8000" / "mountaincar_mc8000_sparse_eval_success_compare_by_seed.png",
        "Figure 5. MountainCar sparse checkpoint evaluation success aggregated by seed "
        "across the selected Student A reruns. Only a small number of seeds show late "
        "success spikes, highlighting the instability of the improvement.",
    ),
]

APPENDIX_FIGURES = [
    (
        ROOT / "visualizations" / "analysis" / "cartpole" / "cartpole_dense_compare_by_alpha_dqn_entropy.png",
        "Figure A1. CartPole dense mean training reward for DQN entropy, grouped by "
        "alpha across the complete Student A scope.",
    ),
    (
        ROOT / "visualizations" / "analysis" / "cartpole" / "cartpole_dense_compare_by_beta_dqn_rnd.png",
        "Figure A2. CartPole dense mean training reward for DQN RND, grouped by beta "
        "across the complete Student A scope.",
    ),
    (
        ROOT / "visualizations" / "analysis" / "cartpole" / "cartpole_sparse_compare_by_algorithm.png",
        "Figure A3. CartPole sparse mean training reward aggregated by algorithm. The "
        "binary sparse reward quickly saturates and therefore needs to be interpreted "
        "together with the evaluation-success plots.",
    ),
    (
        ROOT / "visualizations" / "analysis" / "mountaincar_mc8000" / "mountaincar_mc8000_sparse_compare_by_beta_dqn_rnd.png",
        "Figure A4. MountainCar sparse 8000-episode mean training reward for DQN RND, "
        "grouped by beta across the follow-up Student A reruns.",
    ),
]

REFERENCES = [
    '[1] V. Mnih et al., "Human-level control through deep reinforcement learning," '
    "Nature, vol. 518, no. 7540, pp. 529-533, 2015.",
    '[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, '
    '"Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.',
    '[3] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell, '
    '"Curiosity-driven Exploration by Self-supervised Prediction," in Proceedings of '
    "the 34th International Conference on Machine Learning, pp. 2778-2787, 2017.",
    '[4] Y. Burda, H. Edwards, A. Storkey, and O. Klimov, '
    '"Exploration by Random Network Distillation," in International Conference on '
    "Learning Representations, 2019.",
    '[5] R. T. Lange, "{gymnax}: A {JAX}-based Reinforcement Learning Environment '
    'Library," software, 2022. [Online]. Available: http://github.com/RobertTLange/gymnax',
]


@dataclass(frozen=True)
class AggregatedConfig:
    env_name: str
    reward_type: str
    algorithm: str
    param_name: str
    param_value: float
    learning_rate: float
    gamma: float
    mean_reward: float
    se_reward: float
    mean_success: float
    se_success: float
    n: int
    config_key: str


@dataclass(frozen=True)
class CoverageSummary:
    included_groups: int
    excluded_groups: int
    excluded_keys: tuple[str, ...]


@dataclass(frozen=True)
class StandoutRun:
    label: str
    path: Path
    mean_reward: float
    success_rate: float
    mean_length: float
    wall_time_s: float


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(text: str) -> float:
    return float(text)


def mean_and_se(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(values))


def load_best_configs(path: Path) -> dict[tuple[str, str, str], AggregatedConfig]:
    grouped: dict[tuple[str, str, str, str, float, float, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        key = (
            row["env_name"],
            row["reward_type"],
            row["algorithm"],
            row["mechanism_param_name"],
            to_float(row["mechanism_param_value"]),
            to_float(row["learning_rate"]),
            to_float(row["gamma"]),
            row["config_key"],
        )
        grouped[key].append(row)

    best: dict[tuple[str, str, str], AggregatedConfig] = {}
    for key, rows in grouped.items():
        rewards = [to_float(row["final_mean_reward"]) for row in rows]
        successes = [to_float(row["final_success_rate"]) for row in rows]
        mean_reward, se_reward = mean_and_se(rewards)
        mean_success, se_success = mean_and_se(successes)
        config = AggregatedConfig(
            env_name=key[0],
            reward_type=key[1],
            algorithm=key[2],
            param_name=key[3],
            param_value=key[4],
            learning_rate=key[5],
            gamma=key[6],
            mean_reward=mean_reward,
            se_reward=se_reward,
            mean_success=mean_success,
            se_success=se_success,
            n=len(rows),
            config_key=key[7],
        )
        best_key = (config.env_name, config.reward_type, config.algorithm)
        if best_key not in best:
            best[best_key] = config
            continue
        current = best[best_key]
        if (config.mean_success, config.mean_reward) > (current.mean_success, current.mean_reward):
            best[best_key] = config
    return best


def load_coverage_summary(path: Path) -> CoverageSummary:
    rows = read_csv_rows(path)
    included = sum(row["include_in_analysis"] == "True" for row in rows)
    excluded_rows = [row["config_key"] for row in rows if row["include_in_analysis"] != "True"]
    return CoverageSummary(
        included_groups=included,
        excluded_groups=len(excluded_rows),
        excluded_keys=tuple(sorted(excluded_rows)),
    )


def load_standout_run(label: str, path: Path) -> StandoutRun:
    row = read_csv_rows(path)[0]
    return StandoutRun(
        label=label,
        path=path,
        mean_reward=to_float(row["final_mean_reward"]),
        success_rate=to_float(row["final_success_rate"]),
        mean_length=to_float(row["mean_length"]),
        wall_time_s=to_float(row["wall_time_s"]),
    )


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-3) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise ValueError(f"{label} mismatch: expected {expected}, observed {actual}")


def validate_inputs(best: dict[tuple[str, str, str], AggregatedConfig], coverage: CoverageSummary) -> None:
    expected_configs = {
        ("cartpole", "dense", "DQN_Entropy"): ("alpha", 0.05, 0.01, 0.95, 200.000, 0.000, 1.000, 0.000),
        ("cartpole", "dense", "DQN_RND"): ("beta", 1.0, 0.01, 0.95, 198.344, 1.653, 0.920, 0.079),
        ("cartpole", "sparse", "DQN_Entropy"): ("alpha", 0.01, 0.0005, 0.95, 0.995, 0.005, 0.005, 0.005),
        ("cartpole", "sparse", "DQN_RND"): ("beta", 0.1, 0.0005, 0.95, 1.000, 0.000, 0.000, 0.000),
        ("mountaincar", "dense", "DQN_Entropy"): ("alpha", 0.1, 0.001, 0.99, -200.000, 0.000, 0.000, 0.000),
        ("mountaincar", "dense", "DQN_RND"): ("beta", 0.001, 0.001, 0.99, -200.000, 0.000, 0.000, 0.000),
        ("mountaincar", "sparse", "DQN_Entropy"): ("alpha", 0.1, 0.001, 0.99, -1.000, 0.000, 0.000, 0.000),
        ("mountaincar", "sparse", "DQN_RND"): ("beta", 0.01, 0.001, 0.99, -0.852, 0.148, 0.074, 0.074),
    }

    for key, expected in expected_configs.items():
        observed = best[key]
        expected_name, expected_value, expected_lr, expected_gamma, expected_reward, expected_reward_se, expected_success, expected_success_se = expected
        if observed.param_name != expected_name:
            raise ValueError(f"{key} parameter mismatch: expected {expected_name}, observed {observed.param_name}")
        assert_close(observed.param_value, expected_value, f"{key} parameter value")
        assert_close(observed.learning_rate, expected_lr, f"{key} learning rate")
        assert_close(observed.gamma, expected_gamma, f"{key} gamma")
        assert_close(observed.mean_reward, expected_reward, f"{key} mean reward")
        assert_close(observed.se_reward, expected_reward_se, f"{key} reward SE")
        assert_close(observed.mean_success, expected_success, f"{key} mean success")
        assert_close(observed.se_success, expected_success_se, f"{key} success SE")

    if coverage.included_groups != 58 or coverage.excluded_groups != 3:
        raise ValueError(
            "Coverage summary mismatch: expected 58 included groups and 3 excluded groups, "
            f"observed {coverage.included_groups} included and {coverage.excluded_groups} excluded."
        )


def require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def format_metric(mean: float, se: float) -> str:
    return f"{mean:.3f} +/- {se:.3f}"


def format_value(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-12):
        return f"{value:.1f}"
    return f"{value:g}"


def format_config(config: AggregatedConfig) -> str:
    return (
        f"{config.param_name}={format_value(config.param_value)}, "
        f"lr={config.learning_rate:g}, gamma={config.gamma:g}"
    )


def set_cell_text(cell, text: str, *, bold: bool = False, font_size: int = 9) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(font_size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)

    heading1 = doc.styles["Heading 1"]
    heading1.font.name = "Times New Roman"
    heading1.font.size = Pt(12)
    heading1.font.bold = True
    heading1.paragraph_format.space_before = Pt(10)
    heading1.paragraph_format.space_after = Pt(4)

    heading2 = doc.styles["Heading 2"]
    heading2.font.name = "Times New Roman"
    heading2.font.size = Pt(11)
    heading2.font.bold = True
    heading2.paragraph_format.space_before = Pt(8)
    heading2.paragraph_format.space_after = Pt(4)


def add_title_block(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Exploration Mechanisms for On- and Off-Policy RL Under Sparse Rewards")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)

    group_info = doc.add_paragraph()
    group_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = group_info.add_run("Group 8 Final Project Report\nUniversity of Ottawa - ELG5214 / CSI5340")
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)

    team = doc.add_paragraph()
    team.alignment = WD_ALIGN_PARAGRAPH.CENTER
    team_text = "\n".join(f"{role}: {name} ({email})" for role, name, email in TEAM_MEMBERS)
    run = team.add_run(team_text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)


def add_section(doc: Document, title: str, body: Iterable[str]) -> None:
    doc.add_heading(title, level=1)
    for paragraph_text in body:
        doc.add_paragraph(paragraph_text)


def add_bullet_list(doc: Document, items: Iterable[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_figure(doc: Document, path: Path, caption: str, *, width_inches: float = 5.8) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(path), width=Inches(width_inches))

    caption_paragraph = doc.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_paragraph.add_run(caption)
    caption_run.italic = True
    caption_run.font.name = "Times New Roman"
    caption_run.font.size = Pt(9)


def build_results_table(doc: Document, best: dict[tuple[str, str, str], AggregatedConfig]) -> None:
    order = [
        ("cartpole", "dense", "DQN_Entropy"),
        ("cartpole", "dense", "DQN_RND"),
        ("cartpole", "sparse", "DQN_Entropy"),
        ("cartpole", "sparse", "DQN_RND"),
        ("mountaincar", "dense", "DQN_Entropy"),
        ("mountaincar", "dense", "DQN_RND"),
        ("mountaincar", "sparse", "DQN_Entropy"),
        ("mountaincar", "sparse", "DQN_RND"),
    ]

    table = doc.add_table(rows=1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    headers = ["Condition", "Best complete config", "Mean reward +/- SE", "Success +/- SE", "Seeds"]
    for cell, label in zip(table.rows[0].cells, headers):
        set_cell_text(cell, label, bold=True)
        shade_cell(cell, "D9E2F3")

    for key in order:
        config = best[key]
        row = table.add_row().cells
        set_cell_text(row[0], f"{config.env_name.title()} {config.reward_type} / {config.algorithm}")
        set_cell_text(row[1], format_config(config))
        set_cell_text(row[2], format_metric(config.mean_reward, config.se_reward))
        set_cell_text(row[3], format_metric(config.mean_success, config.se_success))
        set_cell_text(row[4], str(config.n))

    note = doc.add_paragraph()
    note_run = note.add_run(
        "Table 1. Best complete Student A configuration per environment, reward type, and algorithm. "
        "Rows are selected by mean final greedy success, with mean final reward used as the tie-breaker. "
        "All values are averaged over 10 seeds from metrics/analysis/selected_scope_summary.csv."
    )
    note_run.italic = True
    note_run.font.name = "Times New Roman"
    note_run.font.size = Pt(9)


def create_report() -> Path:
    required_paths = [SUMMARY_CSV, COVERAGE_CSV]
    required_paths.extend(path for path, _ in BODY_FIGURES)
    required_paths.extend(path for path, _ in APPENDIX_FIGURES)
    required_paths.extend(
        [
            ROOT / "results" / "dqn_rnd_mc_8000" / "mountaincar_sparse" / "lr0.001_g0.99_b0.05_seed3_summary.csv",
            ROOT / "results" / "dqn_rnd_mc_8000" / "mountaincar_sparse" / "lr0.001_g0.99_b0.01_seed5_summary.csv",
        ]
    )
    require_files(required_paths)

    best = load_best_configs(SUMMARY_CSV)
    coverage = load_coverage_summary(COVERAGE_CSV)
    validate_inputs(best, coverage)

    standout_seed3 = load_standout_run(
        "seed 3 / beta 0.05",
        ROOT / "results" / "dqn_rnd_mc_8000" / "mountaincar_sparse" / "lr0.001_g0.99_b0.05_seed3_summary.csv",
    )
    standout_seed5 = load_standout_run(
        "seed 5 / beta 0.01",
        ROOT / "results" / "dqn_rnd_mc_8000" / "mountaincar_sparse" / "lr0.001_g0.99_b0.01_seed5_summary.csv",
    )
    assert_close(standout_seed3.mean_reward, 0.42, "standout seed 3 reward", tol=5e-3)
    assert_close(standout_seed3.success_rate, 0.71, "standout seed 3 success", tol=5e-3)
    assert_close(standout_seed5.mean_reward, 0.48, "standout seed 5 reward", tol=5e-3)
    assert_close(standout_seed5.success_rate, 0.74, "standout seed 5 success", tol=5e-3)

    doc = Document()
    configure_document(doc)
    add_title_block(doc)

    add_section(
        doc,
        "1. Abstract",
        [
            "Group 8 investigated how entropy regularization, Random Network Distillation (RND), and Intrinsic "
            "Curiosity Modules (ICM) influence exploration in on-policy and off-policy reinforcement learning under "
            "dense and sparse reward settings. The project was framed around PPO and DQN agents in Gymnax "
            "CartPole-v1 and MountainCar-v0, with scripted sweeps, fixed seeds, and saved metrics for reproducibility. "
            "This document preserves that full group framing, but the quantitative evidence reported here is limited to "
            "the Student A subset currently available in the repository: DQN with entropy regularization, DQN with RND, "
            "and longer 8000-episode MountainCar follow-up reruns. Within that subset, CartPole dense is the clearest "
            "success case, with DQN entropy reaching perfect final greedy performance and DQN RND closely following. "
            "Sparse CartPole is more subtle: the binary sparse training reward looks strong, but final greedy success "
            "stays near zero, so the condition should not be treated as solved from the Student A evidence alone. "
            "MountainCar remains difficult in both dense and sparse settings, although the longer sparse-reward RND "
            "reruns produce two isolated successes and a small aggregate improvement over entropy. Overall, the Student A "
            "results suggest that simple entropy regularization is sufficient for the easy dense control setting, while "
            "RND can occasionally help on the hardest sparse task but does not deliver a stable or consistently "
            "reliable improvement.",
        ],
    )

    add_section(
        doc,
        "2. Introduction and Motivation",
        [
            "Sparse rewards remain one of the central obstacles in deep reinforcement learning. When the environment "
            "withholds task feedback for long stretches of interaction, the agent must discover useful behavior through "
            "exploration alone. This makes optimization unstable, slows down credit assignment, and can leave otherwise "
            "strong learning algorithms trapped in uninformative regions of the state space.",
            "The full Group 8 project was motivated by a practical question: when should a practitioner prefer a "
            "simple exploration mechanism such as entropy regularization over curiosity-driven bonuses such as RND or "
            "ICM, and does that answer depend on whether the learning algorithm is on-policy or off-policy? The project "
            "proposal framed this comparison around PPO [2] and DQN [1] in Gymnax [5] CartPole-v1 and MountainCar-v0, "
            "each evaluated under both dense and sparse reward variants.",
            "Those questions matter because the same exploration rule can behave very differently across environments. "
            "CartPole is comparatively forgiving and dense-reward solutions are easy to discover, while MountainCar "
            "requires the agent to learn a counter-intuitive swing-up strategy before it can ever reach the goal. Under "
            "sparse rewards, that discovery problem becomes much harder and provides a useful test bed for exploration "
            "mechanisms that are supposed to reward novelty or uncertainty.",
            "This report therefore keeps the broader group framing from the proposal, but it also makes an important "
            "scope distinction. The repository currently contains complete Student A quantitative outputs for DQN "
            "entropy and DQN RND, including a focused MountainCar follow-up rerun campaign. The narrative below uses "
            "the full project motivation and planned methodology, while all tables, figures, and performance claims are "
            "restricted to the Student A artifacts that are actually present.",
        ],
    )
    doc.add_heading("2.1 Research Questions", level=2)
    add_bullet_list(
        doc,
        [
            "Under sparse rewards, do curiosity-driven intrinsic rewards improve practical exploration more than a simple entropy bonus?",
            "How do reward density and environment difficulty change the apparent value of entropy regularization versus RND?",
            "Within the Student A subset, what can DQN-based evidence say about dense CartPole, sparse CartPole, and the much harder MountainCar conditions?",
        ],
    )

    add_section(
        doc,
        "3. Related Work",
        [
            "DQN [1] established deep Q-learning as a strong off-policy baseline for discrete-action control by combining "
            "function approximation with replay and target networks. PPO [2] later became a widely used on-policy "
            "baseline because its clipped objective offers a practical compromise between stable policy updates and good "
            "empirical performance. Together, those two algorithms define the on-policy/off-policy contrast at the core "
            "of this project's framing.",
            "Curiosity-driven exploration methods attempt to replace some of the missing task signal in sparse-reward "
            "settings. Pathak et al. [3] proposed the Intrinsic Curiosity Module, where forward-model prediction error "
            "in a learned feature space becomes an intrinsic reward. Burda et al. [4] simplified that idea with Random "
            "Network Distillation, which measures novelty through the prediction error of a trainable network against a "
            "fixed random target.",
            "Entropy regularization plays a different role. Rather than rewarding novelty directly, it discourages the "
            "policy or action-value distribution from collapsing too early. In practice, it is cheaper than curiosity "
            "modules because it does not require an additional dynamics model or a separate predictor-target pair, but "
            "it also provides a weaker exploration signal. That tradeoff is exactly why it is useful as a baseline in "
            "this comparison.",
            "Finally, Gymnax [5] provides JAX-native environments and fast functional rollouts, making it a natural "
            "fit for the project's emphasis on reproducibility and clean experimentation. The gap addressed by the full "
            "Group 8 design is not the invention of a new exploration method, but a controlled comparison of multiple "
            "mechanisms across reward densities and algorithm families. The Student A subset reported here only covers "
            "the DQN side of that design, so the present conclusions should be read as partial rather than project-wide.",
        ],
    )

    add_section(
        doc,
        "4. Methodology",
        [
            "Environment and task design. The project uses Gymnax CartPole-v1 and MountainCar-v0 [5]. CartPole uses a "
            "4-dimensional observation and two actions, while MountainCar uses a 2-dimensional observation and three "
            "actions. The dense variants preserve the original Gymnax rewards. The sparse wrappers implemented in "
            "src/envs/cartpole_env.py and src/envs/mountaincar_env.py replace those signals with end-of-episode task "
            "rewards: CartPole sparse returns 1 only when the pole survives to the time limit, and MountainCar sparse "
            "returns 1 on goal reach and -1 on timeout.",
            "Planned algorithm suite. The full Group 8 proposal covered vanilla PPO, PPO plus entropy, PPO plus RND, "
            "PPO plus ICM, vanilla DQN, DQN plus entropy, DQN plus RND, and DQN plus ICM. Random and heuristic "
            "baselines were also part of the planned controls. However, the quantitative sections of this report only "
            "populate the Student A subset currently present in the repository: DQN entropy, DQN RND, and the "
            "MountainCar 8000-episode reruns built on those implementations.",
            "Student A model and update rules. The shared DQN backbone in src/networks/q_network.py is a two-hidden-layer "
            "MLP with 64 ReLU units per layer. DQN entropy keeps the standard replay-buffer and target-network DQN "
            "structure, but optimizes TD loss minus alpha times the entropy of softmax(Q(s, :)). DQN RND keeps the same "
            "Q-network and TD loss, augments the extrinsic reward with beta times the predictor-target mean squared "
            "error, and trains a separate RND predictor against a fixed random target network from src/networks/rnd_networks.py. "
            "The RND networks also use two 64-unit ReLU hidden layers, followed by a 32-dimensional embedding.",
            "Hyperparameter search and follow-up reruns. The group proposal defined a grid over learning rate, discount "
            "factor, seeds, and mechanism-specific coefficients. The Student A CartPole sweeps in the stored results "
            "cover 10 seeds, learning rates in {0.01, 0.001, 0.0005}, gammas in {0.99, 0.95}, alpha in {0.01, 0.05}, "
            "and beta in {0.1, 1.0}. The later MountainCar follow-up reruns use dedicated configs with 8000 episodes, "
            "lr=0.001, gamma=0.99, alpha in {0.1, 0.2}, beta in {0.001, 0.01, 0.05}, faster epsilon decay, and 8 "
            "updates per episode. Those reruns were added because the initial MountainCar results showed persistent failure.",
            "Metrics, compute, and reproducibility. Episode-level metrics and final summaries are logged through "
            "src/utils/reusable.py. Final performance comes from greedy evaluation over 100 episodes in the Student A "
            "test entry points. During training, checkpoint evaluation is descriptive rather than decisive: DQN entropy "
            "logs every 50 episodes with 25 greedy evaluation episodes, while DQN RND logs every 100 episodes with 10 "
            "evaluation episodes. The analysis pipeline in notebooks/results_analysis_helpers.py filters the Student A "
            "scope, writes merged CSVs under metrics/analysis, and generates the comparison plots in visualizations/analysis. "
            "The committed summaries show that Student A runs were executed under both CPU and CUDA backends, so wall-clock "
            "times are informative but not a clean single-machine benchmark. Reproducibility is supported through "
            "requirements.txt, environment.yml, README.md, scripts/run_sweep.sh, and scripts/run_studentA_mountaincar_8000.sh.",
        ],
    )

    add_section(
        doc,
        "5. Experimental Results",
        [
            "Reported quantitative scope. The merged Student A analysis scope contains 58 complete hyperparameter groups "
            "with 10 seeds each and excludes 3 incomplete groups according to metrics/analysis/config_coverage_audit.csv. "
            "The excluded groups are dqn_entropy/cartpole_dense/lr0.1_g0.99_a0.01, "
            "dqn_entropy_mc_8000/mountaincar_dense/lr0.001_g0.99_a0.05, and "
            "dqn_entropy_mc_8000/mountaincar_sparse/lr0.001_g0.99_a0.05. The notebook helper then produces 28 figures "
            "under visualizations/analysis. In the discussion below, the summary table reports the single best complete "
            "configuration for each environment, reward type, and Student A algorithm, while the cohort figures average "
            "over all included hyperparameter groups within the selected scope.",
        ],
    )
    build_results_table(doc, best)

    cartpole_dense_entropy = best[("cartpole", "dense", "DQN_Entropy")]
    cartpole_dense_rnd = best[("cartpole", "dense", "DQN_RND")]
    cartpole_sparse_entropy = best[("cartpole", "sparse", "DQN_Entropy")]
    cartpole_sparse_rnd = best[("cartpole", "sparse", "DQN_RND")]
    mountaincar_dense_entropy = best[("mountaincar", "dense", "DQN_Entropy")]
    mountaincar_dense_rnd = best[("mountaincar", "dense", "DQN_RND")]
    mountaincar_sparse_entropy = best[("mountaincar", "sparse", "DQN_Entropy")]
    mountaincar_sparse_rnd = best[("mountaincar", "sparse", "DQN_RND")]

    doc.add_heading("5.1 CartPole", level=2)
    doc.add_paragraph(
        "CartPole dense is the strongest Student A success case. The best DQN entropy configuration "
        f"({format_config(cartpole_dense_entropy)}) achieves {format_metric(cartpole_dense_entropy.mean_reward, cartpole_dense_entropy.se_reward)} "
        f"final reward and {format_metric(cartpole_dense_entropy.mean_success, cartpole_dense_entropy.se_success)} final success. "
        f"The best DQN RND configuration ({format_config(cartpole_dense_rnd)}) reaches "
        f"{format_metric(cartpole_dense_rnd.mean_reward, cartpole_dense_rnd.se_reward)} reward and "
        f"{format_metric(cartpole_dense_rnd.mean_success, cartpole_dense_rnd.se_success)} success. The best-config table therefore "
        "shows that both mechanisms can perform well on dense CartPole, although entropy is the cleaner solver in this subset."
    )
    add_figure(doc, *BODY_FIGURES[0])

    doc.add_paragraph(
        "CartPole sparse needs a more careful interpretation. The best Student A final summaries are "
        f"{format_metric(cartpole_sparse_entropy.mean_reward, cartpole_sparse_entropy.se_reward)} reward with "
        f"{format_metric(cartpole_sparse_entropy.mean_success, cartpole_sparse_entropy.se_success)} success for DQN entropy, and "
        f"{format_metric(cartpole_sparse_rnd.mean_reward, cartpole_sparse_rnd.se_reward)} reward with "
        f"{format_metric(cartpole_sparse_rnd.mean_success, cartpole_sparse_rnd.se_success)} success for DQN RND. "
        "Because the sparse reward is binary and tied to the terminal event, raw reward trajectories can appear strong "
        "even when the final greedy policy is poor. The Student A evidence therefore does not support describing sparse "
        "CartPole as solved."
    )
    add_figure(doc, *BODY_FIGURES[1])

    doc.add_heading("5.2 MountainCar 8000-Episode Follow-up Reruns", level=2)
    doc.add_paragraph(
        "MountainCar dense remains unsolved in the Student A reruns. The best DQN entropy configuration "
        f"({format_config(mountaincar_dense_entropy)}) ends at {format_metric(mountaincar_dense_entropy.mean_reward, mountaincar_dense_entropy.se_reward)} "
        f"reward and {format_metric(mountaincar_dense_entropy.mean_success, mountaincar_dense_entropy.se_success)} success. "
        f"The best DQN RND configuration ({format_config(mountaincar_dense_rnd)}) is identical at the failure floor. "
        "Longer training and additional updates per episode were therefore insufficient to make either Student A DQN "
        "variant discover the MountainCar swing-up policy under dense rewards."
    )
    add_figure(doc, *BODY_FIGURES[2])

    doc.add_paragraph(
        "MountainCar sparse is the only condition where RND shows a meaningful aggregate advantage over entropy in the "
        "Student A subset, but the improvement is still small and unstable. DQN entropy remains at "
        f"{format_metric(mountaincar_sparse_entropy.mean_reward, mountaincar_sparse_entropy.se_reward)} reward and "
        f"{format_metric(mountaincar_sparse_entropy.mean_success, mountaincar_sparse_entropy.se_success)} success. "
        f"DQN RND improves to {format_metric(mountaincar_sparse_rnd.mean_reward, mountaincar_sparse_rnd.se_reward)} reward and "
        f"{format_metric(mountaincar_sparse_rnd.mean_success, mountaincar_sparse_rnd.se_success)} success with "
        f"{format_config(mountaincar_sparse_rnd)}. Two runs account for most of that gain: "
        f"{standout_seed3.label} reaches final reward {standout_seed3.mean_reward:.2f} with success {standout_seed3.success_rate:.2f}, "
        f"and {standout_seed5.label} reaches final reward {standout_seed5.mean_reward:.2f} with success {standout_seed5.success_rate:.2f}. "
        "Those standout runs are encouraging, but they are not representative of the full 10-seed distribution."
    )
    add_figure(doc, *BODY_FIGURES[3])
    add_figure(doc, *BODY_FIGURES[4])

    add_section(
        doc,
        "6. Analysis and Discussion",
        [
            "The Student A subset supports three main takeaways. First, entropy regularization is fully adequate for the "
            "easy dense-reward control task. Its best CartPole dense configuration reaches the task ceiling and the "
            "aggregate cohort curve rises quickly even when weaker hyperparameter settings are averaged in. RND also "
            "works on dense CartPole, but it introduces extra moving parts without improving the final result.",
            "Second, sparse-reward training curves can be misleading if they are interpreted without greedy evaluation. "
            "In CartPole sparse, the binary training reward saturates near 1.0 because occasional successful episodes are "
            "enough to lift the mean reward. However, the checkpoint evaluation plot and the final summary table both show "
            "that those successes do not translate into robust greedy policies. This is exactly why the final 100-episode "
            "greedy evaluation is a better indicator of solved behavior than raw sparse reward alone.",
            "Third, MountainCar confirms the central difficulty of sparse exploration. The dense variant already requires "
            "counter-intuitive momentum building, and the sparse variant removes almost all external learning signal. "
            "Entropy regularization cannot reliably overcome that barrier in the Student A DQN implementation. RND does "
            "help in a limited sense: it produces the only non-zero aggregate success in MountainCar sparse and yields two "
            "clear standout runs. Still, those gains are concentrated in very few seeds, so the evidence points to a high-variance "
            "effect rather than a stable mechanism-level win.",
            "These results only partially answer the full group research questions. Within the Student A DQN subset, the "
            "evidence suggests that entropy regularization is a strong low-cost choice for easier dense tasks, while RND "
            "is more appropriate for extremely sparse tasks where novelty bonuses may occasionally unlock discovery. What "
            "the current repository cannot yet answer is how those trends compare against PPO or ICM, because those "
            "quantitative result sets are not available here.",
        ],
    )

    add_section(
        doc,
        "7. Limitations, Risks, and Future Work",
        [
            "The largest limitation is scope coverage. This report is framed as the full Group 8 project, but only Student "
            "A numerical results are currently included. That means the writeup can describe the intended full algorithm "
            "suite, yet it cannot make direct quantitative claims about PPO, ICM, Student B's DQN baseline and ICM runs, "
            "or the full group comparison that the proposal originally targeted.",
            "A second limitation is that the completed Student A runs are not perfectly matched by training budget. The "
            "standard DQN entropy sweeps, the standard DQN RND sweeps, and the later MountainCar reruns use different "
            "episode counts and checkpoint frequencies because the reruns were added after earlier failures. The reported "
            "results are therefore faithful to the completed repository state, but they are not a fully balanced matched-compute "
            "study in the strictest sense.",
            "A third limitation is incomplete coverage and instability. The notebook helper excludes 3 incomplete config "
            "groups, and MountainCar sparse improvements are driven by only a few seeds. That combination makes it risky "
            "to over-generalize from the standout runs. In addition, the summary CSVs record both CPU and CUDA backends, "
            "so wall-clock times should be interpreted as mixed-hardware observations rather than a tightly controlled benchmark.",
            "Future work should start by finishing the missing teammate result sets and rerunning the full comparison with "
            "uniform budgets and identical evaluation schedules. Within the Student A branch specifically, the next step "
            "should be a denser MountainCar sparse sweep around beta in the 0.01-0.05 range, plus additional seeds to test "
            "whether the current standout runs are repeatable or merely lucky. It would also be useful to add state-coverage "
            "diagnostics, prioritized replay, and direct comparisons against PPO and ICM once those results are available.",
        ],
    )

    add_section(
        doc,
        "8. Conclusion",
        [
            "The full Group 8 project asked how different exploration mechanisms behave across dense and sparse rewards, "
            "and whether those behaviors depend on the underlying reinforcement learning algorithm. The Student A subset "
            "reported here provides a partial but still informative answer on the DQN side of that question.",
            "For dense CartPole, DQN entropy is the strongest and simplest solution in the available data, while DQN RND "
            "also performs well but offers no compelling advantage. For sparse CartPole, raw sparse reward can exaggerate "
            "progress, and the final greedy evaluations show that neither Student A DQN variant should be treated as a "
            "robust solver. For MountainCar, entropy regularization fails in both dense and sparse forms, while RND shows "
            "only a small aggregate sparse-reward benefit driven by a handful of successful seeds.",
            "Taken together, these results suggest that curiosity bonuses may help only when the task is hard enough to "
            "justify their extra complexity, and even then the benefit can remain fragile. The repository already contains "
            "the code, scripts, configs, and analysis assets needed to reproduce the Student A findings, and it also "
            "provides a foundation for completing the missing group-wide comparison in future work.",
        ],
    )

    doc.add_page_break()
    doc.add_heading("9. References", level=1)
    for reference in REFERENCES:
        doc.add_paragraph(reference)

    doc.add_page_break()
    add_section(
        doc,
        "Appendix A. Additional Figures",
        [
            "This appendix collects supporting Student A figures from visualizations/analysis that were not necessary in "
            "the main body but are useful for inspecting mechanism-specific trends.",
        ],
    )
    for figure_path, caption in APPENDIX_FIGURES:
        add_figure(doc, figure_path, caption, width_inches=5.3)

    final_note = doc.add_paragraph()
    final_run = final_note.add_run(
        "Repository and reproducibility note: the completed codebase, environment specification, run scripts, and stored "
        f"results are available under {GITHUB_URL}."
    )
    final_run.font.name = "Times New Roman"
    final_run.font.size = Pt(10)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> None:
    output = create_report()
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
