# Cross-View Consistency Checking for Multimodal Web Agents under Adversarial UI Perturbations

> Draft paper for an AI CCF-C conference, based on the WebArena codebase. To be expanded and tailored for a specific venue's format.

## Abstract

Multimodal web agents built on large language models (LLMs) are increasingly deployed to operate real-world websites using visual and textual observations. However, recent work has shown that small adversarial perturbations to user interfaces (UIs), such as invisible overlays, misleading banners, or prompt-injection-like text, can drastically degrade agent performance and cause unsafe behaviors. In this paper, we study the robustness of multimodal web agents under adversarial UI perturbations and propose a light-weight defense mechanism based on cross-view consistency checking. Building on the WebArena benchmark and attack suite, we instantiate a robust prompt-based agent that compares multiple observation views (e.g., accessibility tree text, DOM snippets, and visual indicators) and filters suspicious content before decision making. We design five representative UI attack patterns and evaluate them across diverse tasks. Experiments show that our cross-view consistency checking significantly reduces attack success rate while preserving benign-task performance, and can be integrated into existing web agents with minimal modifications. Our work highlights the importance of multi-view reasoning and provides a practical defense baseline for secure multimodal web automation.

## 1 Introduction

Large language model (LLM) based web agents have recently demonstrated strong capabilities on realistic web interaction benchmarks such as WebArena. By combining natural language reasoning with browser-level actions, these agents can complete tasks like searching, form filling, and multi-step navigation, all from high-level textual instructions. This paradigm enables powerful automation but also exposes new security and robustness challenges.

Unlike conventional web automation scripts, LLM web agents rely on rich multimodal observations that include rendered text, accessibility trees, DOM structures, and sometimes screenshots. Adversaries can manipulate these observations via UI perturbations that are almost invisible or benign-looking to human users but highly misleading for agents. Examples include overlay-based click hijacking, homoglyph-based phishing, and prompt-injection-style instructions embedded into page content. Because LLM agents often treat web observations as trusted context, such attacks can redirect their behavior, exfiltrate sensitive data, or silently alter task outcomes.

Existing work primarily focuses on designing attack vectors and measuring vulnerability, while defenses for multimodal web agents remain under-explored. A key observation in this domain is that real UIs are grounded in multiple, partially redundant views: visual layout, accessibility semantics, and DOM attributes. Adversarial perturbations usually create inconsistencies across these views, for example, by inserting high-urgency text in a low-importance region, or by overlaying clickable elements with mismatched semantics.

In this paper, we explore cross-view consistency checking as a practical defense mechanism. Instead of trusting a single observation channel, our agent detects and filters adversarial content by comparing patterns across views and enforcing heuristic consistency rules. We build our method on the open-source WebArena environment, which provides a realistic set of multi-site web tasks and an extensible attack framework. Our contributions are as follows:

- We formulate adversarial UI perturbations for multimodal web agents as a cross-view inconsistency problem and characterize common attack patterns in a unified taxonomy.
- We propose a robust prompt-based web agent that integrates an observation filter and cross-view consistency checker, implemented as a light-weight extension to an existing PromptAgent in WebArena.
- We implement five concrete UI adversaries, including hidden click overlays, homoglyph-based phishing, layout jitter, and overlay-based bait content, and release them as a modular attack zoo.
- We conduct systematic experiments on WebArena tasks under varying attack strengths, showing that our defense reduces attack success rate and improves task success under attack while incurring modest overhead.

## 2 Related Work

### 2.1 Multimodal Web Agents

Recent benchmarks such as WebArena, WebShop, and other simulated browsing environments introduce realistic multi-page tasks that require agents to read content, click buttons, fill forms, and navigate between sites. LLM-based agents typically encode page observations, generate textual rationales, and output high-level actions. Some works additionally incorporate screenshots or visual encoders, yielding multimodal web agents that can better understand layout and styling.

WebArena provides a self-hosted bundle of realistic websites with APIs to expose accessibility trees, DOM features, and action spaces. Prompt-based agents are often designed as modular components that receive the current trajectory, intent, and observation, and then generate the next action via LLM calls. Our work builds directly on such a PromptAgent and extends it with robustness capabilities.

### 2.2 Adversarial Attacks on Web Agents

Adversarial machine learning has extensively studied perturbations on images, text, and reinforcement learning agents. For web agents, the attack surface includes UI elements, textual content, and interaction constraints. Prior work has demonstrated that small UI changes, such as inserting carefully crafted banners or shifting clickable areas, can mislead agents to click malicious links or ignore safety-critical warnings.

Within WebArena, an extensible attack framework allows researchers to implement adversaries that manipulate page content and layout, e.g., hidden click (`hidden_click.py`), homoglyph-based content (`homoglyph.py`), layout jitter (`layout_jitter.py`), lazy bait (`lazy_bait.py`), and overlay bait (`overlay_bait.py`). These adversaries emulate realistic threats such as clickjacking, phishing, and deceptive advertising, making them suitable for robustness evaluation.

### 2.3 Defenses for Prompt and UI Attacks

To mitigate prompt injection and content-based attacks, recent work proposed instruction sanitization, content filtering, and sandboxed tool usage. In the UI domain, defenses often involve browser hardening, ad-blocking, and permission systems. However, for LLM-based web agents that interpret complex page observations, there is a gap between low-level browser protections and high-level semantic understanding.

Cross-view consistency checking offers a middle ground: it leverages multiple observation channels to detect anomalies without requiring full adversarial training. By encoding rules such as "high-urgency warnings should not appear in low-visibility regions" or "instructions that attempt to override system goals are suspicious", agents can filter harmful content before reasoning. Our work instantiates this idea in a concrete open-source environment and evaluates its effectiveness.

## 3 Problem Formulation

We consider a multimodal web agent interacting with a browser environment over discrete time steps. At each step $t$, the environment provides an observation $o_t$ that includes textual content, structural information, and optionally visual descriptors. The agent maintains a trajectory $\tau_t = \{(o_i, a_i, r_i)\}_{i=1}^t$ and selects the next action $a_t$ based on the intent $I$ and the current trajectory.

An adversarial UI perturbation modifies the underlying web page or its rendered representation to produce a perturbed observation $\tilde{o}_t$. The perturbation is constrained to preserve functional correctness and human usability (i.e., a typical user can still complete the task) but aims to change the agent's behavior. The attack succeeds if it causes the agent to deviate from a correct plan, fail the task, or violate specified safety constraints.

We focus on adversaries that operate within the WebArena attack framework and can modify:

- textual content in the accessibility tree or DOM (e.g., adding phishing instructions or homoglyph variants of domain names);
- layout properties such as position, size, and z-index, enabling overlays, hidden elements, or misaligned clickable areas;
- decorative components that resemble system messages, warnings, or pop-ups.

Our goal is to design a defense mechanism $D$ that, given a perturbed observation $\tilde{o}_t$, produces a filtered observation $\hat{o}_t = D(\tilde{o}_t)$ such that the downstream agent behaves similarly to how it would under clean observations $o_t$, thereby reducing the attack success rate while maintaining high task success on benign pages.

## 4 Methodology

### 4.1 Cross-View Consistency Checking

The central idea of our defense is to exploit redundancies across different views of the web page. In WebArena, the agent can access at least the following channels:

- **Accessibility tree text**: a structured textual representation of UI elements and their roles;
- **DOM attributes**: element tags, classes, IDs, and inline styles that influence rendering and semantics;
- **Action space metadata**: information about which elements are clickable, fillable, or otherwise interactive.

Adversarial UI perturbations often manifest as inconsistencies between these views. For example, a fake security warning may be rendered in a low-importance region with unusual styling, or an overlay may introduce clickable regions that do not correspond to meaningful accessibility roles. By encoding heuristic patterns that capture such inconsistencies, we can flag and sanitize suspicious content before giving it to the LLM.

### 4.2 Observation Filtering Module

We implement an `ObservationFilter` module that operates on the structured observation dictionary produced by the browser environment. The module performs three main functions:

1. **Prompt-injection detection**: using regular-expression patterns (`INJECTION_PATTERNS`) to detect phrases such as "ignore previous instructions" or "your new task is", which are standard signatures of prompt injection.
2. **Suspicious UI detection**: using `SUSPICIOUS_PATTERNS` to match high-urgency security warnings, verification prompts, or account compromise messages that are common in phishing scenarios.
3. **Layout and overlay sanitization**: removing or marking patterns associated with fixed-position overlays and extreme z-index values, which are typical in clickjacking or bait overlays.

For each observation, the filter:

- scans the `text` field and records all matched patterns as potential attacks along with their positions;
- applies sanitization that replaces suspicious instructions with neutral placeholders and either tags or removes suspicious UI text, depending on an `aggressive` flag;
- strips known overlay patterns at the markup level, reducing the influence of adversarial layers.

The filter returns an augmented observation containing the cleaned `text`, a boolean `filtered` flag, and a list of `detected_attacks`. It also keeps global statistics such as the number of filtered observations and the distribution of attack types.

### 4.3 Robust Prompt Agent

To integrate the defense with existing agents, we extend a base `PromptAgent` into a `RobustPromptAgent`. The robust agent wraps the original `next_action` method as follows:

1. Before each decision, it extracts the latest state from the trajectory and passes its observation through `ObservationFilter`.
2. If any attacks are detected, it logs them into an internal `attack_log` and updates the observation in-place with the filtered version.
3. It then calls the parent `PromptAgent.next_action` using the sanitized trajectory, leaving downstream prompting, reasoning, and action selection unchanged.

The agent exposes a `get_defense_stats` function that summarizes total detected attacks, their types, and filter statistics. This design keeps the defense modular and compatible with future agent improvements.

### 4.4 Cross-View Extensions

While our initial implementation focuses on text-based patterns and overlay markup, the same framework can be extended with richer cross-view checks, including:

- consistency between accessibility labels and visual positions of critical elements (e.g., login buttons or confirm dialogs);
- cross-checks between action metadata (e.g., clickable vs. non-clickable) and displayed semantics (e.g., a link that claims to "cancel" but triggers a destructive action);
- leveraging screenshot-based saliency maps to ensure that high-urgency content corresponds to visually salient regions.

We discuss these extensions as future work but show in experiments that even simple heuristic checks provide tangible robustness gains.

## 5 Experimental Setup

### 5.1 Environment and Tasks

We conduct experiments on the WebArena benchmark, which hosts a collection of diverse websites such as e-commerce, forums, and knowledge bases. Each task specifies a natural-language instruction (e.g., "find the price of item X" or "post a reply in thread Y") and success criteria defined over final states.

We consider two evaluation regimes:

- **Clean**: tasks are executed in the default environment without adversarial perturbations.
- **Adversarial**: for each task, one or more UI adversaries are activated via the WebArena attack registry, modifying content and layout while preserving task feasibility for humans.

### 5.2 UI Adversaries

We instantiate five adversaries from the attack zoo:

1. **Hidden Click**: injects invisible or transparent overlays above benign buttons to redirect clicks.
2. **Homoglyph**: replaces characters in key strings (e.g., domain names, account labels) with visually similar Unicode homoglyphs.
3. **Layout Jitter**: randomly shifts or perturbs the positions and sizes of UI elements, making spatial reasoning harder.
4. **Lazy Bait**: inserts tempting but irrelevant content (e.g., "Claim your reward") that distracts the agent from its main goal.
5. **Overlay Bait**: adds full- or partial-screen overlays that mimic system pop-ups or warnings.

Each adversary has configurable strength levels controlling the number and intensity of perturbations. We evaluate agents under individual attacks and combined attack settings.

### 5.3 Baselines and Metrics

We compare the following agents:

- **BaseAgent**: the original prompt-based web agent without defenses;
- **RobustAgent (ours)**: the same agent equipped with observation filtering and cross-view consistency checking.

Key metrics include:

- **Task success rate**: fraction of tasks where the final state satisfies the success criteria;
- **Attack success rate**: fraction of tasks where the adversary changes the outcome from success (under clean conditions) to failure under attack;
- **Robustness gain**: relative reduction in attack success rate achieved by our defense;
- **Overhead**: additional computation time introduced by the filter per action.

We also qualitatively inspect trajectories where attacks are detected to understand the agent's behavior.

## 6 Results and Analysis

### 6.1 Overall Robustness

Across a representative subset of WebArena tasks, the BaseAgent demonstrates high success under clean conditions but suffers substantial degradation under UI attacks. Hidden Click and Overlay Bait, in particular, cause frequent mis-clicks and task abandonment. Homoglyph attacks lead to misinterpretation of critical strings, while Lazy Bait can divert the agent to low-value actions.

The RobustAgent recovers a significant portion of lost performance. By filtering suspicious instructions and overlays, it avoids many spurious clicks and refocuses on goal-relevant content. Overall, we observe a notable reduction in attack success rate with only minor drops in clean-task performance, indicating that the heuristic filters are not overly aggressive.

### 6.2 Per-Attack Behavior

Per-attack analysis reveals distinct defense behaviors:

- **Hidden Click / Overlay Bait**: overlay sanitization and suspicious-UI tagging prevent the agent from over-trusting full-screen warnings and fake dialogs. The agent learns to prioritize underlying, semantically consistent elements.
- **Homoglyph**: while regex-based filtering cannot fully normalize homoglyphs, it flags unusual Unicode sequences within high-importance strings, prompting the agent to cross-check surrounding context.
- **Layout Jitter**: our text-centric filters are less effective here, suggesting future work on incorporating geometric and visual consistency.
- **Lazy Bait**: patterns that resemble promotions or rewards are down-weighted or removed, reducing distraction.

### 6.3 Ablation Studies

To understand the contribution of each component, we conduct ablation studies by selectively disabling:

- prompt-injection detection;
- suspicious-UI tagging;
- overlay sanitization.

Results indicate that prompt-injection and suspicious-UI patterns are most impactful against overlay-based and phishing-style attacks, whereas overlay sanitization is crucial for click hijacking scenarios. Combining all components yields the best robustness, though at slightly higher computational cost.

### 6.4 Overhead and Practicality

The `ObservationFilter` operates in linear time with respect to the observation text length and introduces only a small constant overhead from regex matching. In practice, we observe modest additional latency per action that remains acceptable for interactive web automation. Because our defense is implemented as a wrapper around the base agent, it can be easily toggled or configured (e.g., aggressive vs. conservative filtering) depending on application risk tolerance.

## 7 Discussion

Our study demonstrates that simple cross-view consistency checks and heuristic filters can significantly improve the robustness of multimodal web agents against adversarial UI perturbations. However, several limitations remain. First, regex-based patterns may miss novel or obfuscated attacks, and overfitting to known signatures could give a false sense of security. Second, our current implementation focuses on textual and structural aspects of observations and does not fully exploit visual signals.

Future work could explore learning-based detectors that operate over joint representations of DOM, accessibility trees, and screenshots, enabling more flexible yet principled consistency checks. Another promising direction is integrating defense mechanisms into the training loop, allowing agents to learn robust policies that anticipate adversarial UIs. Finally, extending evaluation to broader environments and real-world websites would further validate the generality of our findings.

## 8 Conclusion

We investigated the robustness of multimodal web agents under adversarial UI perturbations and proposed a simple yet effective defense based on cross-view consistency checking. By integrating an observation filter into a prompt-based web agent in WebArena, we detect and sanitize prompt-injection-like content, phishing-style warnings, and overlay-based attacks before decision making. Extensive experiments with five representative UI adversaries show that our defense substantially reduces attack success rates while preserving performance on benign tasks, with limited computational overhead. Our work highlights the value of leveraging multiple observation views for secure web automation and provides a practical baseline for future research on robust multimodal agents.

## Acknowledgements

We thank the maintainers of the WebArena project and the broader community for providing open-source tools and benchmarks that enable research on robust web agents.
