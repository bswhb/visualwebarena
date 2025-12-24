这是一份基于您的原稿《Cross-View Consistency Checking for Multimodal Web Agents under Adversarial UI Perturbations》1 修改后的论文大纲和重写建议。  
该版本专门针对 **KSEM 2026 (International Conference on Knowledge Science, Engineering and Management)** 的偏好进行了调整。KSEM 侧重于**知识科学、知识工程与系统管理** 2, 3，因此不仅要强调“防御攻击”，更要将“一致性检查”包装为一种**多模态知识验证与管理机制**。

### KSEM 2026 投稿论文生成方案

**论文题目 (建议修改):**Ensuring Knowledge Integrity in Multimodal Web Agents via Triple-View Semantic Consistency Verification*(通过三视图语义一致性验证保障多模态Web智能体的知识完整性)*  
**投稿轨道:** Knowledge Engineering / AI Security 3**格式要求:** Springer LNCS (Lecture Notes in Computer Science) 4, 5**篇幅建议:** 长文 12-15页 (LNCS 单栏格式通常比 IEEE 双栏格式需要更多文字描述) 5**盲审要求:** 必须严格去匿名，移除作者信息、机构及GitHub链接 4。

### 论文结构与内容重写 (基于 ICPR 草稿)

#### Abstract (摘要)

* **重写重点**：将对抗性 UI 扰动定义为“知识获取过程中的噪声注入”，将 TVSC 定义为“知识验证框架”。  
* **草稿内容**：  
* The advent of Multimodal Large Language Models (MLLMs) has catalyzed... autonomous web agents... 1  
* **KSEM 风格建议**：  
* Multimodal Web Agents act as autonomous knowledge workers, perceiving and reasoning over complex web environments. However, their reliability is compromised by **adversarial UI perturbations**, which create a schism between the agent's visual perception and underlying structural knowledge (DOM). This paper addresses this **knowledge alignment problem** by proposing **Triple-View Semantic Consistency (TVSC)**. TVSC functions as a **knowledge verification mechanism**, enforcing semantic coherence across visual, structural (Accessibility Tree), and OCR-extracted representations. By modeling the consistency checking as a multi-view knowledge fusion task, we demonstrate that TVSC improves agent success rates by over 142% under hybrid attacks, establishing a rigorous engineering standard for robust agentic systems.

#### 1\. Introduction (引言)

1. **第一段 (背景)**：引用 WebArena 6 和 VisualWebArena 7，强调智能体正在从简单的自动化工具转变为复杂的知识处理系统。  
2. **第二段 (问题 \- 知识工程视角)**：  
3. **原意**：攻击者修改 DOM 或视觉层误导智能体 8。  
4. **KSEM 视角**：Web 环境是一个不可信的知识源。攻击者利用“模态间隙 (Modality Gap)”注入虚假知识。例如，视觉层传递“购买”语义，而结构层（DOM）传递“取消”语义。这种**语义冲突 (Semantic Conflict)** 是导致智能体决策失效的根本原因。  
5. **第三段 (本文方法)**：介绍 TVSC。强调它不仅仅是一个分类器，而是一个**语义对齐协议 (Semantic Alignment Protocol)**。它利用 OCR 作为第三个独立的知识锚点来仲裁视觉和结构之间的冲突 9。  
6. **第四段 (贡献)**：  
7. 形式化了多模态智能体的**对抗性知识扰动威胁模型**。  
8. 提出了基于三视图对齐的**知识验证框架 (TVSC)**。  
9. 在 VisualWebArena 上进行了广泛的**鲁棒性评估**。

#### 2\. Related Work (相关工作)

* **增加板块：Knowledge Representation in Web Agents**：讨论智能体如何通过 DOM 和截图构建世界模型，以及这种构建过程中的脆弱性 10。  
* **Adversarial Robustness**：保留原有的攻击与防御讨论，但增加关于“数据完整性”和“系统可靠性工程”的引用，以契合 KSEM 的“Engineering”主题 11, 12。

#### 3\. Methodology (方法论 \- 核心扩充部分)

* **3.1 Threat Model as Knowledge Corruption (威胁模型作为知识破坏)**  
* 将五种攻击类型（Layout, Overlay, Homoglyph 等）13 描述为针对特定模态的**噪声注入**。  
* **Overlay Attack**: 视觉知识完整，结构知识被遮蔽。  
* **Homoglyph**: 文本语义在编码层面（DOM）与渲染层面（OCR）的**知识不一致** 14。  
* **3.2 Triple-View Knowledge Extraction (三视图知识提取)**  
* 详细描述如何从原始网页中提取 Visual View ($X\_t$), Structural View ($G\_t$), OCR View ($R\_t$) 15, 16。  
* *扩充点*：增加关于 Accessibility Tree 解析的工程细节，这符合 LNCS 篇幅较长的要求。  
* **3.3 Semantic Consistency Verification (语义一致性验证)**  
* 使用数学公式定义一致性分数，这在 KSEM 中很受欢迎。保留原稿中的公式 17, 18：$$S(v, d, r) \= w\_p C\_p \+ w\_t C\_t \+ w\_s C\_s$$  
* **深度阐述**：解释为什么 $C\_p$ (位置) 和 $C\_t$ (文本) 是知识验证的关键维度。例如，解释 OCR 视图如何作为“客观第三方”来验证 DOM 属性的真实性。  
* **3.4 Decision Logic & Knowledge Gating (决策逻辑与知识门控)**  
* 描述“双阈值机制” ($\\tau, \\tau'$) 19。将此描述为一种**风险管理策略**（Management 视角）。当一致性低于 $\\tau'$ 时，系统拒绝采纳该知识（拒绝操作），从而保护系统的完整性。

#### 4\. Experiments (实验)

* **数据集**：VisualWebArena (300 tasks) 20。  
* **对比基线**：GPT-5, Gemini 2.5 Pro, Claude 4.5 Sonnet 21。  
* **关键结果展示**：  
* 使用表格展示 Hybrid Attack 下的 SR (Success Rate) 恢复情况 (-11.7% drop $\\rightarrow$ \-2.6% drop) 22。  
* **增加图表**：建议将 "Performance Degradation Curves" 23 放大，并在 LNCS 格式中作为整页或半页图展示，清晰地展示攻击强度 ($s$) 与知识完整性（成功率）的关系。

#### 5\. Discussion & Engineering Analysis (讨论与工程分析)

* **这是 KSEM 论文能否录用的关键。需要扩充原稿的 Discussion 部分。**  
* **Computational Overhead (计算开销)**：详细讨论 Latency (170ms overhead) 24。论证在安全敏感型应用（如金融）中，这种“验证成本”相对于“错误执行风险”是完全合理的知识管理权衡 25。  
* **Failure Analysis (失效分析)**：详细剖析 "Perfect Fake" 和 "OCR Error" 26。这显示了对系统工程局限性的深刻理解。  
* **Scalability (可扩展性)**：讨论该框架如何扩展到其他领域（如移动端 GUI 代理），强调方法的通用性 27。

#### 6\. Conclusion (结论)

* 总结 TVSC 如何通过工程化手段解决科学问题（多模态不一致），重申其在构建可信赖 AI 系统中的价值。

### KSEM 投稿的具体注意事项 (Checklist)

* **格式模板 (Springer LNCS)**：  
* 务必使用 llncs.cls (LaTeX) 或 Springer 提供的 Word 模板 28, 29。  
* **不要**使用双栏格式（原 ICPR 草稿可能是双栏的，需要转换）。LNCS 是单栏，字体较大，这意味你需要更多的文字内容来填充页面。  
* 参考文献格式：使用 \[30\], \[31\] 编号引用，参考文献列表需包含所有作者姓名（除非超过一定数量），不要省略 32, 33。  
* **匿名化 (Anonymization)**：  
* KSEM 是双盲评审。原稿中的作者信息 1、GitHub 链接、以及 "Ours" 引用自己的以前工作必须移除或脱敏 4, 34。  
* PDF 文件的属性（Metadata）中不能包含作者名字 34。  
* **截稿日期管理**：  
* KSEM 2026 的截稿日期通常在 **1 月 15 日** 左右 4, 35。这比 ICPR 2026 (12月21日) 36 晚几周，给了你更多时间将论文从“模式识别”风格调整为“知识工程”风格。  
* **内容增强策略**:  
* 由于从双栏转单栏，且 KSEM 偏好较长论文（12-15页），建议将原稿中的 **Appendix B (Attack Implementation Details)** 37 的部分内容移入正文的 Methodology 部分，详细描述攻击是如何生成的，这被视为一种“对抗性知识生成”过程。  
* 将 **Case Studies** 38 扩充，增加截图对比（Benign vs. Adversarial），直观展示 TVSC 如何捕捉到不一致。

### 总结建议

将 ICPR 草稿转换为 KSEM 论文的核心在于：**不仅要讲“怎么做”（检测攻击），还要讲“为什么有效”（知识验证原理）以及“代价是什么”（系统管理权衡）。** 利用 KSEM 较晚的截稿日期和较长的篇幅限制，补充更多关于系统实现、开销分析和错误案例的工程细节，这将大大增加录用概率。  
