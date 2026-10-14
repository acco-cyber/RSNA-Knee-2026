# RSNA Knee 2026: rules and L2 validation audit

Checked 2026-10-15 KST. Internal research handoff; no credentials used and no Kaggle/GitHub mutations performed. Official Kaggle pages were client-rendered in `open`, so factual competition claims below come from indexed official Kaggle content returned by web search. Exact account quota and currently remaining submissions must come from the authenticated root workflow.

## Official competition contract

Competition: [`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview). Metric is the unweighted mean of twelve study-level ROC-AUC values. Submit a committed notebook producing `submission.csv`, with internet disabled, within nine hours on CPU or GPU. Publicly available free external data and pretrained models are allowed subject to the rules.

The indexed official timeline lists final submission **2026-10-22 23:59 UTC = 2026-10-23 08:59 KST**, entry and team merger **2026-10-15 23:59 UTC = 2026-10-16 08:59 KST**, and winners' delivery **2026-11-05 23:59 UTC**. Source: official overview, web refs `turn20search0`, `turn27search2`.

The [official rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) confirm **five submissions per day** and **up to two final submissions**. Sharing competition code requires making it available through competition-associated Kaggle discussions/notebooks. An incompatible input-data/pretrained-model license has an explicit carve-out from relicensing that input under the winners' license; this is not a waiver of its own terms. Sources: `turn57search0`, `turn60search0`, `turn54search0`.

The nine-hour limit is a competition inference-notebook limit; it does not prove this account has GPU quota or two unused slots. Parent authenticated metadata can establish those. The user is asking at 04:49 KST on October 15, when it is still October 14 UTC: do not assume a KST midnight resets Kaggle's daily submission count.

## Official data and exact target names

The [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=sample_submission.csv) describes training reports and a labeled subset, per-series descriptors, and DICOM slices under study/series directories. `train_series.csv` and `test_series.csv` carry `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`. Fluid sensitivity and fat suppression are not interchangeable.

Visible test data has three studies; scoring replaces it with approximately 1,300 studies. **No test `Report` is provided**. The submission must derive predictions from MRI and permitted metadata. Prevalence can differ between train, public, and private sets. DICOM transfer syntaxes, intensities, orientations, and resolutions vary. Source: `turn58search1` and `turn57search3`.

Required columns, in order:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

There are no PCL/LCL targets. The memory summary's older target listing is unsuitable for implementation.

The task context's 4,407 training / 58 gold / 4,349 report-only counts should be checked against the downloaded CSV before building L2; they were not separately counted in this audit.

## Host severity rubric

Source: host Po-Hao Chen's [clinical overview, discussion 733343](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343), web ref `turn48search0`. Whole-exam labels reflect two independent MSK readers plus adjudication; borderline findings are negative.

| Target | Positive criterion / key exclusion |
| --- | --- |
| ACL | High-grade partial/full tear, >50% fibers disrupted; degeneration or thickening alone negative. |
| MCL | High-grade acute partial/complete tear with disrupted fibers/edema; low sprain and remote change negative. |
| Medial/Lateral Meniscus | Definite surface-contacting signal on ≥2 images or tear-related morphology; intrasubstance degeneration negative. |
| Medial/Lateral/PF OA | >50% cartilage-thickness loss over a moderate/large area, approximately ≥1 cm. |
| Effusion | Moderate/large distending fluid. |
| Synovitis | Inflamed, thickened synovial lining. |
| Baker's | Moderate/large posterior cyst. |
| Contusion | Impact-related marrow edema without discrete fracture line. |
| Fracture | Acute cortical break/fracture line. |

L2 report prompts should explicitly preserve this distinction between an abnormality mentioned in a report and a finding meeting the challenge threshold. An unmentioned finding is not demonstrated negative; retaining unknown masks/confidence is our methodological recommendation, not a host-mandated encoding.

## Hosted LLM labels and external-data status

Host [announcement 733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) explicitly permits commercially hosted LLMs/external inference services subject to competition-rule compliance, reasonable accessibility, and minimal cost. Primary indexed snippet verifies permission (`turn55search0`), the private-sharing rationale (`turn56search0`), and the report-text/API example (`turn31search0`). Earlier participant guesses that commercial APIs are prohibited are stale. Full primary thread body did not render in this audit; do not present an unseen extended quotation as retrieved.

The [OAI-specific host answer, discussion 741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) is conditional: free access for all without institutional review/sign-off or legal-team access. It does not waive OAI steward terms. Source `turn48search1`. [Discussion 733652](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652) contains questions and participant interpretations, not blanket approval of MRNet, fastMRI, OAI, or SKM-TEA (`turn40search1`). OAI-derivative rehost question [738156](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738156) has no retrieved host approval (`turn49search7`). Do not infer permission solely from a public Kaggle/Hugging Face mirror.

General public pretrained-weight permission is established by the overview. For specific licenses, the original [DINOv2 README](https://github.com/facebookresearch/dinov2/blob/main/README.md) identifies original DINOv2 code and weights as Apache-2.0 (`turn59search7`, `turn60view1`). [timm](https://github.com/huggingface/pytorch-image-models) distinguishes its code license from pretrained-weight terms (`turn59search12`, `turn59search14`). The original [RadImageNet repository LICENSE](https://github.com/BMEII-AI/RadImageNet/blob/main/LICENSE) is MIT for software (`turn60view0`); that alone does not establish the license of every separately distributed checkpoint. The [CC-BY-NC/SA weights question, discussion 735121](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121) yielded no retrieved host answer (`turn49search6`). Check the actual attached artifact/model-card terms; do not claim all NC weights are disallowed or all timm weights inherit Apache-2.0.

## Local lineage finding: Gold58 is already contaminated as a selection set

Direct source inspected: `/workspace/scratch/2d32e5629d42/RSNA-Knee-2026/rsna-knee-v224-r7-reader-students10.ipynb`, cell 12. Its comments state that Raptor trained on every training study and was selected on the 58 gold studies, leaving no clean labeled set to fit blend weights. The notebook also documents multiple earlier selection panels on 45/58 gold studies. This is local-code evidence, not an allegation about Kaggle hidden labels.

Both recovered h957/v224 notebooks acknowledge controlled-access OAI, and h957 cell 0 describes fastMRI auxiliary training. Attribution is not proof that the exact access route meets current host conditions. Record access/weight provenance before choosing this lineage for a prize-eligible final solution.

## L2 audit requirements and score interpretation

The following is our inference/recommendation. The official [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html?highlight=mean_squared_error) explains that fitting transformations or making choices with evaluation data inflates performance (`turn58search0`).

1. **The reported 0.940 is target-label agreement, not a trained L2 model's held-out AUC or Kaggle score.** It cannot justify naming a measured score of 0.962. The notebook may call 0.962 a target, with the actual score recorded only after scoring.
2. Public teacher predictions on a study the teacher trained on are distillation targets, not OOF validation evidence. Averaging them with an OOF student's predictions does not make the combined target OOF.
3. Round-2 students trained on targets chosen with Gold58 are indirectly influenced by Gold58 even if their image optimizer excludes those rows. Recursive pseudo-labeling propagates that influence into new training targets. Mark validation accordingly.
4. Freeze outer study/patient folds before target generation. For each outer validation fold, every teacher, target-mix fit, calibrator, blend-weight search, and checkpoint selection used to evaluate it must exclude that fold. Use inner CV for mixing if fitting is necessary. Retrain a final model on all permissible training data after model selection; this is compatible with honest prior evaluation.
5. For the seven label sources, save source URL/version/hash, method, report-language coverage, target rubric, and whether one is derived from another. Seven correlated LLM outputs are not seven independent experts. Fit only low-capacity/shrunk per-target reliabilities within inner folds, or start from fixed equal source-family weights.
6. Preserve study UID joins and target order, reject duplicates/missing IDs, and emit coverage/disagreement/mask summaries. A new-label source needs exactly the intended reports with no partially duplicated batches; count and hash each batch.
7. Where report evidence is high confidence, preserve it as one training signal; where unknown/conflicting, permit an image teacher to contribute with separately recorded confidence. Tune that policy only in allowed inner training folds. In-sample targets can be used for final distillation when clearly disclosed, but cannot certify generalization.
8. Report all twelve AUCs and paired study-bootstrap uncertainty. A 58-study panel is too small for a narrow 0.960–0.964 promise, especially after repeated per-label searches. A nominal uplift needs an independent check, not just a better tuned Gold58 macro average.
9. For the requested two attempts, commit distinct candidates only after offline integrity checks: exact hidden-test UID coverage, unique UIDs, 12 finite numeric target columns, confidence range, validated DICOM fallbacks, all required checkpoint files, and realistic timing for ~1,300 studies. A visible-three-study run proves format, not full runtime. Keep the two attempts distinct and document what each tests.

## Unresolved items for root

- Actual remaining submission count/reset time and GPU quota: authenticated account state.
- Recover all 45 labeling batches, `build_l2.py`, source CSVs, student/teacher OOF artifacts and fold assignments: not present in inspected public repository.
- Validate exact OAI/fastMRI artifact origins and access terms; existing acknowledgement does not settle eligibility.
- Independent evidence for 0.960–0.964: none in this rules audit. Only scored Kaggle results can establish the public score.
