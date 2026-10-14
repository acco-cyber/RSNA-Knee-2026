"""Fail closed when a fold-0 L2 smoke artifact differs from its selected sources."""


def verify_l2_smoke(checkpoint, completion, manifest, expected, checkpoint_sha256,
                    warm_start_sha256, targets):
    def require(condition, message):
        if not condition:
            raise RuntimeError("L2 provenance: " + message)

    require(expected is not None, "final, completely annotated label generation has not been pinned")
    require(expected.get("annotations_complete") is True, "selected labels are an incomplete pilot")
    args, provenance = checkpoint["args"], checkpoint["provenance"]
    exact_args = {"backbone": "convnext_tiny", "mode": "l2", "fold": 0,
                  "res": 256, "k_train": 4, "k_eval": 12, "epochs": 1,
                  "dim": 384, "depth": 2, "drop": .2, "drop_path": .1,
                  "confidence_weighted_bce": True}
    for key, value in exact_args.items():
        require(args.get(key) == value, "checkpoint argument differs: " + key)
    require(provenance.get("targets") == targets, "checkpoint target order differs")
    require(provenance.get("fold") == 0, "checkpoint provenance is a different fold")
    require(provenance.get("completed_epochs") == 1, "exactly one full smoke epoch is required")
    require(provenance.get("requested_epochs") == 1, "requested epoch count differs")
    require(args.get("epoch") == 0, "checkpoint is not the sole fixed epoch")
    require(provenance.get("gold_excluded_from_gradient_training") is True, "gold entered gradient training")
    require(provenance.get("gold_auc_independent_validation") is False, "gold diagnostic must disclose tuning exposure")
    require(provenance.get("weak_val_independent_of_label_generation") is False, "weak validation must disclose upstream exposure")
    require(provenance.get("selection") == "last fixed epoch; no gold-best selection", "checkpoint selection differs")
    require(provenance.get("preprocessing") ==
            "A1 cached32 slices; area256; JPEGq95; real six slots without FS fallback",
            "preprocessing contract differs")
    source_hashes = provenance.get("source_hashes", {})
    for name, sha256 in expected["source_hashes"].items():
        require(source_hashes.get(name) == sha256, "source digest differs: " + name)
    require(source_hashes.get("warm_start") == warm_start_sha256, "warm start differs from the incumbent r2 fold 0")
    for name in ("train_ids_sha256", "val_ids_sha256"):
        if name in expected:
            require(provenance.get(name) == expected[name], "partition digest differs: " + name)
    exact_completion = {"fold": 0, "completed_epochs": 1, "requested_epochs": 1,
                        "k_train": 4, "k_eval": 12,
                        "checkpoint": "student_l2_convnext_tiny_f0.pt",
                        "checkpoint_sha256": checkpoint_sha256}
    for key, value in exact_completion.items():
        require(completion.get(key) == value, "completion receipt differs: " + key)
    exact_manifest = {"model_family": "l2_convnext_tiny", "requested_folds": [0],
                      "requested_epochs": 1, "k_train": 4, "k_eval": 12,
                      "smoke_only": True, "five_folds_complete": False,
                      "dataset": expected["dataset"], "targets": targets,
                      "gold_auc_is_independent_validation": False}
    for key, value in exact_manifest.items():
        require(manifest.get(key) == value, "training manifest differs: " + key)
    require(len(manifest.get("receipts", [])) == 1, "exactly one training receipt is required")
    for key, value in exact_completion.items():
        require(manifest["receipts"][0].get(key) == value, "manifest receipt differs: " + key)
    return {"verified": True, "completed_epochs": 1, "train_windows": 4,
            "eval_windows": 12, "annotations_complete": True,
            "annotated_studies": expected["annotated_studies"],
            "dataset": expected["dataset"], "dataset_version": expected.get("dataset_version"),
            "checkpoint_sha256": checkpoint_sha256}
