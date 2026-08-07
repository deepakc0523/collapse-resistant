"""
================================================================================
scrs/probe_loader.py
================================================================================

Loader module for Probe representation drift reports.

Parses and validates probe_out/representation_drift_report.json to extract
the required representation metrics for SCRS computation.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

from scrs.utils import get_scrs_logger


@dataclass
class ProbeMetrics:
    """Dataclass holding extracted raw Probe metrics."""
    hidden_state_cosine_similarity: float
    embedding_cosine_similarity: float
    attention_cosine_similarity: float
    kl_divergence: float
    js_divergence: float
    prediction_agreement_top1: float
    raw_data: Dict[str, Any]


class ProbeLoader:
    """Loads and validates Probe report JSON files."""

    def __init__(self, report_path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.report_path = Path(report_path)
        self.logger = logger or get_scrs_logger("scrs.probe_loader")

    def load(self) -> ProbeMetrics:
        """
        Loads the probe JSON report and extracts raw representation metrics.

        Returns
        -------
        ProbeMetrics
            Dataclass containing parsed raw metrics.

        Raises
        ------
        FileNotFoundError
            If the JSON report file does not exist.
        KeyError / ValueError
            If expected JSON metrics are missing or malformed.
        """
        if not self.report_path.is_file():
            self.logger.error("Probe report not found at %s", self.report_path)
            raise FileNotFoundError(f"Probe report file not found: {self.report_path}")

        self.logger.info("Loading Probe report from %s", self.report_path)
        with open(self.report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Hidden state cosine similarity (average across layers)
        hidden_analysis = data.get("layer_wise_hidden_state_analysis", {})
        if not hidden_analysis:
            raise KeyError("Missing 'layer_wise_hidden_state_analysis' in Probe report.")
        
        hidden_cosines = [
            layer_info["cosine_similarity"]
            for layer_info in hidden_analysis.values()
            if "cosine_similarity" in layer_info
        ]
        mean_hidden_cosine = sum(hidden_cosines) / len(hidden_cosines) if hidden_cosines else 0.0

        # 2. Embedding cosine similarity (dynamic token embeddings mean)
        emb_analysis = data.get("embedding_analysis", {}).get("dynamic", {})
        emb_cosine = emb_analysis.get("token_embeddings_cosine_similarity_mean")
        if emb_cosine is None:
            # Fallback to static if dynamic not present
            emb_cosine = data.get("embedding_analysis", {}).get("static", {}).get("token_embeddings_cosine_similarity", 0.0)

        # 3. Attention cosine similarity (average across layers)
        att_cos_dict = data.get("attention_analysis", {}).get("layer_wise_attention_cosine_similarity", {})
        if att_cos_dict:
            att_cosine = sum(att_cos_dict.values()) / len(att_cos_dict)
        else:
            att_cosine = 0.0

        # 4. Logit analysis metrics (KL, JS, Agreement)
        logit_analysis = data.get("logit_analysis", {})
        kl_div = float(logit_analysis.get("mean_kl_divergence", 0.0))
        js_div = float(logit_analysis.get("mean_js_divergence", 0.0))
        pred_agree = float(logit_analysis.get("prediction_agreement_top1", 0.0))

        self.logger.info(
            "Extracted Probe Metrics -> HiddenCos: %.4f, EmbCos: %.4f, AttCos: %.4f, KL: %.4f, JS: %.4f, Agreement: %.4f",
            mean_hidden_cosine,
            emb_cosine,
            att_cosine,
            kl_div,
            js_div,
            pred_agree,
        )

        return ProbeMetrics(
            hidden_state_cosine_similarity=mean_hidden_cosine,
            embedding_cosine_similarity=emb_cosine,
            attention_cosine_similarity=att_cosine,
            kl_divergence=kl_div,
            js_divergence=js_div,
            prediction_agreement_top1=pred_agree,
            raw_data=data,
        )
