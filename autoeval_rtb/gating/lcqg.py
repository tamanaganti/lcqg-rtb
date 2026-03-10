import time
import numpy as np
import pandas as pd

from autoeval_rtb.scoring.cri import score_cri


class LCQG:
    """Latency-Constrained Quality Gate."""

    def __init__(self, threshold=0.50, fusion_mode='expert', context_aware=False,
                 results_dir=None):
        self.threshold     = threshold
        self.fusion_mode   = fusion_mode
        self.context_aware = context_aware
        self.results_dir   = results_dir

    def process_bid_request(self, creative_id, embedding, brand_embedding,
                            prs_noise, device_type, slot_type):
        """
        Process a single bid request.

        Returns
        -------
        dict: {creative_id, prs, bci, acs, cri, gate_decision, latency_ms, fusion_mode}
        """
        t0 = time.perf_counter()
        result = score_cri(
            embedding, brand_embedding, prs_noise,
            device_type, slot_type,
            mode=self.fusion_mode,
            context_aware=self.context_aware,
            threshold=self.threshold,
            results_dir=self.results_dir,
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return {
            'creative_id':   creative_id,
            'prs':           result['prs'],
            'bci':           result['bci'],
            'acs':           result['acs'],
            'cri':           result['cri'],
            'gate_decision': result['gate'],
            'latency_ms':    latency_ms,
            'fusion_mode':   result['fusion_mode'],
        }

    def process_batch(self, df):
        """
        Process a DataFrame of creatives.

        Parameters
        ----------
        df : pd.DataFrame with columns: creative_id, embedding, brand_embedding,
             prs_noise, device_type, slot_type

        Returns
        -------
        pd.DataFrame with added columns: prs, bci, acs, cri, gate_decision, latency_ms
        """
        records = []
        for _, row in df.iterrows():
            r = self.process_bid_request(
                creative_id=row['creative_id'],
                embedding=np.asarray(row['embedding']),
                brand_embedding=np.asarray(row['brand_embedding']),
                prs_noise=row['prs_noise'],
                device_type=row['device_type'],
                slot_type=row['slot_type'],
            )
            records.append(r)

        result_df = pd.DataFrame(records)
        return result_df
