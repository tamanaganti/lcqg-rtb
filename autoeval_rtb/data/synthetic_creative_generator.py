import numpy as np
import pandas as pd
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def _unit_vector(dim, seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    return v / np.linalg.norm(v)


def generate_data(n_creatives, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    rng_main = np.random.RandomState(42)

    n_high = int(round(n_creatives * 0.40))
    n_med  = int(round(n_creatives * 0.35))
    n_low  = n_creatives - n_high - n_med

    # Latent quality by tier
    z_high = np.random.RandomState(42).beta(7, 2, n_high)   # mean≈0.78
    z_med  = np.random.RandomState(43).beta(4, 4, n_med)    # mean≈0.50
    z_low  = np.random.RandomState(44).beta(2, 7, n_low)    # mean≈0.22

    z_all    = np.concatenate([z_high, z_med, z_low])
    tier_all = np.array([2] * n_high + [1] * n_med + [0] * n_low)  # 2=high,1=med,0=low

    # PRS noise
    prs_noise_all = np.random.RandomState(500).normal(0, 0.07, n_creatives)

    # Embeddings
    dim = 128
    mu_high = _unit_vector(dim, 144)   # dot(mu_high, W_PRS) ≈ +0.227
    mu_med  = _unit_vector(dim, 52)    # dot(mu_med,  W_PRS) ≈ +0.201
    mu_low  = _unit_vector(dim, 487)   # dot(mu_low,  W_PRS) ≈ -0.212

    mu_map = {2: mu_high, 1: mu_med, 0: mu_low}

    # Brand embeddings
    n_brands = 10
    brand_embs = np.stack([_unit_vector(dim, 100 + i) for i in range(n_brands)])  # (10, 128)

    # Brand assignments
    brand_ids = np.random.RandomState(42).randint(0, n_brands, n_creatives)

    # Style noise
    style_rng = np.random.RandomState(42)
    style_noise_all = style_rng.normal(0, 0.10, (n_creatives, dim))

    embeddings = np.zeros((n_creatives, dim))
    for i in range(n_creatives):
        mu = mu_map[tier_all[i]]
        be = brand_embs[brand_ids[i]]
        raw = z_all[i] * mu + 0.3 * be + style_noise_all[i]
        norm = np.linalg.norm(raw)
        embeddings[i] = raw / norm if norm > 0 else raw

    # Artifact injection
    artifact_flags = np.zeros(n_creatives, dtype=int)
    artifact_types = np.array(['none'] * n_creatives, dtype=object)

    low_indices = np.where(tier_all == 0)[0]
    art_rng = np.random.RandomState(42)
    for idx in low_indices:
        if art_rng.random() < 0.70:
            artifact_flags[idx] = 1
            r = art_rng.random()
            if r < 0.40:
                artifact_types[idx] = 'A'
                noise = art_rng.normal(0, 1, 16)
                embeddings[idx, 0:16] += noise
            elif r < 0.75:
                artifact_types[idx] = 'B'
                dims_b = art_rng.choice(dim, 8, replace=False)
                signs = art_rng.choice([-1, 1], 8)
                embeddings[idx, dims_b] = signs * 3.0
            else:
                artifact_types[idx] = 'C'
                embeddings[idx, 48:80] = 0.0
                embeddings[idx, 48:80] += art_rng.normal(0, 0.05, 32)
            norm = np.linalg.norm(embeddings[idx])
            if norm > 0:
                embeddings[idx] /= norm

    # Context fields
    ctx_rng = np.random.RandomState(42)
    device_types_list = ['mobile', 'desktop', 'tablet']
    slot_types_list   = ['banner', 'native', 'sidebar', 'instream']

    device_choices = ctx_rng.choice(device_types_list, n_creatives, p=[0.55, 0.35, 0.10])
    slot_choices   = ctx_rng.choice(slot_types_list,   n_creatives, p=[0.40, 0.25, 0.20, 0.15])

    device_map = {d: i for i, d in enumerate(device_types_list)}
    slot_map   = {s: i for i, s in enumerate(slot_types_list)}
    context_ids = np.array([
        device_map[device_choices[i]] * len(slot_types_list) + slot_map[slot_choices[i]]
        for i in range(n_creatives)
    ])

    # Engagement proxy
    brand_rng = np.random.RandomState(300)
    brand_effects = brand_rng.uniform(-0.005, 0.005, n_brands)

    def context_multiplier(device, slot):
        if device == 'mobile' and slot == 'banner':
            return 0.85
        elif device == 'mobile' and slot == 'native':
            return 1.10
        elif device == 'desktop' and slot == 'instream':
            return 1.20
        else:
            return 1.00

    noise_rng = np.random.RandomState(42)
    mult_noise = noise_rng.normal(0, 0.08, n_creatives)

    engagement_proxy = np.zeros(n_creatives)
    for i in range(n_creatives):
        base = np.clip(0.015 + 0.04 * z_all[i] + brand_effects[brand_ids[i]], 0.005, 0.15)
        cm   = context_multiplier(device_choices[i], slot_choices[i])
        engagement_proxy[i] = base * cm * np.exp(mult_noise[i])

    records = []
    for i in range(n_creatives):
        records.append({
            'creative_id':       i,
            'quality_tier':      int(tier_all[i]),
            'latent_quality':    float(z_all[i]),
            'brand_id':          int(brand_ids[i]),
            'artifact_flag':     int(artifact_flags[i]),
            'artifact_type':     artifact_types[i],
            'device_type':       device_choices[i],
            'slot_type':         slot_choices[i],
            'context_id':        int(context_ids[i]),
            'engagement_proxy':  float(engagement_proxy[i]),
            'prs_noise':         float(prs_noise_all[i]),
        })

    df = pd.DataFrame(records)
    df['embedding']       = [embeddings[i].tolist() for i in range(n_creatives)]
    df['brand_embedding'] = [brand_embs[brand_ids[i]].tolist() for i in range(n_creatives)]

    parquet_path = os.path.join(results_dir, 'synthetic_creatives.parquet')
    df.to_parquet(parquet_path, index=False)

    brand_embs_path = os.path.join(results_dir, 'brand_embeddings.npy')
    np.save(brand_embs_path, brand_embs)

    tier_names = {2: 'HIGH', 1: 'MED', 0: 'LOW'}
    n_h = (df['quality_tier'] == 2).sum()
    n_m = (df['quality_tier'] == 1).sum()
    n_l = (df['quality_tier'] == 0).sum()

    art_a = (df['artifact_type'] == 'A').sum()
    art_b = (df['artifact_type'] == 'B').sum()
    art_c = (df['artifact_type'] == 'C').sum()
    art_n = (df['artifact_type'] == 'none').sum()

    mean_z_h = df.loc[df['quality_tier'] == 2, 'latent_quality'].mean()
    mean_z_m = df.loc[df['quality_tier'] == 1, 'latent_quality'].mean()
    mean_z_l = df.loc[df['quality_tier'] == 0, 'latent_quality'].mean()

    mean_eng_h = df.loc[df['quality_tier'] == 2, 'engagement_proxy'].mean()
    mean_eng_m = df.loc[df['quality_tier'] == 1, 'engagement_proxy'].mean()
    mean_eng_l = df.loc[df['quality_tier'] == 0, 'engagement_proxy'].mean()

    print("=== DATA GENERATION SUMMARY ===")
    print(f"Total creatives : {n_creatives}")
    print(f"HIGH / MED / LOW: {n_h} / {n_m} / {n_l}")
    print(f"Artifact types  : A={art_a} B={art_b} C={art_c} none={art_n}")
    print(f"Mean z by tier  : HIGH={mean_z_h:.2f} MED={mean_z_m:.2f} LOW={mean_z_l:.2f}")
    print(f"Mean engagement : HIGH={mean_eng_h:.3f} MED={mean_eng_m:.3f} LOW={mean_eng_l:.3f}")

    return df, brand_embs


def generate_table0(df, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    tier_map = {0: 'LOW', 1: 'MED', 2: 'HIGH'}
    rows = []
    for tier_val, tier_name in sorted(tier_map.items(), key=lambda x: -x[0]):
        sub = df[df['quality_tier'] == tier_val]
        if len(sub) == 0:
            continue
        rows.append({
            'tier':                  tier_name,
            'count':                 len(sub),
            'mean_z':                round(sub['latent_quality'].mean(), 4),
            'mean_engagement_proxy': round(sub['engagement_proxy'].mean(), 5),
            'artifact_prevalence':   round(sub['artifact_flag'].mean(), 4),
            'mean_prs_noise_abs':    round(sub['prs_noise'].abs().mean(), 4),
            'device_mobile_frac':    round((sub['device_type'] == 'mobile').mean(), 4),
            'slot_banner_frac':      round((sub['slot_type'] == 'banner').mean(), 4),
        })

    table0 = pd.DataFrame(rows)
    table0.to_csv(os.path.join(results_dir, 'table0_data_sanity.csv'), index=False)
    print("\n=== TABLE 0: DATA SANITY (§4.1 Simulation Environment) ===")
    print(table0.to_string(index=False))
    return table0
