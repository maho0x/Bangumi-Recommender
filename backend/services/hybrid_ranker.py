"""Hybrid ranking: fuse CF and content-based scores with post-processing."""

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from backend.config import (
    CF_WEIGHT_DEFAULT,
    CF_WEIGHT_COLD_START,
    COLD_START_THRESHOLD,
    DIVERSITY_LAMBDA,
    POPULARITY_WEIGHT,
    NSFW_DEFAULT,
    PROCESSED_DIR,
)


class HybridRanker:
    def __init__(self):
        self.subjects_meta = None
        self.loaded = False

    def load(self):
        """Load subject metadata for post-processing."""
        meta_path = PROCESSED_DIR / "subjects_meta.parquet"
        if meta_path.exists():
            self.subjects_meta = pd.read_parquet(meta_path)
            self.subjects_meta = self.subjects_meta.set_index("id")
            self.loaded = True
            print(f"Loaded {len(self.subjects_meta):,} subjects metadata for ranking")

    def _get_subject_info(self, subject_id: int) -> dict:
        """Get metadata for a subject."""
        if self.subjects_meta is None:
            return {}
        try:
            row = self.subjects_meta.loc[subject_id]
            return row.to_dict()
        except KeyError:
            return {}

    def rank(
        self,
        cf_scores: list[tuple[int, float]],
        content_scores: list[tuple[int, float]],
        user_collections: list[dict],
        subject_type: int = 2,
        limit: int = 20,
        filter_nsfw: bool = NSFW_DEFAULT,
        content_recommender=None,
    ) -> list[dict]:
        """
        Fuse CF and content scores, apply post-processing, return ranked results.
        """
        # Determine CF weight based on collection size
        anime_count = sum(
            1 for c in user_collections
            if c.get("subject", {}).get("type", 0) == 2 or c.get("subject_type", 0) == 2
        )
        alpha = CF_WEIGHT_DEFAULT if anime_count >= COLD_START_THRESHOLD else CF_WEIGHT_COLD_START

        # Only use CF for anime (type=2)
        use_cf = subject_type == 2 and len(cf_scores) > 0

        # Build score dicts
        cf_dict = dict(cf_scores) if use_cf else {}
        content_dict = dict(content_scores)

        # Get all candidate subject IDs
        all_candidates = set(cf_dict.keys()) | set(content_dict.keys())

        # Exclude already collected subjects
        collected_ids = set()
        for col in user_collections:
            sid = col.get("subject_id", col.get("subject", {}).get("id", 0))
            collected_ids.add(sid)
        all_candidates -= collected_ids

        # Compute fused scores
        scored = []
        for sid in all_candidates:
            cf_s = cf_dict.get(sid, 0.0)
            content_s = content_dict.get(sid, 0.0)

            if use_cf and sid in cf_dict:
                fused = alpha * cf_s + (1 - alpha) * content_s
            else:
                # Cold start or non-anime: content only
                fused = content_s

            scored.append((sid, fused, cf_s, content_s))

        # Sort by fused score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Apply MMR diversity re-ranking
        if content_recommender is not None and len(scored) > limit:
            scored = self._mmr_rerank(scored, content_recommender, limit * 3)

        # Post-processing: NSFW filter, popularity boost
        results = []
        for sid, fused, cf_s, content_s in scored:
            info = self._get_subject_info(sid)

            # NSFW filter
            if filter_nsfw and info.get("nsfw", False):
                continue

            # Popularity micro-boost
            collect_count = info.get("collect", 0)
            if collect_count and collect_count > 0:
                pop_boost = POPULARITY_WEIGHT * math.log1p(collect_count) / 10.0
                fused += pop_boost

            # Build tags list (ensure Python list, parquet may return numpy arrays)
            tags = info.get("tag_list", [])
            if isinstance(tags, str):
                import ast
                try:
                    tags = ast.literal_eval(tags)
                except (ValueError, SyntaxError):
                    tags = []
            elif hasattr(tags, "tolist"):
                tags = tags.tolist()
            elif not isinstance(tags, list):
                tags = []

            # Generate recommendation reason
            reason = self._generate_reason(sid, cf_s, content_s, tags, user_collections)

            # Image URL from Bangumi
            image_url = f"https://api.bgm.tv/v0/subjects/{sid}/image?type=medium"

            results.append({
                "subject_id": sid,
                "name_cn": str(info.get("name_cn", "")),
                "name": str(info.get("name", "")),
                "score": round(fused, 4),
                "reason": reason,
                "bangumi_score": info.get("parsed_score") or info.get("score"),
                "tags": tags[:8] if tags else [],
                "image_url": image_url,
                "subject_type": int(info.get("type", subject_type)),
            })

            if len(results) >= limit:
                break

        return results

    def _mmr_rerank(
        self,
        scored: list[tuple],
        content_recommender,
        top_n: int,
    ) -> list[tuple]:
        """Maximal Marginal Relevance re-ranking for diversity."""
        candidates = scored[:top_n]
        if len(candidates) <= 1:
            return candidates

        # Get embeddings for candidates
        emb_cache = {}
        for sid, _, _, _ in candidates:
            emb = content_recommender.get_embedding(sid)
            if emb is not None:
                emb_cache[sid] = emb

        selected = [candidates[0]]
        remaining = list(candidates[1:])

        while remaining and len(selected) < len(candidates):
            best_score = -float("inf")
            best_idx = 0

            for i, (sid, fused, cf_s, ct_s) in enumerate(remaining):
                # Relevance
                relevance = fused

                # Max similarity to already selected
                max_sim = 0.0
                if sid in emb_cache:
                    for sel_sid, _, _, _ in selected:
                        if sel_sid in emb_cache:
                            sim = float(np.dot(emb_cache[sid], emb_cache[sel_sid]))
                            max_sim = max(max_sim, sim)

                # MMR score
                mmr = (1 - DIVERSITY_LAMBDA) * relevance - DIVERSITY_LAMBDA * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _generate_reason(
        self,
        subject_id: int,
        cf_score: float,
        content_score: float,
        tags: list,
        user_collections: list[dict],
    ) -> str:
        """Generate a human-readable recommendation reason."""
        reasons = []

        if cf_score > 0.5:
            reasons.append("与你口味相似的用户也喜欢这部作品")

        if content_score > 0.7 and tags:
            # Find matching tags with user's highly-rated items
            tag_str = "、".join(tags[:3])
            reasons.append(f"在 {tag_str} 等标签上与你的收藏相似")
        elif content_score > 0.5 and tags:
            tag_str = "、".join(tags[:2])
            reasons.append(f"与你收藏的作品在 {tag_str} 上风格接近")

        if not reasons:
            if tags:
                reasons.append(f"基于 {', '.join(tags[:3])} 等特征推荐")
            else:
                reasons.append("根据综合分析推荐")

        return "；".join(reasons)
