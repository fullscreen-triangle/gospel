"""
Build a small sample protein and DNA database with precomputed spectral
embeddings, for the homology-search web tool.

Produces two JSON files under `public/data/`:
  - sample_db_protein.json
  - sample_db_dna.json

Each has the schema:
  {
    "kind": "protein" | "dna",
    "n_coefficients": K,
    "embed_dim": c*K,
    "sequences": [{"id": str, "family": int, "length": int, "text": str}, ...],
    "embeddings_flat": [float, ...],   # length = N * embed_dim, L2-normalised
    "families": [{"id": int, "label": str, "members": [seq_index, ...]}, ...]
  }

The same algorithm is used in the paper, so web results match the paper.
"""

from __future__ import annotations
import json
import os
import sys

import numpy as np

# re-use the paper's common library
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "publications", "shader-based-homology",
                                "experiments"))
from common import (DNA_ALPHABET, PROTEIN_ALPHABET, generate_family, embed_batch,
                    random_sequence, mutate)


def build(kind: str, n_families: int, family_size: int, length: int,
          mutation_rate: float, n_coefficients: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    alphabet = DNA_ALPHABET if kind == "dna" else PROTEIN_ALPHABET

    sequences: list[dict] = []
    family_blocks: list[dict] = []
    member_seqs: list[str] = []

    seq_index = 0
    for f in range(n_families):
        members = generate_family(length, family_size, mutation_rate,
                                  alphabet, rng)
        member_idx = []
        for m_i, seq in enumerate(members):
            seq_id = f"fam{f:04d}_mem{m_i:02d}"
            sequences.append({
                "id": seq_id,
                "family": f,
                "length": len(seq),
                "text": seq,
            })
            member_seqs.append(seq)
            member_idx.append(seq_index)
            seq_index += 1
        family_blocks.append({
            "id": f,
            "label": f"Synthetic family {f:04d}",
            "members": member_idx,
        })

    E = embed_batch(member_seqs, n_coefficients=n_coefficients, kind=kind)

    return {
        "kind": kind,
        "n_coefficients": n_coefficients,
        "embed_dim": int(E.shape[1]),
        "n_sequences": len(sequences),
        "n_families": n_families,
        "family_size": family_size,
        "length": length,
        "mutation_rate": mutation_rate,
        "seed": seed,
        "sequences": sequences,
        "embeddings_flat": E.astype(np.float32).ravel().tolist(),
        "families": family_blocks,
    }


def main():
    out_dir = os.path.join(HERE, "..", "public", "data")
    os.makedirs(out_dir, exist_ok=True)

    protein = build(kind="protein", n_families=120, family_size=6,
                    length=180, mutation_rate=0.10,
                    n_coefficients=12, seed=20260423)
    path = os.path.join(out_dir, "sample_db_protein.json")
    with open(path, "w") as f:
        json.dump(protein, f)
    print(f"wrote {path}: {protein['n_sequences']} sequences, "
          f"embed_dim={protein['embed_dim']}")

    dna = build(kind="dna", n_families=120, family_size=6,
                length=300, mutation_rate=0.10,
                n_coefficients=12, seed=20260424)
    path = os.path.join(out_dir, "sample_db_dna.json")
    with open(path, "w") as f:
        json.dump(dna, f)
    print(f"wrote {path}: {dna['n_sequences']} sequences, "
          f"embed_dim={dna['embed_dim']}")


if __name__ == "__main__":
    main()
