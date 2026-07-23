"""AML-specific pattern detectors: layering-chain and rapid-pass-through
detection. Both operate on the transaction **dataframe**, not the built
`HeteroData` graph -- `HeteroData` is a tensor container for `SAGEConv`/
`HGTConv` consumption (`edge_index` is a `[2,E]` int tensor with no
adjacency/traversal API); reconstructing a Python adjacency structure from it
to do multi-hop chain-walking would be strictly more work than building one
directly from the dataframe already in hand. This also keeps these detectors
runnable standalone, independent of the graph-builder/model pipeline, in the
same "rules first" spirit as `statement_analysis/analysis/patterns.py`.

Each detector returns a list of finding dicts:
`{pattern_type, description, severity, affected_row_indices}`, matching
`statement_analysis`'s convention.
"""

from typing import Dict, List

import pandas as pd

SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH = "low", "medium", "high"

DEFAULT_MAX_HOPS = 6
DEFAULT_MAX_HOP_GAP = pd.Timedelta(hours=72)
DEFAULT_PASS_THROUGH_GAP = pd.Timedelta(hours=6)
DEFAULT_MIN_PASS_THROUGH_AMOUNT = 500.0


def _finding(pattern_type: str, description: str, severity: str, affected_row_indices: List[int]) -> Dict:
    return {
        "pattern_type": pattern_type,
        "description": description,
        "severity": severity,
        "affected_row_indices": affected_row_indices,
    }


def detect_layering_chains(
    df: pd.DataFrame,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_hop_gap: pd.Timedelta = DEFAULT_MAX_HOP_GAP,
    amount_decay_tolerance: float = 0.5,
    amount_growth_tolerance: float = 0.15,
) -> List[Dict]:
    """Greedily walks forward from each transaction: does its recipient
    re-send a plausibly fee-decayed amount onward, within `max_hop_gap`, to
    a new recipient? Bounded to `max_hops`, so cost is roughly linear in the
    number of transactions times the (small) number of same-sender
    candidates considered per hop -- not an exhaustive search over all
    possible chains, just a useful heuristic signal.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    sender_index: Dict[object, List] = {}
    for idx, sender, ts in zip(df.index, df["sender_id"], df["timestamp"]):
        sender_index.setdefault(sender, []).append((ts, idx))
    for candidates in sender_index.values():
        candidates.sort()

    findings = []
    used_in_chain = set()

    for start_idx in df.index:
        if start_idx in used_in_chain:
            continue

        chain_rows = [start_idx]
        current_recipient = df.at[start_idx, "recipient_id"]
        current_time = df.at[start_idx, "timestamp"]
        current_amount = df.at[start_idx, "transaction_amount"]

        for _ in range(max_hops - 1):
            candidates = sender_index.get(current_recipient, [])
            next_hop = None
            for ts, idx2 in candidates:
                if idx2 in chain_rows or ts <= current_time or ts - current_time > max_hop_gap:
                    continue
                amt2 = df.at[idx2, "transaction_amount"]
                if current_amount * (1 - amount_decay_tolerance) <= amt2 <= current_amount * (1 + amount_growth_tolerance):
                    next_hop = (idx2, ts, amt2)
                    break
            if next_hop is None:
                break
            idx2, ts2, amt2 = next_hop
            chain_rows.append(idx2)
            current_recipient = df.at[idx2, "recipient_id"]
            current_time = ts2
            current_amount = amt2

        if len(chain_rows) >= 3:  # at least 2 hand-offs
            used_in_chain.update(chain_rows)
            findings.append(_finding(
                "layering_chain",
                f"Funds passed through {len(chain_rows)} transactions across a chain of accounts "
                f"within {max_hop_gap}, each hop's amount consistent with the previous (a fee-decayed "
                f"pass-through pattern).",
                SEVERITY_HIGH, chain_rows,
            ))

    return findings


def detect_rapid_pass_through(
    df: pd.DataFrame,
    max_gap: pd.Timedelta = DEFAULT_PASS_THROUGH_GAP,
    min_amount: float = DEFAULT_MIN_PASS_THROUGH_AMOUNT,
) -> List[Dict]:
    """Flags accounts where a meaningful sum arrives and then leaves again
    within `max_gap` -- a mule account being used as a quick pass-through
    rather than a genuine destination."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    outgoing_by_sender: Dict[object, pd.DataFrame] = {
        sender: group.sort_values("timestamp") for sender, group in df.groupby("sender_id")
    }

    findings = []
    for recipient, incoming_group in df.groupby("recipient_id"):
        outgoing = outgoing_by_sender.get(recipient)
        if outgoing is None:
            continue

        for in_idx, in_row in incoming_group.iterrows():
            if in_row["transaction_amount"] < min_amount:
                continue
            window = outgoing[
                (outgoing["timestamp"] > in_row["timestamp"])
                & (outgoing["timestamp"] - in_row["timestamp"] <= max_gap)
            ]
            if not window.empty:
                affected = [in_idx] + window.index.tolist()
                findings.append(_finding(
                    "rapid_pass_through",
                    f"Account received {in_row['transaction_amount']:.2f} and forwarded funds onward "
                    f"again within {max_gap} -- consistent with a pass-through mule account rather than "
                    f"a genuine destination.",
                    SEVERITY_HIGH, affected,
                ))

    return findings


def run_all_aml_pattern_detectors(df: pd.DataFrame) -> List[Dict]:
    findings = []
    findings.extend(detect_layering_chains(df))
    findings.extend(detect_rapid_pass_through(df))
    return findings
