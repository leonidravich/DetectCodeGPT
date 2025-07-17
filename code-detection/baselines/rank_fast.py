import torch
import torch.nn.functional as F
from typing import List, Literal, Tuple, Union, Optional

@torch.inference_mode()
def _score_ll_and_rank_batch(
    model,
    tokenizer,
    texts: List[str],
    device: Union[torch.device, str],
    reduction: Literal["sum","mean_per_token"]="mean_per_token",
    max_length: Optional[int] = None,
    pad_to_multiple_of: int = 8,
):
    """
    Vectorized LL + rank scorer for a batch of variable-length texts.

    Returns:
        ll_scores      [B]  tensor (sum or mean NLL; see `reduction`)
        rank_mean      [B]  tensor (mean rank over valid tokens)
        logrank_mean   [B]  tensor (mean log-rank over valid tokens)
    """
    if len(texts) == 0:
        # Return empty tensors
        z = torch.empty(0, device=device)
        return z, z, z

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True if max_length else False,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        add_special_tokens=True,
    )
    # Some tokenizers emit token_type_ids we don't need
    if "token_type_ids" in enc:
        enc.pop("token_type_ids")

    input_ids   = enc["input_ids"].to(device)
    attn_mask   = enc["attention_mask"].to(device)

    # Shift to create inputs/labels
    input_ids_in = input_ids[:, :-1]
    label_ids    = input_ids[:, 1:]
    attn_mask_in = attn_mask[:, 1:]  # valid label positions

    # Forward pass
    outputs = model(input_ids_in, attention_mask=attn_mask[:, :-1])
    logits  = outputs.logits  # [B, T-1, V]

    B, Tm1, V = logits.shape

    # Gather gold logits
    gold_logits = logits.gather(-1, label_ids.unsqueeze(-1))  # [B, T-1, 1]

    # Rank: count how many logits are strictly greater than gold logit
    # NOTE: using > matches argsort semantics for unique floats
    rank = (logits > gold_logits).sum(dim=-1, dtype=torch.int32) + 1  # [B, T-1]

    # Log-rank
    logrank = torch.log(rank.to(torch.float32))  # [B, T-1]

    # NLL per token
    nll_tok = F.cross_entropy(
        logits.reshape(-1, V),
        label_ids.reshape(-1),
        reduction="none"
    ).reshape(B, Tm1)  # [B, T-1]

    # Mask out padded positions (attn_mask_in=0)
    mask = attn_mask_in.to(torch.bool)
    valid_counts = mask.sum(dim=-1).clamp(min=1)

    nll_sum = (nll_tok * mask).sum(dim=-1)
    rank_mean    = (rank.to(torch.float32)    * mask).sum(dim=-1) / valid_counts
    logrank_mean = (logrank                  * mask).sum(dim=-1) / valid_counts

    if reduction == "mean_per_token":
        ll_scores = nll_sum / valid_counts
    else:  # "sum"
        ll_scores = nll_sum

    return ll_scores, rank_mean, logrank_mean


# ------------------------------------------------------------------
# Backward-compatible API shims
# ------------------------------------------------------------------

def get_rank_fast(text: str, args, model_config, log: bool = False) -> float:
    """
    API-compatible with your legacy get_rank(). Returns a *mean* (log-)rank float.
    """
    _, rank_mean, logrank_mean = _score_ll_and_rank_batch(
        model_config["base_model"],
        model_config["base_tokenizer"],
        [text],
        device=args.DEVICE,
        reduction="mean_per_token",  # legacy averaged per token
    )
    return (logrank_mean if log else rank_mean).item()


def get_ranks_fast(
    texts: List[str],
    args,
    model_config,
    log: bool = True,
    batch_size: int = 32,
) -> List[float]:
    """
    Batched version: splits texts into chunks, scores each, concatenates.
    """
    out: List[float] = []
    model   = model_config["base_model"]
    tok     = model_config["base_tokenizer"]
    dev     = args.DEVICE

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        _, rank_mean, logrank_mean = _score_ll_and_rank_batch(
            model, tok, chunk, device=dev, reduction="mean_per_token"
        )
        vals = logrank_mean if log else rank_mean
        out.extend(vals.tolist())
        
        # Clear GPU memory after each batch to prevent accumulation
        if dev == 'cuda':
            torch.cuda.empty_cache()
    return out 