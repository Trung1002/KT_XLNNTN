# -*- coding: utf-8 -*-
"""
lab9_tools.py - Lab 9 (Modern NLP Final Project) - công cụ dùng chung cho SV và GV.

SV chạy:   python lab9_tools.py pack --mssv 2212345
  -> tạo thư mục student_pack_2212345/ gồm dữ liệu RIÊNG + personal_card.json

Mọi tham số cá nhân được sinh TẤT ĐỊNH từ MSSV (SHA-256), nên GV có thể tái tạo lại
đúng thẻ và dữ liệu của từng SV để đối chiếu khi vấn đáp.
"""
import argparse, hashlib, json, math, pathlib, random, re, unicodedata
import pandas as pd

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
SALT = "NLP2026-L9"

TOPICS = ["dang_ky_hoc_phan", "hoc_phi", "hoc_bong", "thi_diem",
          "tot_nghiep", "ky_tuc_xa", "thu_vien_cntt", "ctsv"]

NOISE_TYPES = ["no_diacritics", "teencode", "typo", "code_mix"]
EMBED_MODELS = ["intfloat/multilingual-e5-small",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "bkai-foundation-models/vietnamese-bi-encoder"]
CLF_BACKBONES = {   # tên model: (số lớp, hidden, target_modules cho LoRA)
    "vinai/phobert-base-v2": (12, 768, ["query", "value"]),
    "xlm-roberta-base": (12, 768, ["query", "value"]),
    "distilbert-base-multilingual-cased": (6, 768, ["q_lin", "v_lin"]),
}
ABLATIONS = {
    "A1": "Bỏ chuẩn hóa Unicode NFC + lowercase ở khâu tiền xử lý (so sánh TF-IDF/BM25)",
    "A2": "Tách từ (pyvi) so với tách theo âm tiết cho BM25/TF-IDF",
    "A3": "Learning curve: dùng 25% / 50% / 100% dữ liệu train cho bộ phân loại",
    "A4": "Quét trọng số hybrid alpha = 0.0, 0.1, ..., 1.0 trên tập dev",
}


# ------------------------------------------------------------------ thẻ cá nhân
def _rng(mssv: str) -> random.Random:
    h = hashlib.sha256(f"{SALT}::{mssv}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _ndcg_binary(rels, k):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    n_rel = sum(rels)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / idcg if idcg else 0.0


def toy_ranking_metrics(rels, k=5):
    n_rel = sum(rels)
    hit_k = sum(rels[:k])
    rr = next((1 / (i + 1) for i, r in enumerate(rels) if r), 0.0)
    return {"P@%d" % k: hit_k / k, "R@%d" % k: hit_k / n_rel, "RR": rr, "nDCG@%d" % k: _ndcg_binary(rels, k)}


def dpo_toy(card):
    d = card["dpo"]
    beta = d["beta"]
    rw = beta * (d["logp_pi_chosen"] - d["logp_ref_chosen"])
    rl = beta * (d["logp_pi_rejected"] - d["logp_ref_rejected"])
    h = rw - rl
    loss = -math.log(1 / (1 + math.exp(-h)))
    return {"reward_chosen": rw, "reward_rejected": rl, "margin_h": h, "loss": loss}


def lora_params(card):
    bb = card["clf_backbone"]
    layers, hidden, _ = CLF_BACKBONES[bb]
    r = card["lora"]["r"]
    per_matrix = r * (hidden + hidden)
    return {"per_matrix": per_matrix, "total_qv": layers * 2 * per_matrix,
            "full_matrix": hidden * hidden, "ratio_vs_full_matrix": per_matrix / (hidden * hidden)}


def assign(mssv: str) -> dict:
    mssv = re.sub(r"\s+", "", str(mssv))
    rng = _rng(mssv)
    card = {
        "mssv": mssv,
        "seed": rng.randrange(10_000, 99_999),
        "noise": rng.choice(NOISE_TYPES),
        "topic_pair": sorted(rng.sample(TOPICS, 2)),
        "embed_model": rng.choice(EMBED_MODELS),
        "clf_backbone": rng.choice(list(CLF_BACKBONES)),
        "lora": None,
        "fewshot_k": rng.choice([2, 4, 6]),
        "rag_top_k": rng.choice([2, 3, 5]),
        "ablation": rng.choice(list(ABLATIONS)),
    }
    r = rng.choice([4, 8, 16])
    card["lora"] = {"r": r, "alpha": 2 * r, "dropout": 0.1,
                    "target_modules": CLF_BACKBONES[card["clf_backbone"]][2]}
    card["dpo"] = {"beta": rng.choice([0.1, 0.2, 0.5]),
                   "logp_pi_chosen": round(rng.uniform(-30, -12), 1),
                   "logp_pi_rejected": round(rng.uniform(-45, -20), 1),
                   "logp_ref_chosen": round(rng.uniform(-32, -14), 1),
                   "logp_ref_rejected": round(rng.uniform(-42, -18), 1)}
    # xếp hạng giả định: 8 kết quả, đúng 2 kết quả liên quan
    pos = sorted(rng.sample(range(8), 2))
    card["toy_ranking"] = [1 if i in pos else 0 for i in range(8)]
    card["ablation_desc"] = ABLATIONS[card["ablation"]]
    card["check_code"] = hashlib.sha256(
        json.dumps(card, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8].upper()
    return card


# ------------------------------------------------------------------ nhiễu văn bản
def strip_diacritics(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


TEEN = [("sinh viên", "sv"), ("giảng viên", "gv"), ("không", "ko"), ("được", "đc"), ("bao nhiêu", "bn"),
        ("thế nào", "ntn"), ("như thế nào", "ntn"), ("với", "vs"), ("mình", "m"), ("vậy", "v"),
        ("nhưng", "nhg"), ("học phần", "hp"), ("học kỳ", "hk"), ("ký túc xá", "ktx"), ("thông tin", "tt")]
CODEMIX = [("học phí", "tuition"), ("điểm", "grade"), ("đăng ký", "register"), ("học bổng", "scholarship"),
           ("hồ sơ", "document"), ("thời hạn", "deadline"), ("lệ phí", "fee"), ("nộp", "submit"),
           ("thư viện", "library"), ("tốt nghiệp", "graduation"), ("bảng điểm", "transcript")]


def add_noise(text: str, kind: str, rng: random.Random) -> str:
    t = text
    if kind == "no_diacritics":
        return strip_diacritics(t)
    if kind == "teencode":
        for a, b in sorted(TEEN, key=lambda p: -len(p[0])):
            t = re.sub(rf"\b{a}\b", b, t, flags=re.I)
        return t.lower().replace("?", "").replace(",", "")
    if kind == "code_mix":
        for a, b in sorted(CODEMIX, key=lambda p: -len(p[0])):
            t = re.sub(rf"\b{a}\b", b, t, flags=re.I)
        return t
    if kind == "typo":
        words = t.split(" ")
        for i, w in enumerate(words):
            if len(w) > 3 and rng.random() < 0.18:
                op = rng.choice(["swap", "drop", "dup"])
                j = rng.randrange(1, len(w) - 1)
                if op == "swap":
                    w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
                elif op == "drop":
                    w = w[:j] + w[j + 1:]
                else:
                    w = w[:j] + w[j] + w[j:]
                words[i] = w
        return " ".join(words)
    raise ValueError(kind)


# ------------------------------------------------------------------ tạo gói dữ liệu
def build_pack(mssv: str, out_root=".") -> pathlib.Path:
    card = assign(mssv)
    seed = card["seed"]
    rng = random.Random(seed)
    kb = pd.read_csv(DATA / "kb_passages.csv")
    q = pd.read_csv(DATA / "queries_all.csv")
    ood = pd.read_csv(DATA / "ood_queries.csv")

    # --- tách THEO ITEM (group split): ~22 item test, phân tầng theo chủ đề (3 item/chủ đề, riêng 2 chủ đề trong topic_pair lấy 2)
    items = kb[["item_id", "topic"]]
    test_items = []
    for tp, g in items.groupby("topic"):
        ids = sorted(g["item_id"])
        test_items += rng.sample(ids, 2 if tp in card["topic_pair"] else 3)
    test_items = sorted(test_items)
    train_items = sorted(set(items["item_id"]) - set(test_items))

    clf_train = q[q.item_id.isin(train_items)].sample(frac=1, random_state=seed).reset_index(drop=True)
    clf_test = q[q.item_id.isin(test_items)].reset_index(drop=True)

    nrng = random.Random(seed + 7)
    clf_test_noisy = clf_test.copy()
    clf_test_noisy["text"] = [add_noise(t, card["noise"], nrng) for t in clf_test["text"]]

    # --- truy hồi: dev lấy từ item train (để chỉnh alpha), test lấy từ item test
    def sample_q(df, per_item, r):
        parts = []
        for _, g in df.groupby("item_id"):
            parts.append(g.sample(n=per_item, random_state=r.randrange(10**6)))
        return pd.concat(parts).reset_index(drop=True)

    retr_dev = sample_q(q[q.item_id.isin(train_items)], 2, rng)
    retr_test = sample_q(q[q.item_id.isin(test_items)], 6, rng)
    retr_test_noisy = retr_test.copy()
    retr_test_noisy["text"] = [add_noise(t, card["noise"], nrng) for t in retr_test["text"]]

    # --- LLM: 30 câu test + pool few-shot từ train (5 mẫu/chủ đề) + 10 câu OOD
    llm_eval = clf_test.sample(n=30, random_state=seed).reset_index(drop=True)
    pool = (clf_train.groupby("topic", group_keys=False)
            .apply(lambda g: g.sample(n=5, random_state=seed)).reset_index(drop=True))
    ood_s = ood.sample(n=10, random_state=seed).reset_index(drop=True)

    out = pathlib.Path(out_root) / f"student_pack_{card['mssv']}"
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "kb_passages.csv": kb, "clf_train.csv": clf_train, "clf_test.csv": clf_test,
        "clf_test_noisy.csv": clf_test_noisy, "retr_dev.csv": retr_dev, "retr_test.csv": retr_test,
        "retr_test_noisy.csv": retr_test_noisy, "llm_eval_30.csv": llm_eval,
        "fewshot_pool.csv": pool, "ood_10.csv": ood_s,
    }
    for name, df in files.items():
        df.to_csv(out / name, index=False, encoding="utf-8-sig")
    (out / "preference_pairs.jsonl").write_bytes((DATA / "preference_pairs.jsonl").read_bytes())
    card["split_info"] = {"train_items": len(train_items), "test_items": len(test_items),
                          "n_train": len(clf_train), "n_test": len(clf_test)}
    with open(out / "personal_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, ensure_ascii=False, indent=2)
    return out


def print_card(card):
    print("=" * 64)
    print(f"THẺ CÁ NHÂN - MSSV {card['mssv']}   (check_code: {card['check_code']})")
    print("=" * 64)
    for k in ["seed", "noise", "topic_pair", "embed_model", "clf_backbone", "lora", "fewshot_k",
              "rag_top_k", "ablation", "ablation_desc", "dpo", "toy_ranking"]:
        print(f"{k:14s}: {card[k]}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="tạo gói dữ liệu cá nhân")
    p.add_argument("--mssv", required=True)
    p.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.cmd == "pack":
        out = build_pack(a.mssv, a.out)
        card = json.load(open(out / "personal_card.json", encoding="utf-8"))
        print_card(card)
        print("Đã tạo:", out)


if __name__ == "__main__":
    main()
