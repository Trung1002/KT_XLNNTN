# -*- coding: utf-8 -*-
# Lab 9 - Modern NLP Final Project Development  |  STARTER (mở bằng VS Code / Colab / Jupytext)
# Mỗi "# %%" là một cell. Các cell đánh dấu TODO là phần SV phải tự hoàn thành.
# Cell 0-4 chạy được ngay (baseline). Không sửa các hàm đo (metrics) trừ khi giải thích được lý do.

# %% 0. Cài đặt (Colab)  -- bỏ comment nếu cần
# !pip -q install pandas scikit-learn sentence-transformers peft transformers datasets pyvi

# %% 0. Cấu hình
import json, math, re, pathlib, unicodedata
import numpy as np, pandas as pd
from collections import Counter, defaultdict

MSSV = "23DH114659"                                   # TODO: thay bằng MSSV của em
PACK = pathlib.Path(f"student_pack_{MSSV}")        # tạo bằng: python lab9_tools.py pack --mssv <MSSV>
card = json.load(open(PACK / "personal_card.json", encoding="utf-8"))
SEED = card["seed"]
np.random.seed(SEED)
print("check_code:", card["check_code"], "| noise:", card["noise"], "| topic_pair:", card["topic_pair"])

# %% 1. Nạp dữ liệu
rd = lambda f: pd.read_csv(PACK / f)
kb, tr, te, te_noisy = rd("kb_passages.csv"), rd("clf_train.csv"), rd("clf_test.csv"), rd("clf_test_noisy.csv")
dev, rq, rq_noisy = rd("retr_dev.csv"), rd("retr_test.csv"), rd("retr_test_noisy.csv")
print(tr.shape, te.shape, rq.shape)
print(tr.topic.value_counts())

# Kiểm tra RÒ RỈ DỮ LIỆU: không item nào xuất hiện ở cả train và test
assert set(tr.item_id).isdisjoint(set(te.item_id)), "Rò rỉ dữ liệu: item trùng giữa train/test!"

# %% 2. Tiền xử lý (TODO: giải thích từng bước trong báo cáo)
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)          # bỏ dấu câu
    return re.sub(r"\s+", " ", s).strip()

def tok(s: str):                              # tách theo âm tiết (khoảng trắng)
    return normalize(s).split()

# %% 3. Baseline A - phân loại chủ đề: TF-IDF + Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform(tr.text)
clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, tr.topic)
res = {}
for name, df in [("clean", te), ("noisy", te_noisy)]:
    pred = clf.predict(vec.transform(df.text))
    res[name] = {"acc": accuracy_score(df.topic, pred), "macro_f1": f1_score(df.topic, pred, average="macro")}
print("TF-IDF+LR:", res)

# %% 4. Baseline B - truy hồi BM25 (tự cài đặt) + các metric truy hồi
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tok(d)) for d in docs]
        self.dl = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.dl.mean()
        df = Counter(w for c in self.tf for w in c)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, q):
        s = np.zeros(len(self.tf))
        for w in tok(q):
            if w not in self.idf:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(w, 0)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return s

def rank_of_gold(score_vec, gold_idx):
    """Thứ hạng (bắt đầu từ 1) của đáp án đúng; điểm bằng nhau -> tính theo thứ tự chỉ số (tất định)."""
    order = np.lexsort((np.arange(len(score_vec)), -score_vec))
    return int(np.where(order == gold_idx)[0][0]) + 1

def retrieval_metrics(ranks, ks=(1, 3, 5, 10)):
    ranks = np.asarray(ranks)
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["mrr@10"] = float(np.mean([1 / r if r <= 10 else 0 for r in ranks]))
    out["ndcg@10"] = float(np.mean([1 / math.log2(r + 1) if r <= 10 else 0 for r in ranks]))  # 1 tài liệu đúng -> IDCG = 1
    return out

pid2idx = {p: i for i, p in enumerate(kb.passage_id)}
bm25 = BM25(kb.text.tolist())
def eval_bm25(df):
    ranks = [rank_of_gold(bm25.scores(t), pid2idx[g]) for t, g in zip(df.text, df.gold_passage_id)]
    return retrieval_metrics(ranks)
print("BM25 clean:", eval_bm25(rq))
print("BM25 noisy:", eval_bm25(rq_noisy))

# %% 5. TODO - Dense retrieval với card["embed_model"]
# Gợi ý:
#   from sentence_transformers import SentenceTransformer
#   m = SentenceTransformer(card["embed_model"])
#   - multilingual-e5: thêm tiền tố "query: " / "passage: "
#   - vietnamese-bi-encoder: cần tách từ trước (pyvi.ViTokenizer.tokenize)
#   - encode(..., normalize_embeddings=True) rồi cosine = tích vô hướng
# YÊU CẦU: tính retrieval_metrics cho rq (clean) và rq_noisy; so sánh với BM25.
dense_scores = None   # TODO: ma trận [n_query x n_passage]

# %% 6. TODO - Hybrid: score = alpha * dense_norm + (1 - alpha) * bm25_norm
# Chuẩn hóa điểm (min-max theo từng truy vấn) -> chọn alpha tốt nhất trên `dev` (KHÔNG chọn trên test) -> báo cáo trên rq.
alpha_best = None     # TODO

# %% 7. TODO - Phân loại "hiện đại"
# (7a) BẮT BUỘC: embedding đóng băng (card["embed_model"]) + LogisticRegression -> acc/macro-F1 trên te và te_noisy
# (7b) NÂNG CAO (Colab GPU): LoRA fine-tune card["clf_backbone"] với card["lora"] (r, alpha, dropout, target_modules)
#      from peft import LoraConfig, get_peft_model, TaskType
#      cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=..., lora_alpha=..., lora_dropout=..., target_modules=card["lora"]["target_modules"])
#      model.print_trainable_parameters()   # ghi lại con số này

# %% 8. TODO - Prompting zero-shot vs few-shot (llm_eval_30.csv, fewshot_pool.csv)
# - Viết prompt phân loại chủ đề (8 nhãn). temperature = 0.
# - Few-shot dùng k = card["fewshot_k"] ví dụ TỔNG CỘNG lấy từ fewshot_pool.csv -> ghi rõ cách chọn (ngẫu nhiên theo SEED hay cân bằng nhãn).
# - Ghi lại: tên model + phiên bản, prompt đầy đủ, accuracy zero-shot và k-shot, 3 câu sai tiêu biểu.

# %% 9. TODO - RAG + từ chối trả lời (abstain)
# Quy trình: truy hồi top-K (K = card["rag_top_k"]) -> ghép ngữ cảnh -> sinh/trích xuất câu trả lời.
# - answer_hit_rate: câu trả lời chứa gold_answer (đã normalize) trên rq (lấy mẫu >= 60 câu).
# - Với ood_10.csv: đặt NGƯỠNG điểm truy hồi để từ chối ("Không tìm thấy thông tin trong quy định").
#   Báo cáo abstain_rate_ood và false_abstain_rate (từ chối nhầm câu hợp lệ) ở ngưỡng đã chọn.

# %% 10. LoRA & DPO "tính tay + kiểm bằng code" (dùng số liệu trong personal_card.json)
def lora_trainable(layers, hidden, r, n_targets=2):
    """TODO: trả về số tham số LoRA khi gắn vào n_targets ma trận (hidden x hidden) mỗi lớp."""
    raise NotImplementedError

def dpo_loss(beta, lp_pi_w, lp_pi_l, lp_ref_w, lp_ref_l):
    """TODO: loss = -log sigmoid( beta * ((pi_w - ref_w) - (pi_l - ref_l)) )"""
    raise NotImplementedError

def toy_metrics(rels, k=5):
    """TODO: trả về P@k, R@k, RR, nDCG@k cho danh sách nhị phân `rels` (card['toy_ranking'])."""
    raise NotImplementedError

# %% 11. TODO - Ablation theo card["ablation"]  (A1 / A2 / A3 / A4)
# Báo cáo bảng "bỏ/đổi 1 yếu tố" + 2-3 câu giải thích vì sao kết quả như vậy (kèm số liệu).

# %% 12. TODO - Phân tích lỗi + dữ liệu tự tạo
# - Confusion matrix giữa 2 chủ đề trong card["topic_pair"]; liệt kê >= 5 câu sai, phân loại nguyên nhân lỗi.
# - Tự viết 10 câu hỏi MỚI cho MỖI chủ đề trong topic_pair (không copy mẫu câu có sẵn), lưu own_queries.csv
#   (cột: text, topic, gold_passage_id) và đo Recall@3 của hệ thống trên các câu này.

# %% 13. Ghi kết quả (nộp kèm notebook)
def save_results(results: dict, path="results.json"):
    results = {"mssv": card["mssv"], "check_code": card["check_code"], **results}
    json.dump(results, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print("Đã lưu", path)

# Ví dụ cấu trúc (điền số thật của em):
# save_results({
#   "clf":  {"tfidf_lr": res, "emb_lr": {...}, "lora": {"trainable_params": None}},
#   "retrieval": {"bm25_clean": ..., "bm25_noisy": ..., "dense_clean": ..., "dense_noisy": ...,
#                 "hybrid": ..., "alpha_best": alpha_best},
#   "prompting": {"model": "...", "zero_shot_acc": None, "k_shot_acc": None},
#   "rag": {"answer_hit_rate": None, "abstain_rate_ood": None, "false_abstain_rate": None},
#   "toy": {"ranking": {...}, "dpo": {...}, "lora_params": None},
#   "ablation": {"id": card["ablation"], "table": {...}},
# })
