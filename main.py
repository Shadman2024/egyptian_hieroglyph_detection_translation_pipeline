import os
import json
import math
import re
import time
from dataclasses import dataclass
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------
# Config
# ---------------------------------
st.set_page_config(page_title="Egyptian Hieroglyph Pipeline", layout="wide")
st.title("Egyptian Hieroglyph 5-Stage Pipeline")
st.write(
    "Upload image -> Stage 1 region detection -> Stage 2 symbol detection "
    "-> Stage 3 symbol classification -> Stage 4 transliteration -> Stage 5 translation"
)

DEFAULT_MODEL_A_PATH = "weights/MODEL1_LINE_CARTOUCHE_WEIGHTS.pt"
DEFAULT_MODEL_B_PATH = "weights/MODEL2_GLYPH_DETECT_WEIGHTS.pt"
DEFAULT_MODEL_C_PATH = "weights/MODEL3_CLASSIFIER_CKPT.pth"
DEFAULT_MODEL3_CLASS_DIR = "Glyph2025"

DEFAULT_TRANSLATION_MODEL_NAME = "Asiif/mt5_hieroglyph"
DEFAULT_TRANSLATION_MODEL_COMMIT = "846f2dd0bc50c77fff26d1188021232955e41e19"
DEFAULT_GRAMMAR_MODEL_NAME = "google/flan-t5-base"

TASK_PREFIX = "translate Egyptian transliteration to English: "
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 64

RAW_BASE = "https://raw.githubusercontent.com/SeveredAsif/HieroGlyphTranslator/main"
FILES = {
    "train_egy": "aligned_train_egy.csv",
    "train_eng": "aligned_train_eng.csv",
    "val_egy": "aligned_val_egy.csv",
    "val_eng": "aligned_val_eng.csv",
    "test_egy": "aligned_test_egy.csv",
    "test_eng": "aligned_test_eng.csv",
    "dict": "ancient_egypt_dictionary_compiled.csv",
}

LOCAL_DATA_DIR = os.path.join("data", "hiero_model4_resources")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_CACHE_DIR = os.path.join("weights", "hf_cache")

CLASSIFIER_IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

# ---------------------------------
# Utility helpers
# ---------------------------------
def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    out = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed


def discover_weight_files(base_dir="weights"):
    candidates = []
    exts = (".pt", ".pth", ".bin", ".ckpt")
    if os.path.isdir(base_dir):
        for root, _, files in os.walk(base_dir):
            for name in files:
                if name.lower().endswith(exts):
                    candidates.append(os.path.join(root, name).replace("\\", "/"))
    return sorted(set(candidates))


# ---------------------------------
# Sidebar: model selection
# ---------------------------------
with st.sidebar:
    st.header("Model Selection")
    available_weights = discover_weight_files("weights")

    model_a_options = [DEFAULT_MODEL_A_PATH] + [p for p in available_weights if p != DEFAULT_MODEL_A_PATH]
    model_b_options = [DEFAULT_MODEL_B_PATH] + [p for p in available_weights if p != DEFAULT_MODEL_B_PATH]
    model_c_options = [DEFAULT_MODEL_C_PATH] + [p for p in available_weights if p != DEFAULT_MODEL_C_PATH]

    model_a_path = st.selectbox("Model 1 weights", model_a_options, index=0)
    model_b_path = st.selectbox("Model 2 weights", model_b_options, index=0)
    model_c_path = st.selectbox("Model 3 checkpoint", model_c_options, index=0)

    model_a_path_override = st.text_input("Model 1 custom path (optional)", "").strip()
    model_b_path_override = st.text_input("Model 2 custom path (optional)", "").strip()
    model_c_path_override = st.text_input("Model 3 custom path (optional)", "").strip()

    class_dir = st.text_input("Class folder path", DEFAULT_MODEL3_CLASS_DIR).strip() or DEFAULT_MODEL3_CLASS_DIR

    st.subheader("HF Models")
    translation_model_name = st.text_input("Model 4 translation model", DEFAULT_TRANSLATION_MODEL_NAME).strip() or DEFAULT_TRANSLATION_MODEL_NAME
    translation_model_commit = st.text_input("Model 4 revision/commit", DEFAULT_TRANSLATION_MODEL_COMMIT).strip() or DEFAULT_TRANSLATION_MODEL_COMMIT
    grammar_model_name = st.text_input("Model 5 grammar model", DEFAULT_GRAMMAR_MODEL_NAME).strip() or DEFAULT_GRAMMAR_MODEL_NAME

    hf_cache_dir = st.text_input("HF cache directory", HF_CACHE_DIR).strip() or HF_CACHE_DIR
    os.makedirs(hf_cache_dir, exist_ok=True)

model_a_path = model_a_path_override or model_a_path
model_b_path = model_b_path_override or model_b_path
model_c_path = model_c_path_override or model_c_path


# ---------------------------------
# Model 3 classifier
# ---------------------------------
class SymbolClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------
# Model 4 helpers
# ---------------------------------
sign_pattern = re.compile(r"[A-Za-z]{1,2}\d+[A-Za-z]?", flags=re.IGNORECASE)


def canonicalize_sign(token: str) -> str:
    t = str(token).strip()
    m = re.fullmatch(r"([A-Za-z]{1,2})(\d+)([A-Za-z]?)", t)
    if not m:
        return t
    fam, num, suf = m.groups()
    fam = fam[0].upper() + (fam[1:].lower() if len(fam) > 1 else "")
    suf = suf.upper() if suf else ""
    return f"{fam}{num}{suf}"


def tokenize_signs(text: str):
    raw = sign_pattern.findall(str(text))
    if raw:
        return [canonicalize_sign(x) for x in raw]

    compact = re.sub(r"[^A-Za-z0-9]", "", str(text))
    chunks = sign_pattern.findall(compact)
    if chunks:
        return [canonicalize_sign(x) for x in chunks]

    return []


def expand_gardiner_variants(raw_key: str):
    parts = re.split(r"\s+or\s+|/|\|", str(raw_key).strip(), flags=re.IGNORECASE)
    variants = []
    for p in parts:
        toks = tokenize_signs(p.strip())
        if toks:
            variants.append(tuple(toks))
    return variants


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_local_data():
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    for fname in FILES.values():
        out_path = os.path.join(LOCAL_DATA_DIR, fname)
        if os.path.exists(out_path):
            continue

        url = f"{RAW_BASE}/{fname}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    return LOCAL_DATA_DIR


def split_translit_tokens(line: str):
    return [t for t in line.strip().split() if t]


# ---------------------------------
# Load everything with Streamlit cache
# ---------------------------------
@st.cache_resource
def load_all_resources(
    model_a_path,
    model_b_path,
    model_c_path,
    class_dir,
    translation_model_name,
    translation_model_commit,
    grammar_model_name,
    hf_cache_dir,
    hf_token,
):
    if not os.path.exists(model_a_path):
        raise FileNotFoundError(f"Model A not found: {model_a_path}")
    if not os.path.exists(model_b_path):
        raise FileNotFoundError(f"Model B not found: {model_b_path}")
    if not os.path.exists(model_c_path):
        raise FileNotFoundError(f"Model C not found: {model_c_path}")
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    class_names = sorted([
        d for d in os.listdir(class_dir)
        if os.path.isdir(os.path.join(class_dir, d))
    ])

    model_a = YOLO(model_a_path)
    model_b = YOLO(model_b_path)

    checkpoint = torch.load(model_c_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    normalized_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            normalized_state_dict[k] = v
        else:
            normalized_state_dict[f"backbone.{k}"] = v

    last_linear_weight = normalized_state_dict["backbone.fc.3.weight"]
    ckpt_num_classes = int(last_linear_weight.shape[0])

    model_c = SymbolClassifier(num_classes=ckpt_num_classes)
    model_c.load_state_dict(normalized_state_dict, strict=True)
    model_c.to(DEVICE)
    model_c.eval()

    tok_kwargs = {
        "use_fast": False,
        "revision": translation_model_commit,
        "cache_dir": hf_cache_dir,
    }
    mdl_kwargs = {
        "revision": translation_model_commit,
        "cache_dir": hf_cache_dir,
    }
    if hf_token:
        tok_kwargs["token"] = hf_token
        mdl_kwargs["token"] = hf_token

    translator_tokenizer = AutoTokenizer.from_pretrained(
        translation_model_name,
        **tok_kwargs
    )
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(
        translation_model_name,
        **mdl_kwargs
    ).to(DEVICE)
    translator_model.eval()

    grammar_tok_kwargs = {"cache_dir": hf_cache_dir}
    grammar_model_kwargs = {"cache_dir": hf_cache_dir}
    if hf_token:
        grammar_tok_kwargs["token"] = hf_token
        grammar_model_kwargs["token"] = hf_token

    grammar_tokenizer = AutoTokenizer.from_pretrained(
        grammar_model_name,
        **grammar_tok_kwargs
    )
    grammar_model = AutoModelForSeq2SeqLM.from_pretrained(
        grammar_model_name,
        **grammar_model_kwargs
    ).to(DEVICE)
    grammar_model.eval()

    data_dir = ensure_local_data()
    train_egy = read_lines(os.path.join(data_dir, FILES["train_egy"]))
    train_eng = read_lines(os.path.join(data_dir, FILES["train_eng"]))
    val_egy = read_lines(os.path.join(data_dir, FILES["val_egy"]))
    val_eng = read_lines(os.path.join(data_dir, FILES["val_eng"]))
    test_egy = read_lines(os.path.join(data_dir, FILES["test_egy"]))
    test_eng = read_lines(os.path.join(data_dir, FILES["test_eng"]))

    assert len(train_egy) == len(train_eng), "Train split misaligned"
    assert len(val_egy) == len(val_eng), "Val split misaligned"
    assert len(test_egy) == len(test_eng), "Test split misaligned"

    split_data = {
        "train": {"egy": train_egy, "eng": train_eng},
        "val": {"egy": val_egy, "eng": val_eng},
        "test": {"egy": test_egy, "eng": test_eng},
    }

    df_dict = pd.read_csv(os.path.join(data_dir, FILES["dict"]))
    df_dict = df_dict.dropna(subset=["Gardiner", "Transliteration"])

    phrase_to_trans = defaultdict(Counter)
    for _, row in df_dict.iterrows():
        gard = str(row["Gardiner"]).strip()
        trans = str(row["Transliteration"]).strip()
        if not gard or not trans:
            continue
        for variant in expand_gardiner_variants(gard):
            phrase_to_trans[variant][trans] += 1

    if not phrase_to_trans:
        raise RuntimeError("Dictionary loading failed: no Gardiner phrases found.")

    max_phrase_len = max(len(k) for k in phrase_to_trans.keys())

    unigram = Counter()
    bigram = Counter()
    for line in train_egy + val_egy:
        toks = ["<s>"] + [t for t in line.split() if t] + ["</s>"]
        for i in range(len(toks) - 1):
            unigram[toks[i]] += 1
            bigram[(toks[i], toks[i + 1])] += 1
        unigram[toks[-1]] += 1
    vocab_size = len(unigram)

    known_signs = sorted({s for phrase in phrase_to_trans.keys() for s in phrase})
    known_sign_set = set(known_signs)

    known_families = defaultdict(list)
    for s in known_signs:
        m = re.fullmatch(r"([A-Za-z]{1,2})(\d+)([A-Za-z]?)", s)
        if m:
            fam, num, suf = m.groups()
            known_families[(fam, suf)].append((int(num), s))
    for k in known_families:
        known_families[k] = sorted(known_families[k], key=lambda x: x[0])

    phrase_keys_by_len = defaultdict(list)
    for key in phrase_to_trans.keys():
        phrase_keys_by_len[len(key)].append(key)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
        "translator_tokenizer": translator_tokenizer,
        "translator_model": translator_model,
        "grammar_tokenizer": grammar_tokenizer,
        "grammar_model": grammar_model,
        "class_names": class_names,
        "ckpt_num_classes": ckpt_num_classes,
        "phrase_to_trans": phrase_to_trans,
        "max_phrase_len": max_phrase_len,
        "unigram": unigram,
        "bigram": bigram,
        "vocab_size": vocab_size,
        "known_sign_set": known_sign_set,
        "known_families": known_families,
        "phrase_keys_by_len": phrase_keys_by_len,
        "split_data": split_data,
    }


try:
    RES = load_all_resources(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        model_c_path=model_c_path,
        class_dir=class_dir,
        translation_model_name=translation_model_name,
        translation_model_commit=translation_model_commit,
        grammar_model_name=grammar_model_name,
        hf_cache_dir=hf_cache_dir,
        hf_token=HF_TOKEN,
    )

    model_a = RES["model_a"]
    model_b = RES["model_b"]
    model_c = RES["model_c"]
    translator_tokenizer = RES["translator_tokenizer"]
    translator_model = RES["translator_model"]
    grammar_tokenizer = RES["grammar_tokenizer"]
    grammar_model = RES["grammar_model"]
    CLASS_NAMES = RES["class_names"]

    CKPT_NUM_CLASSES = RES["ckpt_num_classes"]
    phrase_to_trans = RES["phrase_to_trans"]
    max_phrase_len = RES["max_phrase_len"]
    unigram = RES["unigram"]
    bigram = RES["bigram"]
    vocab_size = RES["vocab_size"]
    known_sign_set = RES["known_sign_set"]
    known_families = RES["known_families"]
    phrase_keys_by_len = RES["phrase_keys_by_len"]
    SPLIT_DATA = RES["split_data"]
except Exception as e:
    st.error(f"Failed to load models/resources: {e}")
    st.stop()


# ---------------------------------
# Classifier transform
# ---------------------------------
classifier_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------------------------
# Drawing helpers
# ---------------------------------
def draw_boxes(image, boxes, model, color=(0, 255, 0)):
    output = image.copy()
    if boxes is None or len(boxes) == 0:
        return output

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else None

    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_bgr, (x1, y1), (x2, y2), color, 2)

        label = ""
        if classes is not None:
            cls_id = int(classes[i])
            class_name = model.names.get(cls_id, str(cls_id))
            label = class_name

        if confs is not None:
            label = f"{label} {confs[i]:.2f}".strip()

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top_y = max(y1 - th - 8, 0)
            cv2.rectangle(output_bgr, (x1, top_y), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                output_bgr,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)


def crop_stage1_regions(image_rgb, boxes):
    crops = []
    if boxes is None or len(boxes) == 0:
        return crops

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    h, w = image_rgb.shape[:2]

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crops.append({
            "index": int(i),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "image": crop
        })

    return crops


# ---------------------------------
# Stage 3 classification
# ---------------------------------
def idx_to_class_name(idx: int) -> str:
    if 0 <= idx < len(CLASS_NAMES):
        return str(CLASS_NAMES[idx])
    return str(idx)


def classify_symbol(symbol_rgb, topk=3):
    tensor = classifier_transform(symbol_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model_c(tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=min(topk, CKPT_NUM_CLASSES), dim=1)

    predictions = []
    for idx, prob in zip(top_idxs[0], top_probs[0]):
        pred_idx = int(idx.item())
        predictions.append({
            "label": idx_to_class_name(pred_idx),
            "index": pred_idx,
            "confidence": float(prob.item())
        })

    return predictions


def detect_symbols_only(line_rgb, result_b, margin=4, scale_factor=3):
    annotated = line_rgb.copy()
    extracted_symbol_crops = []

    if result_b.boxes is None or len(result_b.boxes) == 0:
        return annotated, extracted_symbol_crops

    boxes = result_b.boxes.xyxy.cpu().numpy().astype(int)
    h, w = line_rgb.shape[:2]
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)

        crop = line_rgb[y1:y2, x1:x2]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        new_w = int(crop.shape[1] * scale_factor)
        new_h = int(crop.shape[0] * scale_factor)
        crop_big = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(crop_big, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        extracted_symbol_crops.append({
            "index": int(i),
            "image": gray_rgb,
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })

        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated, extracted_symbol_crops


def classify_symbol_crops(symbol_crops, classifier_conf_threshold=0.30):
    classified = []
    total_classifier_seconds = 0.0
    attempted = 0

    for item in symbol_crops:
        attempted += 1
        predictions, t_model3 = timed_call(classify_symbol, item["image"], 3)
        total_classifier_seconds += t_model3

        pred_label = str(predictions[0]["label"])
        pred_conf = float(predictions[0]["confidence"])
        if pred_conf < classifier_conf_threshold:
            continue

        classified.append({
            "index": int(item["index"]),
            "image": item["image"],
            "label": pred_label,
            "confidence": pred_conf,
            "top3": predictions,
            "box": [int(x) for x in item["box"]],
            "model3_inference_sec": float(t_model3),
        })

    return classified, total_classifier_seconds, attempted


# ---------------------------------
# Stage 4 resolver
# ---------------------------------
def log_bigram_prob(prev_tok: str, tok: str, alpha: float = 0.1):
    num = bigram[(prev_tok, tok)] + alpha
    den = unigram[prev_tok] + alpha * max(1, vocab_size)
    return math.log(num / den)


def score_translit_sequence(prev_tok: str, tokens):
    s = 0.0
    p = prev_tok
    for t in tokens:
        s += log_bigram_prob(p, t)
        p = t
    return s


def closest_known_sign(raw_sign: str, max_num_gap: int = 3):
    s = canonicalize_sign(raw_sign)
    if s in known_sign_set:
        return s

    m = re.fullmatch(r"([A-Za-z]{1,2})(\d+)([A-Za-z]?)", s)
    if not m:
        return s

    fam, num, suf = m.groups()
    num = int(num)
    bucket = known_families.get((fam, suf), [])
    if not bucket:
        return s

    best_num, best_sign = min(bucket, key=lambda x: abs(x[0] - num))
    return best_sign if abs(best_num - num) <= max_num_gap else s


def normalize_input_signs(signs):
    fixed, changes = [], []
    for s in signs:
        c = closest_known_sign(s)
        fixed.append(c)
        if canonicalize_sign(s) != c:
            changes.append((str(s), str(c)))
    return fixed, changes


def sign_distance(a: str, b: str):
    if a == b:
        return 0.0

    pa = re.fullmatch(r"([A-Za-z]{1,2})(\d+)([A-Za-z]?)", a)
    pb = re.fullmatch(r"([A-Za-z]{1,2})(\d+)([A-Za-z]?)", b)
    if not pa or not pb:
        return 3.0

    fam_a, num_a, suf_a = pa.groups()
    fam_b, num_b, suf_b = pb.groups()

    dist = 0.0
    if fam_a != fam_b:
        dist += 2.0
    if suf_a != suf_b:
        dist += 0.5
    dist += min(abs(int(num_a) - int(num_b)) / 6.0, 2.0)
    return dist


def get_fuzzy_phrase_candidates(normalized, pos: int, max_candidates: int = 16):
    n = len(normalized)
    cands = []
    max_l = min(max_phrase_len, n - pos)

    for l in range(max_l, 0, -1):
        segment = normalized[pos:pos + l]
        for key in phrase_keys_by_len.get(l, []):
            d = sum(sign_distance(segment[i], key[i]) for i in range(l)) / l
            cands.append((key, l, d))

    cands.sort(key=lambda x: (x[2], -x[1]))
    return cands[:max_candidates]


@dataclass
class Hyp:
    pos: int
    prev: str
    out: list
    score: float
    segments: list


def resolve_sign_sequence(signs, beam_size: int = 6, span_bonus: float = 2.5, lm_weight: float = 1.0, fuzzy_penalty: float = 2.5):
    if not signs:
        return [], 0.0, [], []

    normalized, changes = normalize_input_signs(signs)
    beams = [Hyp(pos=0, prev="<s>", out=[], score=0.0, segments=[])]
    n = len(normalized)

    while True:
        if all(h.pos >= n for h in beams):
            break

        new_beams = []
        for h in beams:
            if h.pos >= n:
                new_beams.append(h)
                continue

            found_any = False
            max_l = min(max_phrase_len, n - h.pos)

            for l in range(max_l, 0, -1):
                key = tuple(normalized[h.pos:h.pos + l])
                if key not in phrase_to_trans:
                    continue

                found_any = True
                for tr, freq in phrase_to_trans[key].items():
                    tr_toks = split_translit_tokens(tr)
                    if not tr_toks:
                        continue

                    lm = score_translit_sequence(h.prev, tr_toks)
                    sc = h.score + (lm_weight * lm) + 0.2 * math.log(freq + 1.0) + span_bonus * l
                    seg = {
                        "start": h.pos,
                        "end": h.pos + l,
                        "input_signs": list(normalized[h.pos:h.pos + l]),
                        "matched_key": list(key),
                        "translit": tr,
                        "exact": True,
                        "distance": 0.0
                    }
                    new_beams.append(
                        Hyp(
                            pos=h.pos + l,
                            prev=tr_toks[-1],
                            out=h.out + tr_toks,
                            score=sc,
                            segments=h.segments + [seg]
                        )
                    )

            if not found_any:
                fuzzy = get_fuzzy_phrase_candidates(normalized, h.pos, max_candidates=16)
                for key, l, dist in fuzzy:
                    for tr, freq in phrase_to_trans[key].items():
                        tr_toks = split_translit_tokens(tr)
                        if not tr_toks:
                            continue

                        lm = score_translit_sequence(h.prev, tr_toks)
                        sc = h.score + (lm_weight * lm) + 0.2 * math.log(freq + 1.0) + span_bonus * l - fuzzy_penalty * dist
                        seg = {
                            "start": h.pos,
                            "end": h.pos + l,
                            "input_signs": list(normalized[h.pos:h.pos + l]),
                            "matched_key": list(key),
                            "translit": tr,
                            "exact": False,
                            "distance": float(dist)
                        }
                        new_beams.append(
                            Hyp(
                                pos=h.pos + l,
                                prev=tr_toks[-1],
                                out=h.out + tr_toks,
                                score=sc,
                                segments=h.segments + [seg]
                            )
                        )

        if not new_beams:
            new_beams = [Hyp(pos=n, prev="</s>", out=[], score=-1e9, segments=[])]

        new_beams = sorted(new_beams, key=lambda x: x.score, reverse=True)[:beam_size]
        beams = new_beams

    best = max(beams, key=lambda x: x.score)
    return best.out, best.score, changes, best.segments


def parse_gardiner_input(gardiner_input):
    if isinstance(gardiner_input, list):
        return [canonicalize_sign(str(x).strip()) for x in gardiner_input if str(x).strip()]

    text = str(gardiner_input).strip()
    if not text:
        return []

    parts = [p for p in text.split() if p]
    if parts:
        return [canonicalize_sign(p) for p in parts]

    return tokenize_signs(text)


def _decode_signs(signs, decode_beam_size: int = 6):
    out_tokens, score, changes, segments = resolve_sign_sequence(signs, beam_size=decode_beam_size)
    translit = " ".join(out_tokens).strip()
    exact_ratio = (sum(1 for s in segments if s["exact"]) / max(1, len(segments)))
    return translit, score, changes, segments, exact_ratio


def _generate_local_orders(n: int, max_combinations: int = 20, n_segments: int = 3, max_shuffle_span: int = 3):
    base = tuple(range(n))
    if n <= 1 or max_combinations <= 1:
        return [base]

    rng = np.random.default_rng(42 + n)
    orders = [base]
    seen = {base}
    segments = [arr.tolist() for arr in np.array_split(np.arange(n), max(1, min(n_segments, n)))]

    attempts = 0
    max_attempts = max_combinations * 25
    while len(orders) < max_combinations and attempts < max_attempts:
        attempts += 1
        order = list(base)

        edits = 1 if rng.random() < 0.6 else 2
        for _ in range(edits):
            seg = segments[int(rng.integers(0, len(segments)))]
            if len(seg) <= 1:
                continue

            if rng.random() < 0.6:
                i, j = rng.choice(seg, size=2, replace=False)
                order[int(i)], order[int(j)] = order[int(j)], order[int(i)]
            else:
                span = min(max_shuffle_span, len(seg))
                if span < 2:
                    continue
                w = int(rng.integers(2, span + 1))
                start = int(rng.integers(0, len(seg) - w + 1))
                idx_slice = seg[start:start + w]
                vals = [order[i] for i in idx_slice]
                rng.shuffle(vals)
                for k, idx in enumerate(idx_slice):
                    order[idx] = vals[k]

        t = tuple(order)
        if t not in seen:
            seen.add(t)
            orders.append(t)

    return orders


def optimize_sign_order(signs, max_combinations: int = 20, min_improvement: float = 0.5):
    signs = list(signs)
    if len(signs) <= 2:
        return signs, {"used": False, "reason": "sequence too short", "tested_orders": 1}

    _, base_score, _, _, _ = _decode_signs(signs, decode_beam_size=3)
    orders = _generate_local_orders(
        len(signs),
        max_combinations=max_combinations,
        n_segments=3,
        max_shuffle_span=3
    )

    best_order = orders[0]
    best_score = base_score

    for order in orders:
        cand = [signs[i] for i in order]
        _, sc, _, _, _ = _decode_signs(cand, decode_beam_size=3)
        if sc > best_score:
            best_score = sc
            best_order = order

    improved = (best_score - base_score) >= min_improvement
    best_signs = [signs[i] for i in best_order] if improved else signs
    info = {
        "used": improved,
        "reason": "better local permutation found" if improved else "no meaningful gain",
        "score_before": float(base_score),
        "score_after": float(best_score),
        "score_gain": float(best_score - base_score),
        "tested_orders": len(orders),
        "max_combinations": max_combinations
    }
    return best_signs, info


def gardiner_sequence_to_translit(gardiner_input, try_reorder: bool = True, max_combinations: int = 20, return_debug: bool = True):
    signs = parse_gardiner_input(gardiner_input)
    translit, score, changes, segments, exact_ratio = _decode_signs(signs, decode_beam_size=6)

    chosen_signs = signs
    reorder_info = {"used": False, "reason": "not attempted", "tested_orders": 1}

    if try_reorder and 4 < len(signs) <= 48 and exact_ratio < 0.70:
        reordered, reorder_info = optimize_sign_order(
            signs,
            max_combinations=max_combinations,
            min_improvement=0.5
        )
        if reorder_info.get("used", False):
            chosen_signs = reordered
            translit, score, changes, segments, exact_ratio = _decode_signs(chosen_signs, decode_beam_size=6)

    dbg = {
        "normalized_signs": signs,
        "chosen_sign_order": chosen_signs,
        "auto_corrections": changes,
        "decode_score": float(score),
        "segments": segments,
        "exact_segment_ratio": float(exact_ratio),
        "reorder_info": reorder_info,
    }
    return (translit, dbg) if return_debug else translit


# ---------------------------------
# Stage 4 -> English
# ---------------------------------
def translate_transliteration(translit_sentence: str):
    if not translit_sentence.strip():
        return ""

    src = TASK_PREFIX + translit_sentence
    inputs = translator_tokenizer(
        src,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LEN
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        gen = translator_model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=6,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return translator_tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def grammar_cleanup(english_text: str):
    if not english_text.strip():
        return ""

    prompt = (
        "Rewrite into natural English. Remove repeated phrases. "
        "Keep original meaning. If the sentence does not make sense, "
        "make it meaningful and grammatically correct. "
        "Do not add new facts. Sentence: " + english_text.strip()
    )

    inputs = grammar_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = grammar_model.generate(
            **inputs,
            num_beams=8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.4,
            min_new_tokens=4,
            max_new_tokens=50,
            do_sample=False
        )

    return grammar_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def gardiner_to_english(gardiner_input, try_reorder=True, max_combinations=20, return_debug=False):
    translit, dbg = gardiner_sequence_to_translit(
        gardiner_input,
        try_reorder=try_reorder,
        max_combinations=max_combinations,
        return_debug=True
    )
    english = translate_transliteration(translit)

    if return_debug:
        return translit, english, dbg
    return translit, english


def model4_demo_from_split_line(split: str = "train", line_index: int = 0, max_combinations: int = 20, show_debug: bool = True):
    split = split.lower().strip()
    if split not in SPLIT_DATA:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    egy_lines = SPLIT_DATA[split]["egy"]
    eng_lines = SPLIT_DATA[split]["eng"]
    if line_index < 0 or line_index >= len(egy_lines):
        raise IndexError(f"line_index out of range for split {split}. Valid range: 0..{len(egy_lines)-1}")

    input_line = egy_lines[line_index]
    ground_truth = eng_lines[line_index]

    inputs = translator_tokenizer(
        input_line,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        gen = translator_model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    decoded_output = translator_tokenizer.decode(gen[0], skip_special_tokens=True)

    return {
        "split": split,
        "line_index": int(line_index),
        "input_line": input_line,
        "ground_truth": ground_truth,
        "predicted": decoded_output,
        "show_debug": bool(show_debug),
        "max_combinations": int(max_combinations),
    }


# ---------------------------------
# Global UI controls
# ---------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    conf_a = st.slider("Model 1 confidence", 0.0, 1.0, 0.25, 0.01)
with col2:
    conf_b = st.slider("Model 2 confidence", 0.0, 1.0, 0.40, 0.01)
with col3:
    classifier_conf = st.slider("Model 3 confidence threshold", 0.0, 1.0, 0.30, 0.01)
with col4:
    max_combinations = st.slider("Model 4 reorder combinations", 1, 50, 20, 1)

use_reorder = st.checkbox("Use Model 4 sign reordering optimization", value=True)
use_translation = st.checkbox("Run Model 4 translation stage", value=True)
use_grammar = st.checkbox("Run Model 5 grammar cleanup", value=True)
show_debug = st.checkbox("Show Model 4 debug info", value=True)

st.caption(f"Device: {DEVICE} | HF cache: {hf_cache_dir}")


tab_full, tab_m1, tab_m2, tab_m3, tab_m4, tab_m5 = st.tabs([
    "Full Pipeline",
    "Model 1 Module",
    "Model 2 Module",
    "Model 3 Module",
    "Model 4 Module",
    "Model 5 Module",
])


# ---------------------------------
# Full pipeline tab
# ---------------------------------
with tab_full:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="full_upload")

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)

        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)

        if st.button("Run Model 1", key="run_full_stage1"):
            with st.spinner("Running Model 1..."):
                result_a, t_model1 = timed_call(model_a.predict, source=image_rgb, conf=conf_a, verbose=False)
                result_a = result_a[0]

            st.session_state["full_stage1_annotated"] = draw_boxes(image_rgb, result_a.boxes, model_a, color=(0, 255, 0))
            st.session_state["full_stage1_crops"] = crop_stage1_regions(image_rgb, result_a.boxes)
            st.session_state["full_stage1_time"] = float(t_model1)

            st.session_state.pop("full_selected_region", None)
            st.session_state.pop("full_model2", None)
            st.session_state.pop("full_model3", None)
            st.session_state.pop("full_model4", None)
            st.session_state.pop("full_model5", None)

        stage1_annotated = st.session_state.get("full_stage1_annotated")
        stage1_crops = st.session_state.get("full_stage1_crops", [])
        t_model1 = st.session_state.get("full_stage1_time", 0.0)

        if stage1_annotated is not None:
            st.subheader("Model 1 Output")
            st.image(stage1_annotated, caption="Model 1 detections", use_container_width=True)
            st.info(f"Model 1 inference time: {t_model1:.3f}s")
            st.success(f"Model 1 found {len(stage1_crops)} region(s).")

            if stage1_crops:
                options = [int(c["index"]) for c in stage1_crops]
                selected_region_idx = st.selectbox("Select one Stage 1 crop for downstream pipeline", options, key="full_region_select")
                region_map = {int(c["index"]): c for c in stage1_crops}
                selected_region = region_map[selected_region_idx]
                st.session_state["full_selected_region"] = selected_region

                st.image(selected_region["image"], caption=f"Selected Model 1 crop: {selected_region_idx}", use_container_width=True)

                c_run2, c_show2 = st.columns(2)
                with c_run2:
                    if st.button("Run Model 2 on selected crop", key="run_full_stage2"):
                        with st.spinner("Running Model 2..."):
                            result_b, t_model2 = timed_call(model_b.predict, source=selected_region["image"], conf=conf_b, verbose=False)
                            result_b = result_b[0]
                            model2_annotated, symbol_crops = detect_symbols_only(selected_region["image"], result_b)

                        st.session_state["full_model2"] = {
                            "annotated": model2_annotated,
                            "symbol_crops": symbol_crops,
                            "timing": float(t_model2),
                            "region_index": selected_region_idx,
                        }
                        st.session_state.pop("full_model3", None)
                        st.session_state.pop("full_model4", None)
                        st.session_state.pop("full_model5", None)

                with c_show2:
                    show_model2 = st.checkbox("Show Model 2 output", value=False, key="show_full_stage2")

                m2 = st.session_state.get("full_model2")
                if show_model2 and m2 is not None:
                    st.subheader("Model 2 Output")
                    st.image(m2["annotated"], caption=f"Model 2 boxes on crop {m2['region_index']}", use_container_width=True)
                    st.info(f"Model 2 inference time: {m2['timing']:.3f}s")
                    st.success(f"Model 2 found symbols: {len(m2['symbol_crops'])}")

                if m2 is not None:
                    c_run3, c_show3 = st.columns(2)
                    with c_run3:
                        if st.button("Run Model 3", key="run_full_stage3"):
                            with st.spinner("Running Model 3..."):
                                classified, t_model3, attempted = classify_symbol_crops(
                                    m2["symbol_crops"],
                                    classifier_conf_threshold=classifier_conf
                                )
                            classified = sorted(classified, key=lambda x: int(x["box"][0]))
                            labels = [str(item["label"]) for item in classified]
                            st.session_state["full_model3"] = {
                                "classified": classified,
                                "labels": labels,
                                "timing": float(t_model3),
                                "attempted": int(attempted),
                            }
                            st.session_state.pop("full_model4", None)
                            st.session_state.pop("full_model5", None)

                    with c_show3:
                        show_model3 = st.checkbox("Show Model 3 output", value=False, key="show_full_stage3")

                    m3 = st.session_state.get("full_model3")
                    if show_model3 and m3 is not None:
                        st.subheader("Model 3 Output")
                        st.info(f"Model 3 total inference time: {m3['timing']:.3f}s")
                        st.success(
                            f"Model 3 classifications: kept={len(m3['classified'])}, attempted={m3['attempted']}"
                        )
                        if m3["classified"]:
                            st.write("Recognized label sequence:", " | ".join(m3["labels"]))
                            crop_cols = st.columns(4)
                            for i, item in enumerate(m3["classified"]):
                                with crop_cols[i % 4]:
                                    top3_text = ", ".join([
                                        f"{pred['label']} ({pred['confidence']:.2f})"
                                        for pred in item["top3"]
                                    ])
                                    st.image(
                                        item["image"],
                                        caption=f"{item['label']} ({item['confidence']:.2f})",
                                        use_container_width=True
                                    )
                                    st.caption(
                                        f"Top-3: {top3_text} | Model 3 time: {item['model3_inference_sec']:.4f}s"
                                    )

                    if m3 is not None and m3["labels"]:
                        c_run4, c_show4 = st.columns(2)
                        with c_run4:
                            if st.button("Run Model 4", key="run_full_stage4"):
                                with st.spinner("Running Model 4..."):
                                    (transliteration, debug_info), t_model4_resolver = timed_call(
                                        gardiner_sequence_to_translit,
                                        m3["labels"],
                                        use_reorder,
                                        max_combinations,
                                        True,
                                    )
                                    translation = ""
                                    t_model4_translate = 0.0
                                    if use_translation and transliteration:
                                        translation, t_model4_translate = timed_call(translate_transliteration, transliteration)

                                st.session_state["full_model4"] = {
                                    "transliteration": transliteration,
                                    "translation": translation,
                                    "debug": debug_info,
                                    "resolver_time": float(t_model4_resolver),
                                    "translate_time": float(t_model4_translate),
                                }
                                st.session_state.pop("full_model5", None)

                        with c_show4:
                            show_model4 = st.checkbox("Show Model 4 output", value=False, key="show_full_stage4")

                        m4 = st.session_state.get("full_model4")
                        if show_model4 and m4 is not None:
                            st.subheader("Model 4 Output")
                            st.info(
                                f"Model 4 timing: resolver={m4['resolver_time']:.3f}s, "
                                f"translator={m4['translate_time']:.3f}s"
                            )
                            st.write(f"Resolved transliteration: {m4['transliteration'] if m4['transliteration'] else '(empty)'}")
                            if m4["translation"]:
                                st.write(f"Raw translation: {m4['translation']}")

                            if show_debug and m4["debug"]:
                                with st.expander("Model 4 debug", expanded=False):
                                    st.write("Normalized signs:", m4["debug"].get("normalized_signs", []))
                                    st.write("Chosen sign order:", m4["debug"].get("chosen_sign_order", []))
                                    st.write("Auto-corrections:", m4["debug"].get("auto_corrections", []))
                                    st.write("Exact segment ratio:", m4["debug"].get("exact_segment_ratio", 0.0))
                                    st.write("Decode score:", m4["debug"].get("decode_score", 0.0))
                                    st.json(to_python_types(m4["debug"].get("reorder_info", {})))
                                    st.json(to_python_types(m4["debug"].get("segments", [])))

                        if m4 is not None and m4["translation"] and use_grammar:
                            c_run5, c_show5 = st.columns(2)
                            with c_run5:
                                if st.button("Run Model 5", key="run_full_stage5"):
                                    with st.spinner("Running Model 5..."):
                                        cleaned_translation, t_model5 = timed_call(grammar_cleanup, m4["translation"])
                                    st.session_state["full_model5"] = {
                                        "cleaned": cleaned_translation,
                                        "timing": float(t_model5),
                                    }

                            with c_show5:
                                show_model5 = st.checkbox("Show Model 5 output", value=False, key="show_full_stage5")

                            m5 = st.session_state.get("full_model5")
                            if show_model5 and m5 is not None:
                                st.subheader("Model 5 Output")
                                st.info(f"Model 5 inference time: {m5['timing']:.3f}s")
                                st.success(f"Final English: {m5['cleaned']}")


# ---------------------------------
# Model 1 module tab
# ---------------------------------
with tab_m1:
    st.subheader("Model 1: Region Detection Test")
    file_m1 = st.file_uploader("Upload image for Model 1", type=["jpg", "jpeg", "png"], key="m1_upload")
    if file_m1 is not None:
        img = np.array(Image.open(file_m1).convert("RGB"))
        st.image(img, caption="Input", use_container_width=True)
        if st.button("Run Model 1", key="run_m1"):
            pred, t_m1 = timed_call(model_a.predict, source=img, conf=conf_a, verbose=False)
            pred = pred[0]
            out = draw_boxes(img, pred.boxes, model_a, color=(0, 255, 0))
            st.image(out, caption="Model 1 output", use_container_width=True)
            count = 0 if pred.boxes is None else len(pred.boxes)
            st.success(f"Detections: {count}")
            st.info(f"Model 1 inference time: {t_m1:.3f}s")


# ---------------------------------
# Model 2 module tab
# ---------------------------------
with tab_m2:
    st.subheader("Model 2: Glyph Detection Test")
    file_m2 = st.file_uploader("Upload line image for Model 2", type=["jpg", "jpeg", "png"], key="m2_upload")
    if file_m2 is not None:
        img = np.array(Image.open(file_m2).convert("RGB"))
        st.image(img, caption="Input", use_container_width=True)
        if st.button("Run Model 2", key="run_m2"):
            pred, t_m2 = timed_call(model_b.predict, source=img, conf=conf_b, verbose=False)
            pred = pred[0]
            out = draw_boxes(img, pred.boxes, model_b, color=(255, 0, 0))
            st.image(out, caption="Model 2 output", use_container_width=True)
            count = 0 if pred.boxes is None else len(pred.boxes)
            st.success(f"Detected glyph boxes: {count}")
            st.info(f"Model 2 inference time: {t_m2:.3f}s")


# ---------------------------------
# Model 3 module tab
# ---------------------------------
with tab_m3:
    st.subheader("Model 3: Glyph Classification Test")
    file_m3 = st.file_uploader("Upload single glyph image for Model 3", type=["jpg", "jpeg", "png"], key="m3_upload")
    if file_m3 is not None:
        img = np.array(Image.open(file_m3).convert("RGB"))
        st.image(img, caption="Input", width=220)
        if st.button("Run Model 3", key="run_m3"):
            preds, t_m3 = timed_call(classify_symbol, img, 5)
            st.info(f"Model 3 inference time: {t_m3:.4f}s")
            for i, p in enumerate(preds, 1):
                st.write(f"{i}. {p['label']} (idx={p['index']}), confidence={p['confidence']:.4f}")


# ---------------------------------
# Model 4 module tab
# ---------------------------------
with tab_m4:
    st.subheader("Model 4: Gardiner -> Transliteration -> English")
    m4_mode = st.radio(
        "Model 4 mode",
        ["Custom Gardiner Sequence", "Split Line Demo (train/val/test)"],
        index=0,
        key="m4_mode",
    )

    if m4_mode == "Custom Gardiner Sequence":
        gardiner_input = st.text_area(
            "Enter Gardiner sequence (space-separated, e.g. A1 D36 N35)",
            "",
            key="m4_input"
        )
        run_m4_translation = st.checkbox("Also run Model 4 translation", value=True, key="m4_translate")
        if st.button("Run Model 4", key="run_m4"):
            signs = parse_gardiner_input(gardiner_input)
            if not signs:
                st.warning("Please provide a valid Gardiner sequence.")
            else:
                (translit, dbg), t_resolve = timed_call(
                    gardiner_sequence_to_translit,
                    signs,
                    use_reorder,
                    max_combinations,
                    True,
                )
                st.write("Parsed signs:", signs)
                st.success(f"Resolved transliteration: {translit if translit else '(empty)'}")
                st.info(f"Model 4 resolver time: {t_resolve:.3f}s")

                t_translate = 0.0
                translation = ""
                if run_m4_translation and translit:
                    translation, t_translate = timed_call(translate_transliteration, translit)
                    st.write(f"Raw translation: {translation}")
                    st.info(f"Model 4 translation time: {t_translate:.3f}s")

                if show_debug:
                    with st.expander("Model 4 debug", expanded=False):
                        st.write("Normalized signs:", dbg.get("normalized_signs", []))
                        st.write("Chosen sign order:", dbg.get("chosen_sign_order", []))
                        st.write("Auto-corrections:", dbg.get("auto_corrections", []))
                        st.write("Exact segment ratio:", dbg.get("exact_segment_ratio", 0.0))
                        st.write("Decode score:", dbg.get("decode_score", 0.0))
                        st.json(to_python_types(dbg.get("reorder_info", {})))
                        st.json(to_python_types(dbg.get("segments", [])))
    else:
        split = st.selectbox("Select split", ["train", "val", "test"], index=0, key="m4_split")
        egy_lines = SPLIT_DATA[split]["egy"]
        eng_lines = SPLIT_DATA[split]["eng"]
        max_idx = max(0, len(egy_lines) - 1)
        line_index = st.number_input(
            f"Line index for {split} (0 to {max_idx})",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1,
            key="m4_line_idx",
        )

        if st.button("Run Model 4 Split-Line Demo", key="run_m4_split_demo"):
            idx = int(line_index)
            demo_result, t_demo = timed_call(
                model4_demo_from_split_line,
                split,
                idx,
                max_combinations,
                show_debug,
            )

            st.write(f"Split: {demo_result['split']} | line[{demo_result['line_index']}]")
            st.write("Input line (Egyptian transliteration):", demo_result["input_line"])
            st.write("Ground truth English:", demo_result["ground_truth"])
            st.write("Predicted English:", demo_result["predicted"])
            st.info(f"Model 4 split-line total time: {t_demo:.3f}s")


# ---------------------------------
# Model 5 module tab
# ---------------------------------
with tab_m5:
    st.subheader("Model 5: Grammar Cleanup")
    m5_text = st.text_area("Enter English text for grammar cleanup", "", key="m5_text")
    if st.button("Run Model 5", key="run_m5"):
        if not m5_text.strip():
            st.warning("Please enter text.")
        else:
            cleaned, t_m5 = timed_call(grammar_cleanup, m5_text)
            st.write(f"Cleaned text: {cleaned}")
            st.info(f"Model 5 inference time: {t_m5:.3f}s")
