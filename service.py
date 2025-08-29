# service.py
import os, io, json, faiss, numpy as np, requests
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import torch, open_clip
from sentence_transformers import SentenceTransformer

# -------------------- Config --------------------
AUTH = os.getenv("AUTH_TOKEN", "dev-secret")
INDEX_DIR = "./index"
os.makedirs(INDEX_DIR, exist_ok=True)
IMG_INDEX_PATH = f"{INDEX_DIR}/img.index"
TXT_INDEX_PATH = f"{INDEX_DIR}/txt.index"
META_PATH      = f"{INDEX_DIR}/meta.json"

# thresholds
MIN_IMG_SIM = float(os.getenv("MIN_IMG_SIM", "0.35"))
MIN_COLOR_PROP = float(os.getenv("MIN_COLOR_PROP", "0.10"))
SPECIES_CLASSES = ["cat", "dog"]
SPECIES_MIN_MARGIN = 0.05

# -------------------- Encoders --------------------
class ImageEnc:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.eval().to(self.device)

        # Zero-shot species detection
        prompts = [f"a photo of a {c}" for c in SPECIES_CLASSES]
        with torch.inference_mode():
            tok = open_clip.tokenize(prompts).to(self.device)
            t = self.model.encode_text(tok)
            t = t / t.norm(dim=-1, keepdim=True)
        self.spec_text = t
        self.spec_labels = SPECIES_CLASSES

    @torch.inference_mode()
    def encode_pil(self, img: Image.Image) -> np.ndarray:
        x = self.pre(img).unsqueeze(0).to(self.device)
        v = self.model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
        return v.cpu().numpy().astype("float32")

    @torch.inference_mode()
    def detect_species_from_vec(self, v_img: np.ndarray) -> str:
        vt = torch.from_numpy(v_img).to(self.device)
        sims = (vt @ self.spec_text.T).squeeze(0)
        top2 = torch.topk(sims, k=min(2, sims.numel()))
        top1_idx = int(top2.indices[0].item())
        top1_val = float(top2.values[0].item())
        margin = top1_val - float(top2.values[1].item()) if top2.values.numel() > 1 else top1_val
        return self.spec_labels[top1_idx] if margin >= SPECIES_MIN_MARGIN else "unknown"

    def encode_url_or_path(self, url: str, local_path: Optional[str]) -> Tuple[np.ndarray, Image.Image]:
        img = fetch_image(url, local_path)
        vec = self.encode_pil(img)
        return vec, img

class TextEnc:
    def __init__(self):
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode(self, text: str) -> np.ndarray:
        v = self.m.encode([text], normalize_embeddings=True)[0]
        return v.astype("float32")

IMG = ImageEnc()
TXT = TextEnc()

# -------------------- Helpers --------------------
def ensure_auth(x_token: Optional[str]):
    if x_token != AUTH:
        raise HTTPException(401, detail={"ok": False, "code": "UNAUTHORIZED", "message": "Missing or invalid X-Token."})

def norm_species(val: Optional[str]) -> Optional[str]:
    return val.lower().strip() if isinstance(val, str) and val.strip() else None

def norm_colors(vals: Optional[List[str]]) -> List[str]:
    if not vals: return []
    return sorted({str(v).lower().strip() for v in vals if str(v).strip()})

def safe_iso(dt_str: Optional[str]) -> Optional[str]:
    if not dt_str: return None
    try: return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).isoformat()
    except Exception: return None

def attrs_text(d: dict) -> str:
    return " | ".join([
        f"species:{d.get('species','')}",
        f"auto_species:{d.get('auto_species','')}",
        f"breed:{d.get('breed','')}",
        f"colors:{', '.join(d.get('colors') or [])}",
        f"auto_colors:{', '.join(d.get('auto_colors') or [])}",
        f"sex:{d.get('sex','')}",
        f"collar:{'yes' if d.get('collar') else 'no'}",
        f"markings:{d.get('markings','')}",
        f"notes:{d.get('notes','')}",
    ])

def fetch_image(url: str, local_path: Optional[str] = None) -> Image.Image:
    if local_path:
        p = Path(local_path)
        if p.exists(): return Image.open(p).convert("RGB")
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(422, detail={"ok": False, "code": "IMAGE_FETCH_FAILED", "message": str(e)})

# simple dominant colors
def detect_image_colors(img: Image.Image, min_prop: float = MIN_COLOR_PROP) -> List[str]:
    small = img.resize((100,100))
    arr = np.asarray(small).astype(np.float32)/255.0
    hsv = rgb_to_hsv(arr)
    H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]
    total = float(H.size)
    tags = []
    def add(mask,name):
        prop = mask.sum()/total
        if prop>=min_prop: tags.append((prop,name))
    add((V>0.9)&(S<0.2),"white")
    add((V<0.15),"black")
    add((S<0.15)&(V>=0.15)&(V<=0.9),"gray")
    add((H>0.05)&(H<0.15)&(S>0.4),"brown")
    add((H>0.1)&(H<0.2)&(S>0.3),"orange")
    tags.sort(reverse=True,key=lambda x:x[0])
    return [n for _,n in tags[:3]]

def rgb_to_hsv(arr: np.ndarray)->np.ndarray:
    r,g,b = arr[...,0],arr[...,1],arr[...,2]
    mx,mn = arr.max(-1),arr.min(-1)
    diff=mx-mn+1e-8
    h=np.zeros_like(mx)
    h[(mx==r)] = ((g-b)/diff)[(mx==r)]
    h[(mx==g)] = ((b-r)/diff)[(mx==g)] + 2
    h[(mx==b)] = ((r-g)/diff)[(mx==b)] + 4
    h=(h/6.0)%1.0
    s=diff/(mx+1e-8)
    v=mx
    return np.stack([h,s,v],-1)

def load_meta():
    if not os.path.exists(META_PATH): return []
    try: return json.load(open(META_PATH,"r",encoding="utf-8"))
    except: return []

def save_meta(m): json.dump(m,open(META_PATH,"w",encoding="utf-8"),ensure_ascii=False)

IMG_IDX=TXT_IDX=None
def try_load_indexes():
    i=t=None
    if os.path.exists(IMG_INDEX_PATH): i=faiss.read_index(IMG_INDEX_PATH)
    if os.path.exists(TXT_INDEX_PATH): t=faiss.read_index(TXT_INDEX_PATH)
    return i,t
def save_indexes():
    if IMG_IDX: faiss.write_index(IMG_IDX,IMG_INDEX_PATH)
    if TXT_IDX: faiss.write_index(TXT_IDX,TXT_INDEX_PATH)

def ensure_index_dims(img_dim,txt_dim):
    global IMG_IDX,TXT_IDX
    if IMG_IDX is None or TXT_IDX is None:
        IMG_IDX,TXT_IDX=try_load_indexes()
    if IMG_IDX is None: IMG_IDX=faiss.IndexFlatIP(img_dim)
    if TXT_IDX is None: TXT_IDX=faiss.IndexFlatIP(txt_dim)

# -------------------- Schemas --------------------
class FoundItem(BaseModel):
    id:str
    image_url:str
    local_path:Optional[str]=None
    species:Optional[str]=None
    breed:Optional[str]=None
    colors:Optional[List[str]]=[]
    sex:Optional[str]=None
    collar:Optional[bool]=None
    markings:Optional[str]=None
    notes:Optional[str]=None
    lat:Optional[float]=None
    lng:Optional[float]=None
    timestamp:Optional[str]=None

class Query(BaseModel):
    image_url:Optional[str]=None
    local_path:Optional[str]=None
    species:Optional[str]=None
    breed:Optional[str]=None
    colors:Optional[List[str]]=[]
    sex:Optional[str]=None
    collar:Optional[bool]=None
    markings:Optional[str]=None
    notes:Optional[str]=None
    lost_lat:Optional[float]=None
    lost_lng:Optional[float]=None
    lost_time:Optional[str]=None
    search_radius_km:Optional[float]=None
    k_img:int=200
    k_txt:int=100
    topk:int=20

# -------------------- FastAPI --------------------
app=FastAPI(title="PetScout Search")
@app.get("/")    
def root(): return {"ok":True}
@app.get("/health")
def health(): return {"ok":True}

# -------------------- Index --------------------
@app.post("/index/upsert")
def upsert(item:FoundItem,x_token:Optional[str]=Header(None)):
    ensure_auth(x_token)
    vec,img=IMG.encode_url_or_path(item.image_url,item.local_path)
    auto_species=IMG.detect_species_from_vec(vec)
    auto_colors=detect_image_colors(img)
    v_txt=TXT.encode(attrs_text({
        **item.dict(),
        "auto_species":auto_species,
        "auto_colors":auto_colors
    }))
    ensure_index_dims(vec.shape[1],v_txt.shape[0])
    IMG_IDX.add(vec)
    TXT_IDX.add(v_txt.reshape(1,-1))
    metas=load_meta()
    m=item.dict()
    m["species"]=norm_species(m.get("species"))
    m["colors"]=norm_colors(m.get("colors"))
    m["auto_species"]=auto_species
    m["auto_colors"]=auto_colors
    metas.append(m)
    save_indexes();save_meta(metas)
    return {"ok":True,"size":len(metas)}

# -------------------- Search --------------------
@app.post("/search")
def search(q:Query,x_token:Optional[str]=Header(None)):
    ensure_auth(x_token)
    global IMG_IDX,TXT_IDX
    if IMG_IDX is None or TXT_IDX is None: IMG_IDX,TXT_IDX=try_load_indexes()
    metas=load_meta()
    if not metas: return {"results":[]}
    img_map={}
    query_species=None;query_colors=[]
    if q.image_url or q.local_path:
        vec,img=IMG.encode_url_or_path(q.image_url or "",q.local_path)
        query_species=IMG.detect_species_from_vec(vec)
        query_colors=detect_image_colors(img)
        sims,ids=IMG_IDX.search(vec,q.k_img)
        img_map={int(i):float(s) for i,s in zip(ids[0],sims[0]) if i!=-1}
    q_dict=q.dict();q_dict["auto_species"]=query_species;q_dict["auto_colors"]=query_colors
    tv=TXT.encode(attrs_text(q_dict)).reshape(1,-1)
    sims,ids=TXT_IDX.search(tv,q.k_txt)
    txt_map={int(i):float(s) for i,s in zip(ids[0],sims[0]) if i!=-1}
    ranked=[]
    for cid in set(img_map)|set(txt_map):
        if cid>=len(metas): continue
        m=metas[cid]
        cand_species=m.get("auto_species") or m.get("species")
        if query_species and cand_species and query_species!=cand_species: continue
        if (q.image_url or q.local_path) and img_map.get(cid,0.0)<MIN_IMG_SIM: continue
        if query_colors and m.get("auto_colors") and len(set(query_colors)&set(m["auto_colors"]))==0: continue
        score=0.9*img_map.get(cid,0.0)+0.1*txt_map.get(cid,0.0)
        ranked.append((score,cid,img_map.get(cid,0.0)))
    ranked.sort(reverse=True)
    out=[{
        "score":round(s,3),
        "id":metas[c]["id"],
        "image_url":metas[c]["image_url"],
        "species":metas[c].get("species"),
        "auto_species":metas[c].get("auto_species"),
        "auto_colors":metas[c].get("auto_colors"),
        "img_sim":round(isim,3)
    } for s,c,isim in ranked[:q.topk]]
    return {"results":out}
