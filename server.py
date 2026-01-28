from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from dotenv import load_dotenv
from text_processing import (
    TextProcessor,
    DEFAULT_AUTHORS,
    current_chart_names,
    chart_names_source,
    set_chart_names,
    MAX_CHART_NAME_COUNT,
    MAX_CHART_NAME_LEN,
    build_system_prompt,
)
from model_manager import ModelManager

app = FastAPI(title="RhythmGame Translation (Prompt-only, TagBank+Rules+Examples)")

# ===== 환경 변수 로드 (.env 지원) =====
load_dotenv()

model_manager = ModelManager()
text_processor = TextProcessor()

LANGS = {"ko", "en", "es"}

# ===== 시스템 프롬프트 빌더 =====
# ===== 위에서 제공한 번역 예시(다수 few-shot) =====
DEFAULT_EXAMPLES = [
    # 쌍 1
    {
      "source": "ko", "target": "en",
      "input": "더블16을 조금 강화한 채보. 기믹이 읽기 어려운 편도 아니며, 겹발이 그렇게 까다롭지도 않음. 이게 판도라 더블20이랑 같은 난이도로 해당된다는게 좀; 체감 난이도는 18.",
      "output": "A slightly buffed rendition of D16. Gimmicks aren't awful to read, and brackets aren't tricky to execute. Should not be considered the same level as Pandora D20. Feels like an 18."
    },
    {
      "source":"en","target":"ko",
      "input":"A slightly buffed rendition of D16. Gimmicks aren't awful to read, and brackets aren't tricky to execute. Should not be considered the same level as Pandora D20. Feels like an 18.",
      "output":"더블16을 조금 강화한 채보. 기믹이 읽기 어려운 편도 아니며, 겹발이 그렇게 까다롭지도 않음. 판도라 D20과 같은 레벨로 보기는 어렵고, 체감 난이도는 18."
    },

    # 쌍 2
    {"source":"ko","target":"en","input":"폭타가 꾸준히 나오는 체력곡.","output":"A stamina chart with lots of runs"},
    {"source":"en","target":"ko","input":"A stamina chart with lots of runs","output":"폭타가 꾸준히 나오는 체력곡."},

    # 쌍 3
    {"source":"ko","target":"en","input":"유튜브로 한 10번 돌려보면 다 외워짐. 피지컬로 커버 안 될 부분은 없는듯. 20보다 쉬움.","output":"Watch the chart on YouTube 10 times and you'll memorize the whole chart. There are no physically demanding parts. Easier than the S20."},
    {"source":"en","target":"ko","input":"Watch the chart on YouTube 10 times and you'll memorize the whole chart. There are no physically demanding parts. Easier than the S20.","output":"유튜브로 한 10번 돌려보면 다 외워짐. 피지컬로 커버 안 될 부분은 없는듯. S20보다 쉬움."},

    # 쌍 4
    {"source":"ko","target":"en","input":"싱글20보다 훨씬 더 넓은 보폭이 필요하기 때문에 어려울 수 있음. 또한, 중간에 있는 빨빨 파파 겹발들이 많이 나오니 익숙치 않으면 쉽게 놓칠수 있음. 대신 포션들이 많이 나옴.","output":"Requires much further reach than the S20. There are many middle P1/P2 brackets (red-red and blue-blue) that may be difficult for the inexperienced."},

    # 쌍 5
    {"source":"ko","target":"en","input":"갓라드의 꿀잼채보. 느린 브픔에 무난한 틀기. 후렴에 몇개 있는 겹발 홀드만 잘 견디면 됨.","output":"Very fun chart by CONRAD. Low BPM with easy twists. Careful on the bracket holds."},

    # 쌍 6
    {"source":"ko","target":"en","input":"겹발 롱노트만 아니었으면 17로 내려가야할 채보. 발판 상태만 괜찮으면 첫트클은 아닐지라도 무조건 깸.","output":"If it weren't for the bracket holds this chart would definitely be a D17. As long as the pads are functional and sensitive enough, this chart should be free."},

    # 쌍 7
    {"source":"ko","target":"en","input":"피지컬로는 커버가 다 가능한 채보라 외우는 능력에 모든게 달려있음. 싱글19을 펼친 채보라 박자는 똑같지만 패턴이 훨씬 까다롭게 나옴. 저속기믹이 끝나기 전에 있는 대계단 주의.","output":"Physically very feasible, but memorization is of paramount importance. It is charted based on the S19, but the patterns are much more complex due to it being a doubles chart. Watch out for the stairs at the end of the gimmick section."},

    # 쌍 8
    {"source":"ko","target":"en","input":"폭타가 거의 없는 하프 틀기 채보. 뭉갤수 있는 구간이 적지 않으니까 클리어 기준으로는 나쁘지 않음. 연습삼아 올틀기로 해보는걸 추천.","output":"A half twist chart with barely any runs. A good amount of parts are mashable, making the chart more forgiving to clear. Would recommend trying to twist all the patterns as intended for practice."},

    # 쌍 9
    {"source":"ko","target":"en","input":"그냥 발이 빨라야 함. 현지인들 힘내세요.","output":"You need speed. Good luck."},

    # 쌍 10
    {"source":"ko","target":"en","input":"귀여운 브가에 그렇지 않은 써니특 틀기 패턴들이 나옴. 특히 마지막 틀기폭타는 잘못 읽으면 위험함. 느린 브픔이라 그래도 도전해볼만한 채보.","output":"Slow but hard twist patterns. Last run is hard to read."},

    # 쌍 11
    {"source":"ko","target":"en","input":"그저 갓채보... 마지막 패턴은 체중이동이 심하므로 집중하는게 좋음.","output":"Super fun twist run patterns. The final run has tons of weight shifting (side-to-side movement) so beware."},

    # 쌍 12
    {"source":"ko","target":"en","input":"솔직히 20중상?정도 되는 물렙. 아주 짧게 나오는 고속폭타만 잘 밟거나 뭉게주면 됨.","output":"Should be downgraded to a D20. The only hard parts are the short high BPM runs."},

    # 쌍 13
    {"source":"ko","target":"en","input":"낮지 않은 브픔에 꽤나 어려운 틀기 폭타 패턴이 나옴. 이거 외에는 무난한 채보라고 생각함.","output":"Pretty tricky run pattern for this BPM, but trivial otherwise."},

    # 쌍 14
    {"source":"ko","target":"en","input":"느린 브픔에 꽤나 어려운 고각틀기. 쉴구간이 상당히 많아서 클리어 하기는 쉬운편.","output":"Slow but hard twist patterns. Lots of pauses to save stamina."},

    # 쌍 15
    {"source":"ko","target":"en","input":"구곡이라서 첫트클 하기에는 어려울수 있지만, 영상을 몇번 돌려보면 충분히 읽기 가능한 채보. 고각틀기에 체력소모가 심할수 있으므로 주의. 브픔이 낮으니까 최대한 가볍고 느리게 움직이는걸 추천.","output":"Hard to sightread due to the chart being old, but fairly readable if you study it a few times. Careful not to drain your stamina on all the high-angle twists. Take advantage of the low BPM by moving lightly and slowly."},

    # 쌍 16
    {"source":"ko","target":"en","input":"폭타가 많지만 짧게 나와서 생각보다 할만한 채보. 마지막 패턴은 읽기 어려울수 있으므로 미리 영상으로 보는걸 추천. 프탐20보다는 쉬움.","output":"Lots of runs but they are short. Around the same difficulty as District V but easier than Prime Time."},

    # 쌍 17
    {"source":"ko","target":"en","input":"브픔은 높지만 폭타가 길진 않음. 마지막까지 체력만 잘 안배하면 충분히 클리어 가능한 채보.","output":"High BPM but relatively short runs; should be easy to clear if you conserve your stamina well."},

    # 쌍 18
    {"source":"ko","target":"en","input":"220이라는 매우 빠른 브픔에 꽤나 많이 나오는 트릴. 사뿐히 밟으면서 체력을 잘 안배하는것이 중요함.","output":"Very fast BPM but lots of drills. Keep your steps on the lighter side and conserve your stamina."},

    # 쌍 19
    {"source":"ko","target":"en","input":"짧게 나오는 떨기에다가 떨기처럼 보이는 겹발 롱놋들. 겹발만 안 새면 무난히 클리어 할 수 있음. 첫클로 깼음.","output":"Short drills and bracket holds disguised as drills. If you're careful on the bracket holds it should be free."},

    # 쌍 20
    {"source":"ko","target":"en","input":"아주 짧게 나오는 트릴과 롱노트만 잘 밟아주면 무난한 채보.","output":"Pretty straightforward chart as long as you nail the mini drills and holds."},

    # 쌍 21
    {"source":"ko","target":"en","input":"낮은 브픔에 끊임없이 나오는 틀기채보. 틀기가 강점이라면 추천.","output":"Low BPM twisty chart."},

    # 쌍 22 — idiom: basically nothing outside of X
    {"source":"en","target":"ko",
     "input":"This chart is basically nothing outside of the runs.",
     "output":"이 채보는 폭타 외에는 별게 없습니다."},

    # 쌍 23 — idiom variant with weight shift
    {"source":"en","target":"ko",
     "input":"It's basically nothing outside of weight shift sections.",
     "output":"체중이동 구간 외에는 별게 없는 편입니다."},

    # 쌍 24 — KO→EN: bare numeral must stay bare, no S/D inference
    {"source":"ko","target":"en",
     "input":"최근 20 추세를 보면 평균 20 수준인 듯.",
     "output":"Looking at the recent 20 trend, it feels around the average 20 level."},

    # 쌍 25 — KO→EN: keep 'S20' when explicit, but keep bare '20' bare
    {"source":"ko","target":"en",
     "input":"최후반 복계단은 S20 입문자에겐 좌절 포인트. 최근 20 추세는 더 어려워지는 중.",
     "output":"The final back-and-forth stairs can be a breaking point for S20 beginners. The recent 20 trend is getting tougher."},

    # 쌍 26 — en→ko: Full Moon reference; easier than an already easy 24
    {"source":"en","target":"ko",
     "input":"Just an easier Full Moon which is already a free 24.",
     "output":"이미 쉬운 풀 문 24보다 더 쉽다."},
    
    # 쌍 27 — KO→EN mid + 18 preserved
    {"source":"ko","target":"en",
     "input":"이게 중일리가 없습니다 다 틀어보면 이게 18인게 이해가 안갈정도입니다 틀기만으로 버거운데 틀폭까지 넣다니 말이 안됩니다",
     "output":"No way this is mid. Even if you twist everything, it's hard to believe it's an 18. The twists alone are tough; adding twist runs makes no sense."},

    # 쌍 28 — KO→EN: 'N치고' phrasing in-line (no trailing 'for a N.')
    {"source": "ko", "target": "en",
     "input": "16치고는 틀폭이 어렵긴 한데 빠른 곡이 아니어서 조금만 연습하면 할 수 있습니다.",
     "output": "The twist runs are tough for a 16, but since it isn't a fast chart, a bit of practice makes it manageable."},

    # 쌍 29
    {"source": "ko", "target": "en",
     "input": "어드2 초견 S인데요?",
     "output": "Got an S on Advanced 2 on first try."},

    # 쌍 30
    {"source": "ko", "target": "en",
     "input": "개구리 하면 어드작 날먹곡. 그냥도 할만한듯.",
     "output": "Using frog stance makes it basically a freebie chart. Though it's still doable even without it."},

    # 쌍 31 — KO→EN: 쉬게해줘 → let me rest / give me a breather
    {"source": "ko", "target": "en",
     "input": "좀 쉬게해줘 제발..",
     "output": "Please, let me rest a bit."},

    # 쌍 32 — KO→EN: 폭타 + 중하 보존 예시
    {"source": "ko", "target": "en",
     "input": "피펨즈 채보치고는 무난함, 할거없는데 한번 시원하게 폭타치고싶으면 추천 / 다만 체력이 좀 들어 서 개인적 견해는 중하",
     "output": "Pretty manageable for a FEFEMZ chart. If you just want to blast some runs, recommended. It does take some stamina, so I'd rate it lower mid."},

    # 쌍 33 — KO→EN: '23최상 라인' → '23 top line'
    {"source": "ko", "target": "en",
     "input": "23최상 라인보단 쉬운 곡",
     "output": "Easier than the 23 top line."},

    # 쌍 34 — KO→EN: 판정 → timing/judgment nuance
    {"source": "ko", "target": "en",
     "input": "폭타 할만한데 판정이 다 나가네요...",
     "output": "The runs are doable, but the timing feels completely off..."},

    # 쌍 35 — KO→EN: 현지인/초견/체력
    {"source": "ko", "target": "en",
     "input": "현지인이라면 초견에 깰 수 있음. 체력 얼마 안들어요",
     "output": "Locals can clear it on sightread. It doesn't require much stamina."},
]

# ---- 메시지 빌더 ----
def build_messages(src: str, tgt: str, text: str,
                   examples: List[Dict[str, str]],
                   authors: List[str],
                   chart_names: List[str],
                   placeholder_tokens: Optional[List[str]] = None) -> list:
    SELECT_SENTENCE = 40
    sys = build_system_prompt(tgt, authors, chart_names)
    msgs = [{"role": "system", "content": sys}]

    # few-shot 예시(최대 SELECT_SENTENCE개). 방향/우선순위 기반으로 결정적으로 선택.
    # 1) 현재 방향(src->tgt) 예시를 우선 필터
    directional = [e for e in examples if e.get("source", src) == src and e.get("target", tgt) == tgt]

    # 2) 우선 키워드(필요 시 확대 가능)
    priority_terms = []
    if src == "en" and tgt == "ko":
        # Only steer for known tricky idioms that aren't covered well by tags
        priority_terms = ["basically nothing", "nothing outside of", "nothing besides", "nothing but", "free ", "full moon"]
    elif src == "ko" and tgt == "en":
        # Rely on glossary + placeholders for KO terms (폭타, 중하, 초견, 어드 등)
        priority_terms = []

    def is_priority(ex):
        s = (ex.get("input") or "") + "\n" + (ex.get("output") or "")
        s_low = s.lower()
        return any(term in s_low for term in priority_terms)

    prioritized = [e for e in directional if is_priority(e)]
    non_priority = [e for e in directional if not is_priority(e)]

    chosen: List[Dict[str, str]] = []
    chosen.extend(prioritized)
    for e in non_priority:
        if len(chosen) >= SELECT_SENTENCE: break
        chosen.append(e)
    if len(chosen) < SELECT_SENTENCE:
        # 3) 부족하면 남은 예시(방향 무관)에서 채움(중복 제외)
        for e in examples:
            if e in chosen: continue
            if len(chosen) >= SELECT_SENTENCE: break
            chosen.append(e)

    for ex in chosen[:SELECT_SENTENCE]:
        ex_src = ex.get("source", src)
        ex_tgt = ex.get("target", tgt)
        ex_in  = ex["input"]
        ex_out = ex["output"]
        msgs.append({"role":"user","content":f"Translate from {ex_src} to {ex_tgt}:\n{ex_in}"})
        msgs.append({"role":"assistant","content":ex_out})

    if placeholder_tokens:
        tokens_str = ", ".join(placeholder_tokens[:50])
        notice = (
            "PLACEHOLDER NOTICE:\n"
            f"The source text includes placeholder tokens: {tokens_str}.\n"
            "Copy each placeholder EXACTLY once; do not edit or drop them.\n"
            "DO NOT translate this notice; it is meta-instruction only."
        )
        if len(placeholder_tokens) > 50:
            notice += " (list truncated)"
        msgs.append({"role": "system", "content": notice})

    user_content = f"Translate from {src} to {tgt}:\n{text}"
    msgs.append({"role": "user", "content": user_content})
    return msgs


# ---- API ----
class ExampleItem(BaseModel):
    source: Optional[str] = None
    target: Optional[str] = None
    input: str
    output: str

class TranslateReq(BaseModel):
    text: str
    source: str  # "ko"|"en"|"es"
    target: str  # "ko"|"en"|"es"
    use_default_examples: Optional[bool] = True
    examples: Optional[List[ExampleItem]] = None
    authors: Optional[List[str]] = None
    chart_names: Optional[List[str]] = None

class ChartNamesUpdateReq(BaseModel):
    chart_names: Optional[List[str]] = None
    reset: Optional[bool] = False

@app.post("/translate")
def translate(req: TranslateReq):
    assert req.source in LANGS and req.target in LANGS, "unsupported language"
    ex = []
    if req.use_default_examples:
        ex.extend(DEFAULT_EXAMPLES)
    if req.examples:
        ex.extend([e.model_dump() for e in req.examples])

    authors = req.authors or DEFAULT_AUTHORS
    chart_names = req.chart_names or current_chart_names()

    if model_manager.backend == "nllb":
        out = model_manager.translate_nllb(req.text, req.source, req.target, text_processor)
    else:
        # LLM path: apply placeholders to preserve glossary/protected (leave digits inline)
        t_in, ph, placeholder_tokens = text_processor.apply_placeholders(
            req.text, req.source, req.target, preserve_digits=False
        )
        t_clean = text_processor.sanitize_prompt(t_in)
        messages = build_messages(
            req.source,
            req.target,
            t_clean,
            ex,
            authors,
            chart_names,
            placeholder_tokens=placeholder_tokens,
        )
        out = model_manager.generate(messages)
        out = text_processor.ensure_placeholder_tokens(out, placeholder_tokens)
        out = text_processor.restore_placeholders(out, ph)
        out = text_processor.postprocess_output(out, req.target, req.text)
    out = text_processor.enforce_tag_glossary(out, req.source, req.target)
    return {"source": req.source, "target": req.target, "text": req.text, "translated": out}

@app.get("/chart_names")
def get_chart_names():
    return {"chart_names": current_chart_names(), "source": chart_names_source()}

@app.post("/chart_names")
def set_chart_names_endpoint(req: ChartNamesUpdateReq):
    if req.reset:
        names = set_chart_names(None)
        return {"chart_names": names, "source": chart_names_source()}
    if not req.chart_names:
        raise HTTPException(status_code=400, detail="chart_names must be provided unless reset=true.")
    cleaned: List[str] = []
    for item in req.chart_names:
        if not item:
            continue
        value = item.strip()
        if not value:
            continue
        if len(value) > MAX_CHART_NAME_LEN:
            preview = value[:32]
            if len(value) > 32:
                preview += "..."
            raise HTTPException(
                status_code=400,
                detail=f"Chart name '{preview}' exceeds {MAX_CHART_NAME_LEN} characters.",
            )
        cleaned.append(value)
    if not cleaned:
        raise HTTPException(status_code=400, detail="chart_names must include at least one non-empty entry.")
    if len(cleaned) > MAX_CHART_NAME_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_CHART_NAME_COUNT} chart names supported.",
        )
    names = set_chart_names(cleaned)
    return {"chart_names": names, "source": chart_names_source()}


# ---- Language detection (ko|en|es) ----
KO_PARTICLES = {"은","는","이","가","을","를","으로","하고","하다","그리고","그러나","하지만","에서"}
ES_STOP = {"el","la","los","las","de","del","que","y","en","un","una","para","como","por","con","al","es","no","sí"}
EN_STOP = {"the","a","an","of","and","to","in","for","on","with","as","is","are","it","that"}


def _lang_scores(text: str) -> Dict[str, float]:
    hangul = sum(0x3130 <= ord(ch) <= 0x318F or 0xAC00 <= ord(ch) <= 0xD7A3 for ch in text)
    latin  = sum(('a' <= ch.lower() <= 'z') for ch in text)
    es_marks = sum(ch in 'áéíóúüñ¡¿ÁÉÍÓÚÜÑ' for ch in text)
    # token-level hints
    tokens = [t.lower() for t in re.split(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+", text) if t]
    ko_hits = sum(t in KO_PARTICLES for t in tokens)
    es_hits = sum(t in ES_STOP for t in tokens)
    en_hits = sum(t in EN_STOP for t in tokens)
    scores = {
        "ko": hangul * 2 + ko_hits * 3,
        "es": es_marks * 3 + es_hits * 2 + max(0, latin - en_hits) * 0.2,
        "en": en_hits * 2 + latin * 0.5,
    }
    return scores


def detect_lang_simple(text: str) -> Dict[str, object]:
    scores = _lang_scores(text)
    lang = max(scores.items(), key=lambda kv: kv[1])[0]
    total = sum(scores.values()) or 1.0
    conf = scores[lang] / total
    return {"lang": lang, "confidence": round(conf, 4), "scores": {k: round(v, 2) for k, v in scores.items()}}


class DetectReq(BaseModel):
    text: str


@app.post("/detect_lang")
def detect_lang(req: DetectReq):
    return detect_lang_simple(req.text)
