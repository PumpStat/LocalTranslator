from typing import List, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, json

app = FastAPI(title="RhythmGame Translation (Prompt-only, TagBank+Rules+Examples)")

# ===== 모델 설정 (RTX 3080: 7B 4bit 권장) =====
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"   # 대안: "meta-llama/Meta-Llama-3.1-8B-Instruct"
USE_4BIT = True
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg if USE_4BIT else None,
    torch_dtype=torch.float16 if not USE_4BIT else None,
    device_map="auto",
)

LANGS = {"ko", "en", "es"}

# ===== Tag Bank (영/한, ES는 직역 지향) =====
TAGS_EN = [
  "drill","gimmick","twist","bracket","side","run","stamina","half","weight_shift",
  "mash","high-angle_twist","horizontal_twist","long_notes","no_bar","fast","slow",
  "drag","jack","3bit","stair","switch","jump","leg_split","12bit","24bit","32bit","double stairs"
]
TAGS_KO = [
  "떨기","기믹","틀기","겹발","사이드","폭타","체력","하프","체중이동",
  "뭉개기","고각틀기","수평틀기","롱잡","무봉","고속","저속",
  "끌기","연타(클릭)","3비트","계단","스위칭","점프","다리찢기","12비트","24비트","32비트","겹계단"
]
assert len(TAGS_EN) == len(TAGS_KO)

ES_DIRECT = {
  "drill":"drill","gimmick":"gimmick","twist":"twist","bracket":"bracket","side":"lado",
  "run":"ráfaga","stamina":"resistencia","half":"mitad","weight_shift":"cambio de peso",
  "mash":"mash","high-angle_twist":"twist de alto ángulo","horizontal_twist":"twist horizontal",
  "long_notes":"notas largas","no_bar":"sin barra","fast":"rápido","slow":"lento",
  "drag":"arrastre","jack":"jack","3bit":"3 bit","stair":"escalera","switch":"cambio",
  "jump":"salto","leg_split":"apertura de piernas","12bit":"12 bit","24bit":"24 bit",
  "32bit":"32 bit","double stairs":"escaleras dobles"
}

TAG_MAP_EN2KO = dict(zip(TAGS_EN, TAGS_KO))
TAG_MAP_KO2EN = dict(zip(TAGS_KO, TAGS_EN))
TAG_MAP_EN2ES = ES_DIRECT

def tag_table_for_target(tgt: str) -> Dict[str, str]:
    table = {}
    if tgt == "ko":
        for en, ko in TAG_MAP_EN2KO.items(): table[en] = ko
    elif tgt == "en":
        for en in TAGS_EN: table[en] = en
        for ko, en in TAG_MAP_KO2EN.items(): table[ko] = en
    elif tgt == "es":
        # 스페인어: 가능하면 직역, 하이픈/언더바는 "출력 시" 공백 처리(프롬프트로 유도)
        for en in TAGS_EN: table[en] = TAG_MAP_EN2ES.get(en, en)
        for ko, en in TAG_MAP_KO2EN.items(): table[ko] = TAG_MAP_EN2ES.get(en, en)
    return table

# ===== 보호 대상: 심플하게 프롬프트에만 명시(치환/검증 없음) =====
DEFAULT_AUTHORS = ["EXC", "SPHAM", "DULKI", "SUNNY", "FEFEMZ"]  # 채보 제작자
DEFAULT_CHART_NAMES = [
    "Pandora", "District V", "Prime Time"
    # 필요한 채보명 계속 추가 가능
]

# ===== 시스템 프롬프트 빌더 =====
def build_system_prompt(target_lang: str,
                        authors: List[str],
                        chart_names: List[str]) -> str:
    table = tag_table_for_target(target_lang)

    return (
        "You are a professional translator specialized in arcade rhythm games "
        "(Pump It Up, maimai, DDR). Follow the glossary and rules STRICTLY.\n\n"
        f"TARGET LANGUAGE: {target_lang}\n\n"

        # Glossary
        "GLOSSARY (Tag-Table, target-specific JSON). If a source sentence contains a key "
        "or its natural variants, you MUST output the exact mapped value:\n"
        + json.dumps(table, ensure_ascii=False, indent=2) + "\n\n"

        # Protected tokens and proper nouns
        "PROTECTED (NEVER translate; keep verbatim, preserve casing):\n"
        "- S1~S26, D1~D28, Coop2~Coop5, C2~C5, BPM\n"
        f"- Chart authors (proper nouns): {authors}\n"
        f"- Chart names (proper nouns): {chart_names}\n\n"

        # Rules
        "SPECIAL RULES:\n"
        "1) break on / break off:\n"
        "   - If target is Korean (ko): output '브렉온' / '브렉오프'.\n"
        "   - If target is English/Spanish: keep 'break on' / 'break off'.\n"
        "2) Clear/Fail mapping:\n"
        "   - Korean <-> English: '깨다', '클리어', '클리어함' ↔ 'clear'.\n"
        "   - '못깨다', '클리어하지 못함' ↔ 'fail'.\n"
        "   - For Spanish, '못깨다/클리어하지 못함' ↔ 'no superado'.\n"
        "3) Notes count:\n"
        "   - Korean 'N놋' ↔ 'N notes' (English/Spanish both use 'notes').\n"
        "4) Spanish styling:\n"
        "   - When target is Spanish, replace '-' and '_' with spaces in translated terms.\n"
        "5) Keep punctuation and numbers stable. DO NOT add explanations. Output only the translated text.\n"
    )

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
]

# ---- 메시지 빌더 ----
def build_messages(src: str, tgt: str, text: str,
                   examples: List[Dict[str, str]],
                   authors: List[str],
                   chart_names: List[str]) -> list:
    sys = build_system_prompt(tgt, authors, chart_names)
    msgs = [{"role": "system", "content": sys}]

    # few-shot 예시(최대 12개 정도 권장)
    for ex in examples[:12]:
        ex_src = ex.get("source", src)
        ex_tgt = ex.get("target", tgt)
        ex_in  = ex["input"]
        ex_out = ex["output"]
        msgs.append({"role":"user","content":f"Translate from {ex_src} to {ex_tgt}:\n{ex_in}"})
        msgs.append({"role":"assistant","content":ex_out})

    msgs.append({"role": "user", "content": f"Translate from {src} to {tgt}:\n{text}"})
    return msgs

def generate(messages: list) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2, top_p=0.9, do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt[len(prompt):].strip()

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

@app.post("/translate")
def translate(req: TranslateReq):
    assert req.source in LANGS and req.target in LANGS, "unsupported language"
    ex = []
    if req.use_default_examples:
        ex.extend(DEFAULT_EXAMPLES)
    if req.examples:
        ex.extend([e.model_dump() for e in req.examples])

    authors = req.authors or DEFAULT_AUTHORS
    chart_names = req.chart_names or DEFAULT_CHART_NAMES

    messages = build_messages(req.source, req.target, req.text, ex, authors, chart_names)
    out = generate(messages)
    return {"source": req.source, "target": req.target, "text": req.text, "translated": out}