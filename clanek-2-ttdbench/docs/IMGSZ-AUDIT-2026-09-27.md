# Audit imgsz: replay vs. FAR kalibrace (2026-09-27)

Read-only audit. Neměnil jsem žádný soubor článku 2 ani `common/`; nic neběželo
na clusteru. Pomocné skripty a mezivýsledky jsou v dočasném scratchpadu
(v dočasném adresáři: `imgsz_check.py`, `score_events.py`,
`match_replay.py`, `event_impact.py`, `calib416.py`, `subset_calib.py`,
`replay_adjust2.py`, `boot.py`). Scratchpad se po relaci smaže, takže pokud mají
čísla zůstat reprodukovatelná, je potřeba skripty přesunout do repa.

## Verdikt

**Potvrzeno.** Ve fázi 2 běžel `yolov8s-weapon` v konfiguracích local-gpu a
remote (lan/wifi/4g) na vstupu **416**, v edge-sim na **640**. Ostatní tři
detektory běžely všude na 640. FAR kalibrace skórovala všechny čtyři modely na
640. Pro yolov8s se tedy prahy kalibrované na 640 aplikují na skóre z 416 ve
4 z 5 konfigurací. Věta v rukopisu „inference was performed at 640 pixels for
all models in both phases“ (`ttd-bench-scirep.tex:188`) je nepravdivá.

## 1. Chování ultralytics 8.4.103

Použil jsem čisté venv (Python 3.11, torch 2.14 CPU/MPS, `ultralytics==8.4.103`,
tedy verzi ze zmrazené hlavičky `hpc/frozen-libs.txt` a raw JSONL). Checkpointy
jsem načetl stejně jako `UltralyticsBackend`: warm-up na dummy 640×640, potom
`predict(frame, device, conf=0.10)` bez `imgsz`.

| checkpoint | `train_args.imgsz` | `predictor.args.imgsz` | tenzor pro snímek 854×480 |
|---|---|---|---|
| yolov8s-weapon.pt | 416 | 416 | (1, 3, 256, 416) |
| yolov8m-weapon.pt | 640 | 640 | (1, 3, 384, 640) |
| yolov12m-weapon.pt | 640 | 640 | (1, 3, 384, 640) |
| yolov26m-weapon.pt | 640 | 640 | (1, 3, 384, 640) |

Mechanismus ve zdroji 8.4.103:
- `engine/model.py:283-286` (`Model._load`): pro `.pt` se volá
  `self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)`.
- `engine/model.py:1050-1070` (`_reset_ckpt_args`) ponechá
  `include = {"imgsz", "data", "task", "single_cls"}`, takže **imgsz z tréninku
  se přenáší do predikce**.
- `engine/model.py:529-530` (`predict`): `args = {**self.overrides, **custom, **kwargs}`.
  Default 640 z `cfg/default.yaml:16` se tedy použije jen tehdy, když checkpoint
  imgsz nenese. Nastavuje se také `rect: True`, takže letterbox je obdélníkový.
- `engine/predictor.py:259` (`setup_source`): `self.imgsz = check_imgsz(self.args.imgsz, ...)`.

Warm-up na dummy 640×640 (`inference.py:112`) velikost vstupu nemění, jen
založí predictor s imgsz=416. Checkpointy fáze 1 (COCO `yolov8n`, `yolov8m`,
`rtdetr`) mají v 8.4.103 imgsz 640, takže **fáze 1 je v pořádku**.

## 2. Jiná místa, která imgsz nastavují, a co zaznamenávají raw logy

- `run_single.py`, `runner.py`, `replay.py`, `ingest.py` ani configy imgsz
  nenastavují. `build_backend` (`inference.py:316-335`) předává jen weights,
  device a threads. Komentář `configs/experiment-scvd.yaml:46` („imgsz 640“)
  je nepřesný, protože pro run_8_s platí 416.
- **ONNX (edge-sim):** všechny weapon exporty mají pevný vstup
  **[1,3,640,640]**, včetně `yolov8s-weapon.onnx`. Metadata uvádějí
  `imgsz [640, 640]`, ultralytics 8.4.91 a datum 2026-07-22, takže v8s se
  exportoval s explicitním `imgsz=640`. Exportní skript pro weapon modely v repu
  není, pro COCO modely ho dělá `scripts/fetch_coco_checkpoints.py:54`
  (imgsz=640). `OnnxBackend` bere velikost z tvaru vstupu (`inference.py:236-238`).
- **Raw JSONL** vstupní tvar ani velikost letterboxu nezaznamenávají. Hlavička
  obsahuje jen verze knihoven (`"ultralytics":"8.4.103"`), per-frame záznamy
  obsahují detekce a časy. Raw logy fáze 2 lokálně nejsou, lokálně jsou jen
  `results-scvd/analysis/traces*.parquet`.
- **Empirické potvrzení ze zmrazených tras.** Offline skóre yolov8s na 34
  event klipech (MPS, 416 i 640) jsem porovnal s per-frame `max_conf`
  zmrazených replay tras. Snímky jsem zarovnal přes `t_rel_onset_ms` (offset
  −2 snímky, při něm je shoda nejlepší):

| replay konfigurace | n snímků | MAE vs 416 | MAE vs 640 | r(416) | r(640) | shoda rozhodnutí @0.5 s 416 / 640 |
|---|---|---|---|---|---|---|
| local-gpu | 32 453 | **0.052** | 0.178 | **0.954** | 0.541 | **0.918** / 0.673 |
| remote (3 profily) | 45 268 | **0.077** | 0.178 | **0.890** | 0.539 | **0.875** / 0.679 |
| edge-sim (ONNX) | 3 968 | 0.194 | **0.142** | 0.493 | **0.679** | 0.685 / **0.757** |

  Zbytková odchylka u 416 (0,05) odpovídá H.264/RTSP řetězci a markeru. Replay
  data tedy nezávisle na čtení kódu potvrzují 416 pro ultralytics konfigurace
  a 640 pro edge-sim.

## 3. Dopad (lokálně, bez clusteru)

### 3a. Event klipy (34 SCVD Weaponized, 7 635 snímků, yolov8s offline, MPS)

| | pre-onset (n=3 804) | post-onset (n=3 831) |
|---|---|---|
| medián max_conf 416 / 640 | 0.344 / 0.410 | 0.370 / 0.460 |
| q95 416 / 640 | **0.877 / 0.809** | **0.899 / 0.838** |
| q99 416 / 640 | 0.921 / 0.891 | 0.921 / 0.869 |
| průměr \|416−640\| | 0.176 | 0.176 |
| korelace | 0.563 | 0.557 |

Při 416 má skóre nižší střed a těžší horní chvost. Podíl snímků nad prahem
(pre / post-onset, 416 vs 640) na kalibrovaných prazích yolov8s:

| práh (cíl) | pre ≥ thr 416/640 | post ≥ thr 416/640 | nesouhlas rozhodnutí | klipy s hitem (K=3, offline) 416/640 |
|---|---|---|---|---|
| 0.500 (fixed) | 0.289 / 0.372 | 0.319 / 0.437 | 30.8 % | 26 / 28 |
| 0.935 (12/h) | 0.002 / 0.003 | 0 / 0 | 0.2 % | 0 / 0 |
| 0.910 (60/h) | 0.014 / 0.009 | 0.024 / 0.000 | 2.3 % | 2 / 0 |
| 0.885 (120/h) | 0.040 / 0.010 | 0.070 / 0.003 | 5.1 % | 3 / 1 |
| 0.810 (300/h) | 0.087 / 0.050 | 0.086 / 0.070 | 4.9 % | 5 / 7 |
| 0.675 (600/h) | 0.139 / 0.150 | 0.138 / 0.197 | 13.9 % | 15 / 20 |

Na prazích ≥ 0.95 (0.1–3/h) nepřekročí práh žádný snímek v žádném rozlišení.
Per-clip shoda rozhodnutí @0.5 mezi .pt@416 a 640 je v průměru 0.70
(min 0.20, max 1.00). Rukopis uvádí shodu ONNX vs PyTorch 0.81–1.00
(`:184`), jenže `backend_check.csv` srovnává .pt@640 s ONNX@640. Skutečná
dvojice ve fázi 2 (.pt@416 vs ONNX@640) se shoduje hůř.

### 3b. Orientační kalibrace při 416 (JEN INDIKATIVNÍ)

Jde o náhodný podvzorek 32 klipů korpusu dostupných lokálně (seed 0, klipy
< 600 s): 12 SCVD Normal, 3 UCF test, 17 UCF train-1, celkem **0.483 h** a
52 161 snímků. MD5 všech klipů souhlasí s manifestem. Yolov8s@416 jsem
skóroval na MPS, 640 jsem vzal z `calib_traces.parquet`. Počty snímků per
klip sedí. Kontrola platformy na 4 klipech (MPS@640 vs cluster@640): střední
\|Δ\| 0.011, max 0.15. Je to o řád méně než rozdíl 416/640 (0.16), ale víc než
dřívější kritérium A4 (max < 0.01). Pro závěry to není podstatné, jen to
zaznamenávám.

FAR (epizody/h, K=3, re-arm) na podvzorku při papírových prazích yolov8s:

| práh | cíl | achieved v článku (93 h, 640) | podvzorek @640 | podvzorek @416 | poměr 416/640 |
|---|---|---|---|---|---|
| 0.500 | fixed | 1247.7 | 1443 | 926 | **0.64** |
| 0.935 | 12/h | 11.0 | 6.2 | 16.6 | 2.7 |
| 0.925 | 30/h | 23.4 | 8.3 | 37.3 | 4.5 |
| 0.910 | 60/h | 53.1 | 22.8 | 45.6 | 2.0 |
| 0.885 | 120/h | 116.6 | 37.3 | 82.8 | 2.2 |
| 0.810 | 300/h | 291.8 | 292 | 197 | **0.67** |
| 0.675 | 600/h | 588.9 | 845 | 302 | **0.36** |

Z toho plyne, že skutečná FAR yolov8s v local-gpu/remote se od nominální liší
zhruba faktorem 0,4–4,5. V pásmu 12–120/h je vyšší (práh byl ve skutečnosti
mírnější), v pásmu 300–1248/h nižší (práh byl přísnější).

**Dopad na Tab. 2 (yolov8s, local-gpu).** Hit rate jsem přepočítal ze
zmrazených replay tras. Reprodukce papírových hodnot sedí přesně: 73.5 / 5.9 /
5.9 / 17.6 / 32.4 %. Potom jsem použil prahy „416-ekvivalentní“: nejmenší
práh, při kterém FAR@416 na podvzorku ≤ FAR@640 papírového prahu. Interval je
90% clip bootstrap (300×) přes podvzorek.

| cíl | papír práh / hit | 416-ekvivalentní práh [90 % int.] | hit [90 % int.] |
|---|---|---|---|
| 120/h | 0.885 / 5.9 % | 0.915 [0.855, 0.935] | 5.9 % [0.0, 8.8] |
| 300/h | 0.810 / 17.6 % | 0.685 [0.650, 0.870] | 32.4 % [8.8, 44.1] |
| 600/h | 0.675 / 32.4 % | 0.548 [0.500, 0.600] | **64.7 % [55.9, 73.5]** |

Remote má stejný směr (600/h: 29.3 → ~60 %). Pro cíle ≤ 60/h zůstává
hit yolov8s 0 až ~6 % a pořadí se nemění. **Při 600/h (pravděpodobně i
300/h) by se ale yolov8s posunul z posledního nebo předposledního místa
k YOLOv8m** (Tab. 2 při 600/h: v8m 70.7, v26m 61.8, v12m 43.5). Tvrzení
„YOLOv8s was last or next to last at every calibrated operating point“
(`:130`), pořadí v SI (`SI:279`) a HR YOLOv8m vs referenční v8s při 600/h
(2.35, `SI:284`) jsou proto v horním pásmu ohrožené. Interval nezahrnuje
nejistotu reprezentativnosti podvzorku: podvzorek je „teplejší“ než korpus
(845 vs 589/h při 0.675). Berte to jako směr a řád velikosti, ne jako čísla
do článku.

**Fixed threshold 0.5.** Výsledky replaye (73.5 %, Cox, Tab. 1) jsou tím, co
systém při 416 skutečně udělal, a jako měření platí. Nesedí k nim ale
přiřazená FAR 1247.7/h, která je naměřená při 640. Podvzorek naznačuje
~0.64×, tedy řádově ~800/h. Yolov8s by pořád měl zdaleka nejvyšší FAR
(v8m 334/h), jen rozpětí „6.7-fold“ by kleslo zhruba na 4–5× a „jeden alarm
každých 2.9 s“ na ~4–5 s. Zároveň platí, že **edge-sim yolov8s (ONNX@640) je
s kalibrací konzistentní**, takže variant b pro edge-sim v8s je v pořádku.
Porovnání prahů edge-sim vs local-gpu pro v8s (0.525 vs 0.885, `:148`) ale
míchá dvě rozlišení.

## 4. Místa, která uvádějí 640 nebo implikují jednotný vstup

### Rukopis (`paper/scirep/`)
- `ttd-bench-scirep.tex:188`: „inference was performed at 640 pixels for all
  models in both phases“. **Nepravda, musí se opravit.**
- `ttd-bench-scirep.tex:56`, `:90`, `:99` (caption Tab. 1), `:168`:
  confound popisují jen jako *trénovací* rozlišení. Chybí, že se liší i
  inferenční rozlišení, a to navíc mezi konfiguracemi (v8s 416 v GPU
  konfiguracích, 640 v edge-sim).
- `ttd-bench-scirep.tex:184`: shoda ONNX–PyTorch 0.81–1.00. Pro v8s neplatí
  pro skutečnou dvojici .pt@416 vs ONNX@640.
- `ttd-bench-scirep.tex:216`: „Score collection used the same confidence floor
  of 0.1 as the frozen logs“. Implikuje shodnou inferenční cestu, ale
  rozlišení u v8s se liší.
- `ttd-bench-scirep.tex:126`, `:130`: yolov8s 1248/h, „every 2.9 s“,
  „last or next to last at every calibrated operating point“.
- `ttd-bench-scirep.tex:135` a `tables/t2_hit_calibrated.tex` (sloupec
  YOLOv8s, generuje `common/scripts/scirep_tables.py`).
- `ttd-bench-scirep.tex:148`: edge-sim vs local-gpu prahy v8s 0.525 vs 0.885.
- SI `ttd-bench-scirep-SI.tex:235` (v8s 0.5 → 1247.7), `:243` (achieved
  0.730/h), `:250` (tabulka prahů), `:274-279` (pořadí), `:284` (Cox při
  kalibrovaných rychlostech vs referenční v8s), `:299` (edge-sim se liší
  „backendem a výpočtem“, rozlišení nezmíněno).

### Dokumentace
- `clanek-2-ttdbench/docs/DEPLOY-FAR-SCORING.md:110-121` („imgsz = 640 pro
  všechny čtyři modely … ultralytics default 640“), což je chybná premisa.
  Dále `:143` (Mac reference imgsz 640).
- `clanek-2-ttdbench/docs/ANALYSIS-PLAN-PHASE2.md:200` („ultralytics backend,
  the Phase-2 local-gpu path“).
- `clanek-2-ttdbench/results-far-calib/README.md:13`.
- `clanek-2-ttdbench/configs/experiment-scvd.yaml:46` (komentář „imgsz 640“).
- Starší zdroje, které se do Sci Rep nepropisují:
  `paper/sections/03-methods.md:169-170` („inference runs at 640 throughout“),
  `paper/latex/main.tex:183`, `paper/latex/main-prepsany.tex:183`,
  `paper/build/clanek-2-full.{md,html}`, `paper/review/mech/_concat.md:330`,
  `paper/review/mech/claims.json:597`.

### Kód (docstringy a konstanty)
- `common/scripts/far_score_clips.py:12-21` (docstring: „no `imgsz`, i.e. the
  ultralytics default 640, for every model“) a `:42` (`IMGSZ = 640 # …
  replay's effective imgsz`).
- `common/scripts/far_backend_check.py:4-7, 31` (srovnává .pt@640 s ONNX@640,
  ne se skutečnou replay dvojicí).
- `common/scripts/far_infer.py:32` (`IMGSZ = 640`).
- `common/scripts/far_platform_check.py:5, 10` („same inference path“,
  „imgsz 640“).
- `common/src/ttdbench/inference.py:100-123`: `UltralyticsBackend` imgsz
  nepředává. Warm-up 640×640 na `:112` svádí k dojmu, že vstup je 640.

## 5. Možnosti a doporučení

**(a) Přeskórovat korpus pro yolov8s při 416 na MetaCentru a přepočítat
kalibraci.**
- Náklad: 10,15 M snímků (94,03 h stopáže). Podle README (`:124`) stojí
  4 modely ~76 GPU-h na L40, jeden model tedy ≤ 19 GPU-h. Yolov8s@416 je
  lehčí a bude spíš limitovaný dekódováním, odhad **~10–20 GPU-h na L40,
  ~25–45 GPU-h na 1080 Ti**. Walltime s 2× rezervou. Stačí stejné pole
  `pbs_far_scoring.sh` s `far_score_clips.py` s parametrem imgsz a jen
  yolov8s (to vyžaduje úpravu `common/`). Potom lokálně `far_traces_merge.py`,
  `far_sweep.py`, `far_reanalysis.py` a regenerace tabulek a obrázků
  (`scirep_build.py`), to jsou minuty.
- **Vyžaduje výslovné schválení autora (MetaCentrum). Nespouštěl jsem.**
- Rozhodnutí o návrhu: variant a pro v8s se stane konfiguračně závislý
  (416 práh pro local-gpu/remote, 640 práh pro edge-sim), případně se edge-sim
  v8s označí jako mismatch. Obojí je potřeba popsat jako novou odchylku D11.
- Text: oprava `:188`; změna čísel v8s v Tab. 2, SI tabulkách prahů, pořadí a
  Cox; revize vět `:126`, `:130` (pravděpodobně změna tvrzení pro 300–600/h),
  `:148`; D11 v SI (Supplementary Note deviations); oprava
  DEPLOY-FAR-SCORING a docstringů.
- Časově: 1–3 dny včetně fronty a regenerace. Riziko: změní se kvalitativní
  tvrzení o v8s v horním pásmu, abstrakt ani hlavní závěr (kalibrace mění
  pořadí) ale ne.

**(b) Znovu spustit replay yolov8s při 640.**
- 34 scénářů × 5 opakování × 4 ultralytics konfigurace = 680 běhů s RTSP
  a síťovou emulací, řádově 10–20 h walltime na GPU uzlech se zmrazeným
  prostředím.
- Mění zmrazená a preregistrovaná primární data (Tab. 1, Cox při 0.5, KM,
  latence v8s) a vyžaduje přepočet celé fáze 2. Navíc by se ztratila
  věrnost „nasazení tak, jak proběhlo“.
- **Do termínu podání se nevejde. Nedoporučuji.**

**(c) Uvést to jako limitaci s naměřeným lokálním dopadem.**
- Minimálně: opravit `:188` (např. „YOLOv8s was trained at 416 pixels and,
  because the harness did not override the checkpoint's stored input size,
  was also run at 416 pixels in the GPU-served configurations; its ONNX export
  used in edge-sim, and all other models, ran at 640 pixels.“), doplnit to do
  limitací `:168` a do SI Note calibration. Do SI dát odstavec s měřením
  z tohoto auditu: replay trasy potvrzují 416, FAR při papírových prazích se
  u v8s liší faktorem 0,4–4,5 a v pásmu 300–600/h je hit v8s pravděpodobně
  podhodnocený.
- Dále zmírnit `:130` („last or next to last at every calibrated operating
  point“ omezit na ≤ 120/h nebo opatřit výhradou), `:126` (FAR při 0.5 uvést
  jako měřenou při 640) a `:184`.
- Čas: hodiny, bez clusteru.
- Slabina: Tab. 2 a SI by dál obsahovaly čísla v8s, o kterých víme, že jsou
  v horním pásmu pravděpodobně zkreslená o desítky procentních bodů.
  Recenzent, který si všimne, že kalibrace neodpovídá nasazení, to může
  označit za metodickou chybu, ne za limitaci.

**Doporučení:** okamžitě udělat **(c)** jako povinné minimum, protože `:188`
je faktická chyba a musí se opravit v každém případě. Pokud termín snese
1–3 dny, **požádat autora o schválení (a)**. Je levná (~10–20 GPU-h L40),
nesahá na zmrazená data fáze 2 a opravuje kalibraci u 4 z 5 konfigurací.
Oprava je pak konzistentní a limitace se zúží na „v8s běžel v edge-sim při
640, jinde při 416“. **(b) nedoporučuji.** Pokud (a) schválena nebude, v (c)
explicitně uvést, že kalibrovaná čísla yolov8s pro local-gpu/remote platí
pro 640, že skutečný provoz byl 416, a uvést orientační směr zkreslení.


## Skripty a mezivýsledky
Zkopírováno ze scratchpadu relace do `imgsz-audit-2026-09-27/` (skripty auditu, skóre 416/640, porovnání s replay trasami). Cesty uvnitř skriptů mohou odkazovat na scratchpad — před znovuspuštěním upravit.
