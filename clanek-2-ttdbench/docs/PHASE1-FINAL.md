# Fáze 1 — FINÁLNÍ zmrazený běh (protokol)

Zásada: **všechna čísla fáze 1 v článku pocházejí výhradně z tohoto běhu.**
Pilot (reps=3, heterogenní uzly) sloužil k vyladění infrastruktury; finální
běh má 5 opakování (§3), homogenní hardware a je zmrazený git tagem + config
hashem.

## 0. Zmrazení kódu (na Macu, jednou)

```bash
cd ttd-bench
git tag -a phase1-final -m "Phase 1 final frozen run"
git rev-parse phase1-final        # hash do labbooku
```

Po tagu žádné změny kódu před/behem finálního běhu. (Config hash každého
běhu je v hlavičce raw logu — musí být jeden jediný napříč všemi 225 logy;
ověření viz krok 5.)

## 1. Výběr homogenního hardwaru (na frontendu)

Vypište GPU klastry s OS a kartou a vyberte JEDEN klastr pro hlavní běh
(1080 Ti = konos) a JEDEN moderní (A40/A100) pro modern-GPU sloupec:

```bash
pbsnodes -a | grep -E "^[a-z]|cluster =|gpu_cap =|os =" \
  | paste - - - - 2>/dev/null | sort -u | less
# alternativně přehledně na webu: https://metavo.metacentrum.cz/pbsmon2/hardware
```

Podmínky pro kandidátní klastr: `os = debian13` (venv!), jednotný typ GPU.
Pinning se dělá vlastností klastru v selektoru, např. pro konos:
`:cl_konos=True`.

## 2. Hlavní finální běh (5 konfigurací, 1080 Ti)

```bash
cd /storage/brno12-cerit/home/hav0254/ttd-bench
# preflight nad finálním configem (225 buněk -> 0-224)
.venv/bin/python scripts/run_single.py --config configs/experiment-phase1-final.yaml --preflight

qsub -q gpu -J 0-224 \
     -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=20gb:os=debian13:cl_konos=True \
     -v EXPERIMENT_CONFIG=configs/experiment-phase1-final.yaml,RESULTS_SUBDIR=results-phase1-final \
     hpc/pbs_array.sh
```

Pozn.: `-l select` na příkazové řádce přebíjí hlavičku skriptu, takže
cluster pinning nevyžaduje žádnou editaci souboru.

## 3. Modern-GPU sloupec (45 buněk)

Po výběru moderního klastru (např. galdor/A40 — zkontrolujte OS! pokud není
debian13, vyberte jiný debian13 klastr s moderní kartou):

```bash
.venv/bin/python scripts/run_single.py --config configs/experiment-phase1-moderngpu.yaml --preflight

qsub -q gpu -J 0-44 \
     -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=20gb:os=debian13:cl_<MODERNI_KLASTR>=True \
     -v EXPERIMENT_CONFIG=configs/experiment-phase1-moderngpu.yaml,RESULTS_SUBDIR=results-phase1-moderngpu \
     hpc/pbs_array.sh
```

## 4. Kontroly po doběhnutí

```bash
grep -L '"type":"end"' results-phase1-final/raw/*.jsonl | wc -l      # musí být 0
grep -L '"type":"end"' results-phase1-moderngpu/raw/*.jsonl | wc -l  # musí být 0
```

## 5. Ověření zmrazení (jeden config hash, jeden git hash, jeden typ GPU)

```bash
for d in results-phase1-final results-phase1-moderngpu; do
  echo "== $d =="
  for f in $d/raw/*.jsonl; do head -1 "$f"; done \
    | .venv/bin/python -c "
import json,sys
H=set(); G=set(); GPU=set()
for line in sys.stdin:
    h=json.loads(line); H.add(h['config_hash']); G.add(h['git_hash']); GPU.add(h['gpu'])
print('config_hash:', H); print('git_hash:', G); print('gpu:', GPU)"
done
```

Každá množina musí mít právě jeden prvek (u `gpu` jeden typ karty).
Tyto tři hodnoty patří do labbooku a do reprodukční sekce článku.

## 6. Agregace a assety

```bash
.venv/bin/python scripts/aggregate.py --config configs/experiment-phase1-final.yaml
.venv/bin/python scripts/make_paper_assets.py \
    --master results-phase1-final/csv/master.csv --out-root results-phase1-final --phase1

.venv/bin/python scripts/aggregate.py --config configs/experiment-phase1-moderngpu.yaml
.venv/bin/python scripts/make_paper_assets.py \
    --master results-phase1-moderngpu/csv/master.csv --out-root results-phase1-moderngpu --phase1
```

Sloučení obou sloupců do jedné publikační tabulky se dělá na Macu nad oběma
master.csv (config_label `local-gpu` vs `local-gpu-modern` se nekryjí).

## 7. Odvoz VŠEHO včetně raw logů (zdroj pravdy)

```bash
# z Macu:
rsync -av hav0254@tarkil.metacentrum.cz:/storage/brno12-cerit/home/hav0254/ttd-bench/results-phase1-final/ \
    ~/Desktop/results-phase1-final/
rsync -av hav0254@tarkil.metacentrum.cz:/storage/brno12-cerit/home/hav0254/ttd-bench/results-phase1-moderngpu/ \
    ~/Desktop/results-phase1-moderngpu/
```

Raw logy archivovat (Zenodo/institucionální úložiště) — článek na ně odkáže.

---

## 8. Co skutečně proběhlo — doložené odchylky (dopsáno 2026-08-03)

Běh doběhl 24. 7. 2026, stažen 3. 8. 2026 do `data-hpc/results-phase1-final/`
(225 raw, 131 csv). Kontrola z kroku 5 **prošla**: jeden config hash
`48c0b912a4f390ef`, jeden git hash, jeden typ GPU (GTX 1080 Ti), uzly výhradně
`konos1`–`konos7`. Hardwarová homogenita, kvůli které se běh dělal, je splněna.

### O1 — běh není na tagu `phase1-final`

Běh je z commitu `356842d`, tedy **dva commity za tagem** `phase1-final`
(`4254639` / `f7e99b8`); §0 je tím formálně porušena. Proč to nevadí:

* diff v `src/` + `scripts/` je čistě aditivní (+91 / −6 řádků; těch 6 jsou
  změny signatur přidávající `timeout_s`),
* přidávají se jen pole `*_meas_ms` (duální definice alarmu — funkce fáze 2),
  nic v cestě `pf_*` / `drop_rate`, na které fáze 1 stojí,
* `configs/experiment-phase1-final.yaml` je mezi tagem a během bajt po bajtu
  totožný,
* `356842d` je **tentýž commit jako zmrazený běh fáze 2**, takže obě fáze
  pocházejí z jednoho stavu kódu.

Do reprodukční sekce patří `356842d`, ne tag. Původním tagem se nehýbe (je to
záznam); skutečný bod běhu nese tag `phase1-final-run`.

### O2 — 94 z 225 běhů (41,8 %) zpracovalo nula snímků

Týž režim selhání jako D1 ve fázi 2, ale zhruba dvojnásobná míra (fáze 2:
22,8 %). Všech 94 je header-only, žádný běh není „bez `end`, ale se snímky" —
jde o selhání infrastruktury úlohy, ne o zmeškanou detekci.

Nezávislost na obsahu drží čistěji než ve fázi 2:

| test | výsledek |
|---|---|
| selhání vs. scénář | chi² = 0,15, df = 2, **p = 0,93** |
| selhání vs. model | chi² = 1,35, df = 2, **p = 0,51** |
| selhání vs. opakování | chi² = 6,47, df = 4, **p = 0,17** |
| selhání vs. pořadí opakování | rho = +0,08, **p = 0,22** |
| selhání vs. konfigurace | chi² = 18,89, df = 4, **p = 0,0008** |

Závislost je jen na konfiguraci (local-gpu 13,3 % vs. edge/remote 46,7–51,1 %).

**Dopsáno 2026-08-03 — příčina je určená, viz
[`DIAGNOSTIKA-VYPADKU.md`](DIAGNOSTIKA-VYPADKU.md).** Původní domněnka („roste
s počtem subprocesů, které konfigurace startuje") **neplatí**. Všech 94 běhů
padlo na `mtx.start()` s chybou `listen udp :8000: bind: address already in
use`, tedy dřív, než se dekódoval první snímek: `mediamtx` binduje pevné UDP
porty :8000/:8001, které šablona configu neparametrizuje, a retry losuje jen
TCP port. Spouštěčem je druhý souběžný job na uzlu. Po podmínění tímhle
režimem **efekt konfigurace mizí beze zbytku (chi² = 0,31, p = 0,989)** —
v jednoslotovém režimu 0/32 selhání, v dvouslotovém 94/193 rovnoměrně napříč
konfiguracemi (42,9–51,1 %). Vyloučení tím dostává mechanistické zdůvodnění
místo pouze empirického.

**Pozor — past:** nulové běhy nesou `drop_rate = 0.0`, ne `NaN`. Nechat je
v průměru **půlí hlavní číslo** (edge-sim 0,82 → 0,44; 4G 0,66 → 0,34) a ředí
`dt_transfer`. `make_paper_assets.py --phase1` proto filtr `n_frames_processed
> 0` aplikuje sám a vypíše, kolik běhů vyloučil. Žádnou míru nikdy nepočítat
z nefiltrovaného `master.csv`.

### O3 — opakování jsou 2–5, ne 5 (§3)

Po vyloučení nezůstala žádná buňka prázdná, ale rozdělení je 16 buněk po 2,
22 po 3, 2 po 4 a jen 5 po pěti.

Obhajoba: fáze 1 tvrdí dekompozici latence (compute-bound vs. network-bound),
což je timing s malým rozptylem, ne šumivý detekční výsledek. Relativní
půlšířka 95% CI na buňku (sdruženo přes 3 scénáře, n = 7–15) je **0,6–7,9 %
u `pf_dt_infer_ms` a 0,5–5,1 % u `pf_latency_ms`**. U `drop_rate` vypadá
relativní CI velká (37–58 %) jen tam, kde je drop blízko nule (local-gpu,
remote-lan) — absolutní půlšířka je tam 0,00–0,06. Pět opakování z §3 bylo pro
tuhle třídu tvrzení předimenzované; pět reálně potřebuje survival analýza
fáze 2. Reportovat u `drop_rate` absolutní CI, ne relativní.

### O4 — „4G drop je model-nezávislý" je formulace, kterou je nutné opravit

Pilotní formulace neobstojí doslova. Uvnitř 4G se drop mezi modely liší:
rtdetr 70,2 %, yolov8m 65,2 %, yolov8n 64,0 % (ANOVA F = 42,6, p < 0,001; SD
jen 0,012–0,016).

Není to ale protimluv — je to přesně to, co predikuje analýza cyklu: síťová
podlaha 4G si vynutí >55 % drop pro *jakýkoli* model a čas serveru se přičítá
(rtdetr +10–17 ms → +4–6 p.b.). Predikce +4–6 p.b., naměřeno **+6,2 p.b.**

Správná formulace je kontrast efektů, ne nezávislost:

| režim | ANOVA drop ~ model | rozsah drop napříč modely |
|---|---|---|
| edge-sim (compute-bound) | F = 8401, p = 3e-31 | 60,8 → 95,9 % (35,1 p.b.) |
| remote-4G (network-bound) | F = 42,6, p < 0,001 | 64,0 → 70,2 % (6,2 p.b.) |

Obojí je signifikantní, ale efekty se liší o řád. To je silnější a
kvantifikované tvrzení než „model-nezávislý": **režimy se skládají,
dekompozice je lokalizuje.**

### Co z toho plyne pro kampaň na Karolině

Homogenitu hardwaru fáze 1 už máme a všechny nálezy pilotu se replikovaly
(edge yolov8n 72 → 77 ms, yolov8m 555 → 573, rtdetr 768 → 819; transfer
3/16/70 → 3,2/18,3/69,3; rtdetr na local-gpu 12,7 → 12,1 % drop). Fáze 1 se dá
vykázat z těch 131 platných běhů s odchylkami O1–O4. Karolina fázi 1 přidá už
jen sloupec moderní GPU (`results-phase1-moderngpu/`, který nikdy neběžel) —
nice-to-have. Skutečně ji potřebuje **fáze 2**: oprava 775 buněk a homogenní
přeběh pro survival analýzu.
