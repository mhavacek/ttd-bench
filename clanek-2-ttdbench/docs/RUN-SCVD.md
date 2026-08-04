# Step-by-step: SCVD event-level experiment na MetaCentru

Spuštění plného experimentu (event-level TTD + dekompozice + miss rate) na
SCVD Weaponized klipech s onsety oanotovanými studentem. Scénáře jsou
integrované a zvalidované v `configs/scenarios-scvd.yaml` (34 použitelných
klipů: 26 train + 8 test; zbylých 90 vyřazeno — zbraň v záběru od začátku
nebo příliš pozdě).

---

## ⚠️ Krok 0 — PŘEDPOKLAD: weapon checkpointy

Tento experiment měří TTD = čas od objevení zbraně po alarm. Alarm vzniká
detekcí třídy „weapon". **Potřebujete detektory doučené na zbraně** — COCO
checkpointy (yolov8n.pt atd.) třídu „weapon" nemají a alarm by nikdy
nevystřelil (všechny běhy = miss, nepoužitelné).

- Máte weapon checkpointy? → pokračujte krokem 1, dejte je do `checkpoints/`
  a vypište v `configs/experiment-scvd.yaml` (`models` / `active_models`).
- Nemáte je zatím? → SCVD běh počká; mezitím jde spustit **Phase-1 latenční
  pilot** ([PHASE1.md](PHASE1.md)) s COCO checkpointy, který weapon třídu
  nepotřebuje a dá dekompozici latence per konfigurace.

Kontrola, že jsou checkpointy na místě (nic nespouští, jen ověří konfiguraci):
```bash
.venv/bin/python scripts/run_single.py --config configs/experiment-scvd.yaml --preflight
```
Dokud chybí weapon checkpointy, vypíše to:
```
2 PROBLEM(S):
  - model 'yolov8n-weapon' (ultralytics): missing checkpoint checkpoints/yolov8n-weapon.pt
  - model 'yolov8n-weapon' (onnxruntime): missing checkpoint checkpoints/yolov8n-weapon.onnx
```

Až checkpointy budete mít, ověřte jednu buňku (očekávejte `hit=True` alespoň
u části scénářů):
```bash
.venv/bin/python scripts/run_single.py --config configs/experiment-scvd.yaml --cell-index 0 --device cpu
```

---

## Proměnné prostředí (přizpůsobte)

```bash
export MC_USER=<login>
export FRONTEND=skirit.metacentrum.cz
export MC_HOME=/storage/brno2/home/$MC_USER
export REPO=$MC_HOME/ttd-bench
export DATA=$MC_HOME/ttd-bench-data          # sem patří SCVD/ a checkpointy
```

## Krok 1 — Přenos repa, dat a checkpointů

```bash
# z lokálního stroje (adresář nad ttd-bench):
rsync -av --exclude .venv --exclude 'results*' --exclude third_party \
    ttd-bench/ $MC_USER@$FRONTEND:$REPO/

# SCVD videa (~2 GB):
rsync -av ~/Datasets/ttd-bench/scvd/ $MC_USER@$FRONTEND:$DATA/scvd/

# weapon checkpointy (nejsou v gitu — *.pt/*.onnx jsou gitignored):
rsync -av ttd-bench/checkpoints/ $MC_USER@$FRONTEND:$REPO/checkpoints/
```

## Krok 2 — Oprava cest ke scénářům (KRITICKÉ)

`configs/scenarios-scvd.yaml` má absolutní macOS cesty
(`/Users/macbook/Datasets/ttd-bench/scvd/...`). Na frontendu:

```bash
cd $REPO
sed -i "s|/Users/macbook/Datasets/ttd-bench/scvd|$DATA/scvd|g" \
    configs/scenarios-scvd.yaml
grep -m1 "file:" configs/scenarios-scvd.yaml     # ověřit, že cesta existuje
ls "$(grep -m1 'file:' configs/scenarios-scvd.yaml | awk '{print $2}')"
```

## Krok 3 — Prostředí (kontejner nebo venv)

```bash
cd $REPO
bash scripts/fetch_mediamtx.sh                   # linux RTSP binárka
apptainer build ttd-bench.sif hpc/apptainer.def  # preferovaně
# fallback bez kontejneru:
#   module add python/3.11 ffmpeg
#   python3 -m venv .venv && .venv/bin/pip install -e '.[ml,dev]'
#   a v hpc/pbs_array.sh USE_CONTAINER=0
```

## Krok 4 — Nastavení PBS skriptu a velikost matice

V `hpc/pbs_array.sh` (hlavička) nastavte:
```bash
REPO_DIR="/storage/brno2/home/<login>/ttd-bench"   # absolutní!
DATA_DIR="/storage/brno2/home/<login>/ttd-bench-data"   # bind do kontejneru
EXPERIMENT_CONFIG="configs/experiment-scvd.yaml"
RESULTS_SUBDIR="results-scvd"
QUEUE="gpu"
```

Velikost matice = 5 konfigurací × N modelů × 34 scénářů × 5 opakování.
**Preflight ověří checkpointy i videa a rovnou vypíše správný rozsah pole** —
spusťte ho na frontendu vždy před `qsub`:
```bash
.venv/bin/python scripts/run_single.py --config configs/experiment-scvd.yaml --preflight
```
```
config:    configs/experiment-scvd.yaml
models:    yolov8n-weapon
scenarios: 34
matrix:    850 cells  ->  qsub -J 0-849
preflight OK — ready to submit
```
(1 model → 850 buněk, 4 modely → 3400 buněk.)
Doporučení: **začněte s 1 modelem** (`active_models: [yolov8n-weapon]`),
ověřte celý běh, pak přidejte zbytek.

## Krok 5 — Generálka v interaktivním jobu (nepřeskakovat)

```bash
qsub -I -q gpu -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=20gb \
     -l walltime=2:00:00
# na uzlu (proměnné si nastavte znovu — job má čisté prostředí):
cd $REPO
BIND="--bind $REPO:$REPO,$DATA:$DATA"
apptainer exec --nv $BIND ttd-bench.sif python3 -m pytest -q
apptainer exec --nv $BIND ttd-bench.sif python3 scripts/run_single.py \
    --config configs/experiment-scvd.yaml --cell-index 0   # očekávej hit=True
exit
```
Pokud cell 0 dá `hit=False` u weapon checkpointu, zkontrolujte práh/třídy
(alarm.weapon_class_names vs. názvy tříd modelu) — ne submitujte celé pole,
dokud jedna buňka nevrací hit.

## Krok 6 — Submit pole

```bash
cd $REPO
qsub -q gpu -J 0-849 hpc/pbs_array.sh      # rozsah dle kroku 4!
```
- Pod-job = 1 buňka (deterministické mapování indexu → faktory).
- Walltime 2 h/buňku je konzervativní (SCVD klip má 3–13 s; běh ~1–2 min).
- Velké pole lze rozdělit: `-J 0-424` a `-J 425-849`.

## Krok 7 — Monitoring

```bash
qstat -u $MC_USER                                  # stav pole
ls $REPO/results-scvd/raw | wc -l                  # hotové běhy
grep -L '"type":"end"' $REPO/results-scvd/raw/*.jsonl   # nedoběhnuté
qsub -q gpu -J 17-17 hpc/pbs_array.sh              # rerun jedné buňky
```

## Krok 8 — Agregace a odvoz výsledků

```bash
cd $REPO
.venv/bin/python scripts/aggregate.py --config configs/experiment-scvd.yaml
.venv/bin/python scripts/make_paper_assets.py \
    --master results-scvd/csv/master.csv --out-root results-scvd
# (bez --phase1: plné event-level TTD, dekompozice, CDF, miss-rate heatmapa)

# z lokálního stroje:
rsync -av $MC_USER@$FRONTEND:$REPO/results-scvd/ ./results-scvd/
```

Výstupy: `results-scvd/tables/*.{csv,tex}` (TTD summary s CI, dekompozice,
párové testy + Bonferroni + effect size) a `figures/*.pdf` (skládaná
dekompozice, box/violin, CDF s TTD₅₀/₉₀, miss-rate heatmapa scénář × konfig).

## Poznámky ke scénářům

- **Onsety jsou z SCVD anotací studenta** (první snímek se zbraní v záběru).
  Pro článek doporučuji namátkovou kontrolu ~10 klipů (viz
  `scripts/annotate_onsets.py`) a report mezianotátorské shody.
- Vyřazené klipy zůstávají v `scenarios-scvd.yaml` (nejsou v `active_scenarios`)
  s `tags.excluded_reason` — nic se neztratilo, jen se neměří.
- Tagy osvětlení/vzdálenost/okluze jsou `unknown` (SCVD je nemá); miss-rate
  heatmapa proto poběží po `scenario` id, ne po tagu. Chcete-li subgroup
  analýzu, tagy lze doplnit ručně.
