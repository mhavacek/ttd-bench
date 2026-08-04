# Fáze 1 — latenční pilot (COCO detektory, per-frame dekompozice)

## Co fáze 1 měří a proč je platná teď

Fáze 1 kvantifikuje **vliv konfigurace nasazení na per-frame latenční
dekompozici** (Δt_acq, Δt_transfer, Δt_infer, Δt_post) a **drop rate**, tj.
jádro centrální hypotézy (identická architektura → různé TTD podle nasazení).
Tyto latence závisí na **architektuře**, ne na trénovaných třídách, takže
COCO-předtrénované YOLOv8n/m a RT-DETR dávají **finálně platná časová čísla**,
která se přenesou na doučené weapon-checkpointy ve fázi 2.

Co fáze 1 **neměří**: event-level TTD, hit/miss, miss-rate heatmapu — ty
vyžadují detektor se třídou „weapon" a ověřené onsety (fáze 2). Alarm proto ve
fázi 1 nikdy nevystřelí (COCO třídu „weapon" nemá) → každý běh je na úrovni
události „miss"; to je očekávané. Výstupem jsou sloupce `pf_*` a Phase-1
grafy/tabulka.

Metodika měření času, definice alarmu (K, práh, timeout) a formát raw logů
jsou **totožné** s fází 2 — obě fáze jsou proto přímo srovnatelné a fáze 2 se
jen přidá ke stejným raw logům.

## Lokální ověření (bez GPU)

```bash
.venv/bin/pip install -e '.[ml]'                 # ultralytics + onnxruntime
.venv/bin/python scripts/fetch_coco_checkpoints.py   # yolov8n/m + rtdetr (+ONNX)

# edge-sim (ONNX/CPU) běží i bez CUDA:
.venv/bin/python scripts/run_single.py --config configs/experiment-phase1.yaml --cell-index 27
# ultralytics buňky na stroji bez CUDA přepněte na CPU/MPS:
.venv/bin/python scripts/run_single.py --config configs/experiment-phase1.yaml --cell-index 0 --device cpu
```

## Submit na MetaCentru

Postup je identický s [METACENTRUM.md](METACENTRUM.md), jen s Phase-1 configem:

1. Přenos repa + dat + **checkpointů** (`checkpoints/*.pt`, `*.onnx` — nejsou
   v gitu; buď je rsyncněte, nebo na frontendu spusťte
   `python scripts/fetch_coco_checkpoints.py`).
2. Opravit cesty v `configs/scenarios-ucf-phase1.yaml`
   (`sed -i "s|/Users/macbook/Datasets/ttd-bench|$DATA|g"`).
3. V `hpc/pbs_array.sh` nastavit
   `EXPERIMENT_CONFIG="configs/experiment-phase1.yaml"` (viz níže) a `DATA_DIR`.
4. Velikost matice:
   ```bash
   python scripts/run_single.py --config configs/experiment-phase1.yaml --list-cells | tail -1
   # 135 buněk (5 konfigurací × 3 modely × 3 scénáře × 3 opakování)
   ```
5. Generálka v interaktivním jobu (pytest + jedna edge-sim + jedna local-gpu
   buňka), pak:
   ```bash
   qsub -q gpu -J 0-134 hpc/pbs_array.sh
   ```
6. Agregace a Phase-1 assety:
   ```bash
   python scripts/aggregate.py --config configs/experiment-phase1.yaml
   python scripts/make_paper_assets.py \
       --master results-phase1/csv/master.csv \
       --out-root results-phase1 --phase1
   ```

Výstupy (`results-phase1/`):
- `tables/phase1_perframe_summary.{csv,tex}` — střední per-frame dekompozice
  na konfiguraci × model + drop rate
- `figures/fig_pf_decomposition.pdf` — skládaný sloupec latenčních komponent
- `figures/fig_pf_infer_violin.pdf` — rozložení Δt_infer na konfiguraci × model
- `figures/fig_pf_droprate.pdf` — drop rate na konfiguraci × model

## Poznámka k `hpc/pbs_array.sh`

Skript má natvrdo `configs/experiment.yaml`. Pro fázi 1 buď:
- přidejte na začátek `EXPERIMENT_CONFIG="configs/experiment-phase1.yaml"` a
  předejte ho do `run_single.py --config "$EXPERIMENT_CONFIG"`, nebo
- jednorázově zkopírujte `experiment-phase1.yaml` na `experiment.yaml`.

(První varianta je čistší a je připravená — viz proměnná v hlavičce skriptu.)

## Přechod na fázi 2

Až budou hotové (a) doučené weapon-checkpointy a (b) ověřené onsety
(SCVD od studenta + verifikované UCF onsety), stačí v `experiment.yaml`
přepnout `active_models` na weapon-checkpointy a `scenarios_file` na ostrou
sadu a submitnout znovu. Event-level TTD, miss rate a všechny 4 hlavní grafy
(`make_paper_assets.py` bez `--phase1`) se vygenerují nad novými raw logy;
Phase-1 latenční čísla zůstávají v platnosti a jdou reportovat vedle nich.
