# Návod: submit TTD-Bench na MetaCentru (PBS Pro)

Aktuální dokumentace MetaCentra: https://docs.metacentrum.cz — názvy front a
frontendů se občas mění, před ostrým během zkontrolujte. Kroky níže
předpokládají účet e-INFRA CZ s aktivním členstvím v MetaCentru.

## 0. Proměnné (přizpůsobte si)

```bash
export MC_USER=<vas_login>                 # MetaCentrum username
export FRONTEND=skirit.metacentrum.cz      # či tarkil, zenith, nympha…
export MC_HOME=/storage/brno2/home/$MC_USER
export REPO=$MC_HOME/ttd-bench
export DATA=$MC_HOME/ttd-bench-data
```

Domovský adresář frontendu leží na /storage svazku viditelném ze všech
výpočetních uzlů — repo i data patří tam (NE do /tmp ani scratch).

## 1. Přenos repa a dat z Macu

```bash
# z lokálního stroje (adresář nad ttd-bench):
rsync -av --exclude .venv --exclude results --exclude third_party \
    ttd-bench/ $MC_USER@$FRONTEND:$REPO/

rsync -av ~/Datasets/ttd-bench/ $MC_USER@$FRONTEND:$DATA/
```

Data (UCF ~11 GB rozbalené, SCVD ~2 GB, Sevilla ~3 GB) + checkpointy
(`checkpoints/*.pt`, `*.onnx`) — checkpointy rsyncněte zvlášť, v repu jsou
gitignored.

## 2. KRITICKÉ: opravit cesty ve scénářích

`configs/scenarios-ucf.yaml` a `scenarios-scvd-template.yaml` obsahují
absolutní macOS cesty (`/Users/macbook/Datasets/ttd-bench/...`). Na frontendu:

```bash
cd $REPO
sed -i "s|/Users/macbook/Datasets/ttd-bench|$DATA|g" configs/scenarios-*.yaml
grep -m1 "file:" configs/scenarios-ucf.yaml   # ověřit
```

## 3. Prostředí na frontendu

```bash
cd $REPO
bash scripts/fetch_mediamtx.sh        # stáhne linux binárku do third_party/

# (a) kontejner — preferovaná cesta:
export APPTAINER_TMPDIR=$SCRATCHDIR   # v interaktivním jobu; na frontendu /tmp
apptainer build ttd-bench.sif hpc/apptainer.def
# pokud build na frontendu selže na oprávněních, použijte interaktivní job
# (viz krok 5) nebo venv fallback:

# (b) venv fallback (bez kontejneru):
module avail python ffmpeg            # zjistit přesné názvy modulů
module add python/3.11.* ffmpeg
python3 -m venv .venv
.venv/bin/pip install -e '.[ml,dev]'
# v hpc/pbs_array.sh pak USE_CONTAINER=0
```

## 4. Konfigurace experimentu

1. `configs/experiment.yaml`: `active_models` (reálné checkpointy),
   `scenarios_file` (ostrá sada scénářů s ověřenými onsety).
2. `hpc/pbs_array.sh`: vyplnit `REPO_DIR="$REPO"` (absolutní cestou),
   `QUEUE="gpu"`, `PROJECT_ID` nechte prázdné (MetaCentrum neúčtuje na
   projekt jako IT4I).
3. Zjistit velikost matice:
   ```bash
   .venv/bin/python scripts/run_single.py --list-cells | tail -1
   # např. "total: 200 cells"  ->  pole 0-199
   ```

## 5. Generálka v interaktivním jobu (nutné!)

```bash
qsub -I -q gpu -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=20gb \
     -l walltime=2:00:00
# na uzlu:
cd $REPO   # (proměnnou si nastavte znovu, job má čisté prostředí)
apptainer exec --nv --bind $REPO:$REPO,$DATA:$DATA ttd-bench.sif \
    python3 -m pytest -q                                  # testy
apptainer exec --nv --bind $REPO:$REPO,$DATA:$DATA ttd-bench.sif \
    python3 scripts/run_single.py --synthetic             # syntetický e2e
apptainer exec --nv --bind $REPO:$REPO,$DATA:$DATA ttd-bench.sif \
    python3 scripts/run_single.py --cell-index 0          # reálná buňka
exit
```

Pozn.: `--bind $DATA` je nutný, jinak kontejner neuvidí videa mimo repo.
Do `hpc/pbs_array.sh` přidejte `$DATA` do `--bind`, pokud data nejsou pod
`REPO_DIR` (výchozí skript binduje jen repo).

## 6. Submit pole

```bash
cd $REPO
qsub -q gpu -J 0-199 hpc/pbs_array.sh     # rozsah dle kroku 4.3!
```

- Pod-job = 1 buňka matice (deterministické mapování `PBS_ARRAY_INDEX` →
  `--cell-index`), běží vždy na jednom uzlu (jedna clock doména).
- Walltime 2 h/buňku je konzervativní (typický běh = délka videa + ~2 min).
- Kolize portů souběžných pod-jobů na témže uzlu řeší per-cell offset.

## 7. Monitoring

```bash
qstat -u $MC_USER          # stav pole
qstat -Jtx <job_id>        # stav jednotlivých pod-jobů
ls $REPO/results/raw | wc -l           # kolik raw logů už je hotových
grep -L '"type":"end"' $REPO/results/raw/*.jsonl   # nedoběhnuté běhy
# stdout/stderr pod-jobů: soubory ttd-bench.o<jobid>.<index> v $PBS_O_WORKDIR
```

Selhané buňky lze přespustit jednotlivě: `qsub -q gpu -J 17-17 hpc/pbs_array.sh`
(rerun přepíše raw log dané buňky).

## 8. Agregace a odvoz výsledků

```bash
# na frontendu (nepotřebuje GPU):
cd $REPO
.venv/bin/python scripts/aggregate.py
.venv/bin/python scripts/make_paper_assets.py

# z lokálního stroje:
rsync -av $MC_USER@$FRONTEND:$REPO/results/ ./results-metacentrum/
```

Raw logy (`results/raw/*.jsonl`) jsou zdroj pravdy — archivujte je celé
(~5 MB/běh); tabulky a grafy jdou kdykoli přegenerovat.

## Časté problémy

| Symptom | Příčina / řešení |
|---|---|
| `mediamtx binary not found` | nespuštěný `scripts/fetch_mediamtx.sh`, nebo binárka pro špatnou architekturu (spouštějte na frontendu, ne na Macu) |
| `FileNotFoundError: scenario video` | neopravené cesty (krok 2) nebo chybějící `--bind $DATA` |
| kontejner nevidí GPU | chybí `--nv` u `apptainer exec`, nebo job bez `ngpus=1` |
| pod-joby padají hromadně hned po startu | špatný `REPO_DIR` v pbs_array.sh (musí být absolutní /storage cesta) |
| `qsub: Job violates queue and/or server resource limits` | příliš velké pole najednou — rozdělte rozsah, např. `-J 0-99` a `-J 100-199` |
| onset za koncem streamu (missing onset_ref) | UCF metadata lžou — importér už počítá snímky dekódováním; u vlastních videí spusťte import/anotátor znovu |
