# Nasazení nové struktury na MetaCentrum a spuštění opravného běhu

**Datum: 2026-08-03.** Navazuje na `RERUN-PREREGISTRACE.md` (závazná kritéria)
a `DIAGNOSTIKA-VYPADKU.md` (příčina). Přihlášení: `ssh tarkil`.

**Pořadí je závazné. Starý strom `ttd-bench/` se nemaže, dokud neprojde
krok 1.** Do té chvíle je to jediná off-site kopie 367 MB zmrazených dat.

---

## 0. Co na klastru zůstává

| co | proč |
|---|---|
| `ttd-bench-data/` | datasety, 16 GB, nesahat |
| `ttd-bench/checkpoints/` | 1,1 GB vah, přesunout do nového stromu, ne nahrávat znovu |
| `ttd-bench/third_party/` | mediamtx + statický ffmpeg (MetaCentrum ffmpeg nemá libx264) |
| `ttd-bench/` celý | dokud neprojde krok 1 |

## 1. Ověřit, že lokální kopie je bit-shodná

Manifest z klastru (běží nebo doběhl) se porovná se stavem na Macu. Cesty se
musí přemapovat, protože lokální strom je restrukturalizovaný:

```
results-scvd/…            -> results/scvd-phase2/…
results-phase1-final/…    -> results/phase1-final/…
```

Manifest patří do repa jako auditní artefakt, ne do dočasného adresáře:

```bash
rsync -av tarkil:manifest-frozen.md5 "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Skola/Programs/Větev-A-DiP/results/cluster-audit/manifest-frozen-2026-08-03.md5"
```

Porovnání:

```bash
cd "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Skola/Programs/Větev-A-DiP" && .venv/bin/python common/scripts/verify_cluster_manifest.py results/cluster-audit/manifest-frozen-2026-08-03.md5
```

**Dokud nevypíše nulové rozdíly v obou směrech, nemaže se na klastru nic.**

## 2. Nasadit novou strukturu vedle staré

Repozitář je privátní; nedávat na sdílený klastr přihlašovací údaje.
Nejjednodušší je poslat pracovní strom včetně `.git` z Macu — na klastru pak
vznikne plnohodnotný git checkout se správným `git_hash`, bez tokenů.

Z Macu, do **nového** adresáře, starý zůstává nedotčený:

```bash
rsync -av --exclude '.venv' --exclude 'results/' --exclude 'archiv/Data-zenodo' --exclude 'clanek-2-ttdbench/checkpoints' --exclude 'clanek-2-ttdbench/third_party' --exclude 'datasets' "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Skola/Programs/Větev-A-DiP/" tarkil:/storage/brno12-cerit/home/hav0254/vetev-a/
```

Na klastru pak přenést velké věci ze starého stromu (přesun v rámci téhož
úložiště je okamžitý, nekopíruje se):

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254 && mkdir -p vetev-a/clanek-2-ttdbench && mv ttd-bench/checkpoints vetev-a/clanek-2-ttdbench/checkpoints && mv ttd-bench/third_party vetev-a/clanek-2-ttdbench/third_party && ls vetev-a/clanek-2-ttdbench'
```

Ověřit, že checkout je čistý a na správném commitu:

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a && git status --short && git log --oneline -1'
```

## 3. Prostředí

Staví se skriptem `hpc/setup_venv.sh`, ne ručně. Dělá to v sedmi krocích a
každý z nich vznikl z konkrétní chyby, na kterou se narazilo:

* atomický zámek (`mkdir`) a odmítnutí startu, když běží jiný pip — dva
  souběžné pipy nad jedním venv si rozeberou torch a výsledek vypadá jako
  poškozený wheel;
* **torch a torchvision první**, z cu126 indexu, **se závislostmi**. CUDA
  runtime dnes chodí jako samostatné `nvidia-*` wheely, takže `--no-deps`
  u torche znamená, že se ani nenaimportuje (`libcudart.so.12: cannot open
  shared object file`). cu126 je nutnost: konos má GTX 1080 Ti = sm_61 a novější
  CUDA buildy Pascal zahodily;
* pin toho, co se právě nainstalovalo, plus **pin knihoven na verze zmrazeného
  běhu** (`hpc/frozen-libs.txt`, generovaný z `hpc/frozen-header.jsonl`).
  Bez něj přijde ultralytics 8.4.115 proti zmrazeným 8.4.103 a onnxruntime
  1.28.0 proti 1.27.0 — a test ekvivalence by pak měřil opravu mediamtx
  *a zároveň* upgrade obou inferenčních backendů;
* na konci import všeho a porovnání prostředí proti zmrazené hlavičce.

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a && nohup bash clanek-2-ttdbench/hpc/setup_venv.sh > ~/venv.log 2>&1 < /dev/null & echo spusteno'
```

To `< /dev/null` je podstatné: bez něj si backgroundovaný proces drží stdin
ssh kanálu a terminál visí, i když build běží v pořádku.

Cesty ve scénářích jsou absolutní macové, přepsat na klastrové. Tím se strom
stane špinavým, což je očekávané a projeví se jako `git_dirty: true` v každé
hlavičce:

```bash
ssh tarkil "cd /storage/brno12-cerit/home/hav0254/vetev-a && sed -i 's|/Users/macbook/Datasets/ttd-bench|/storage/brno12-cerit/home/hav0254/ttd-bench-data|g' common/configs/scenarios-*.yaml && grep -c ttd-bench-data common/configs/scenarios-scvd.yaml"
```

Symlinky `clanek-2-ttdbench/results*` míří do `../results/`, které se na klastr
nenasazuje — založit jim prázdné cíle, ať neselžou:

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a && mkdir -p results/scvd-phase2 results/phase1-final results/phase1-early results/smoke'
```

Preflight a kontrola matice (`--list-cells` končí souhrnným řádkem, proto
`tail -1`, ne `wc -l`):

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench && ../.venv/bin/python scripts/run_single.py --config configs/experiment-scvd.yaml --preflight && ../.venv/bin/python scripts/run_single.py --config configs/experiment-scvd.yaml --list-cells | tail -1'
```

## 4. Co se v `pbs_array.sh` mění a jak číst `select`

Cesty se needitují. Skript bere `REPO_DIR` z `PBS_O_WORKDIR`, takže stačí
**odesílat z `clanek-2-ttdbench/`** — pak `configs/…` i `scripts/run_single.py`
sedí. Jen se předá `VENV_DIR`, protože venv je o úroveň výš.

### Chunkování (nové)

`CELLS_PER_JOB` (výchozí 1) sbalí souvislý úsek `CELL_LIST` do jednoho jobu a
pustí ho **sekvenčně**. Pro rerun `CELLS_PER_JOB=10` → **83 jobů místo 825**.

Proč: měřená režie je **~4,5 min na job** proti mediánu buňky **54 s**
(rozsah 10–120 s). Chunk po deseti tu režii amortizuje devětkrát a hlavně —
83 jobů plánovač uspokojí řádově snáz než 825. Sekvenční běh navíc znamená,
že **job nemůže kolidovat sám se sebou**.

Jedna spadlá buňka nezabije zbytek chunku; selhání se sesbírají a job skončí
nenulovým kódem se seznamem. Čtení seznamu je `while read`, ne `mapfile` —
`mapfile` potřebuje bash 4+ a tenhle skript má být ověřitelný i na stroji, kde
se píše (macOS má bash 3.2).

### `select` řádek, položka po položce

Tohle je audit, ne opis. Každá položka zužuje množinu uzlů, a právě tím se
platí frontou.

| položka | verdikt |
|---|---|
| `ngpus=1` | **nutné** |
| `cl_konos=True` | **nutné vědecky** — týž typ GPU (GTX 1080 Ti) jako zmrazený běh. Bez toho se srovnatelnost ztratí dřív, než ji stihne otestovat kontrolní množina. Tohle je hlavní příčina fronty a nedá se obětovat. |
| `os=debian13` | **nutné při venv režimu** — venv se staví na frontendu (TRIXIE) a jiný python minor na uzlu rozbije `site-packages`. Odpadlo by při běhu v kontejneru (`hpc/apptainer.def`, `USE_CONTAINER=1`); to je jediná legitimní cesta, jak tuhle podmínku zahodit. |
| `ncpus=8` | **ponechat** — zmrazený běh měl 8 a edge-sim si omezuje vlákna sám; měnit prostředí kvůli frontě by poškodilo srovnatelnost. |
| `mem=32gb` | ponechat ze stejného důvodu; největší checkpoint má 66 MB, takže je to velkorysé, ale neškodí. |
| `scratch_local=20gb` | **VYHODIT.** Ve skriptu se `SCRATCHDIR` nikde nepoužívá a `OUT_DIR` míří rovnou na sdílené úložiště. Je to alokovatelný zdroj, kterým se zbytečně zužuje výběr uzlů, a nic za to nedostáváme. |
| `walltime=02:00:00` | **zkrátit na 01:00:00** pro chunk po deseti. Odhad: 10× medián = 9 min, 10× p90 = 14 min, 10× nejhorší pozorovaný případ = 20 min. Hodina je trojnásobná rezerva a **kratší walltime se výrazně líp backfilluje** — což je přesně to, co teď potřebujeme. |

## 5. Spuštění — nejdřív ověřovací test, pak plný běh

### 5a. Ověřovací test: dvě souběžné buňky na JEDNOM uzlu — HOTOVO 2026-08-03

Nesmí se vynechat ani zmenšit: jednoslotový běh tu chybu z principu nevyvolá,
proto proběhlo 1597 buněk zmrazeného běhu bez jediné kolize.

Pouští se `hpc/smoke_concurrent.sh`, **ne** pole `-J 0-1`. Dvě položky pole jsou
dvě nezávislé úlohy a plánovač je smí dát na různé uzly — pak nic nekoliduje a
zelený výsledek nic neznamená. Skript proto pouští obě buňky na pozadí uvnitř
jedné úlohy, takže je překryv jistý.

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a && qsub -q gpu -l select=1:ncpus=8:ngpus=1:mem=32gb:os=debian13:cl_konos=True -l walltime=00:30:00 -v VENV_DIR=/storage/brno12-cerit/home/hav0254/vetev-a/.venv clanek-2-ttdbench/hpc/smoke_concurrent.sh'
```

Test je citlivý, ověřeno v obou směrech lokálně: bez `protocols: [tcp]` jedna
ze dvou buněk padne s produkčním řetězcem `ERR listen udp :8000: bind: address
already in use`, s ní projdou obě.

**Výsledek na klastru: prošel.** Obě buňky doběhly (exit 0/0), nula kolizí,
`git_dirty: true` podle očekávání. Kovariáta souběžnosti funguje — v `end`
záznamu obou běhů jsou oba PID po 316 MiB, tedy se navzájem viděly.

Pozor na jednu vlastnost: v **hlavičce** je `gpu_compute_apps` prázdné pole.
Není to chyba — hlavička se zapisuje dřív, než se model nahraje na GPU
(`runner.py` ř. 133 vs 184), takže hlavička zachycuje *cizí* procesy, které na
kartě byly už při startu, a `end` záznam ty, které tam byly na konci. Filtr
souběžnosti v předregistraci §4 se čte právě takhle.

**Vedlejší, ale zásadní zjištění tohoto běhu:** obě buňky existují i ve
zmrazeném běhu, takže z něj vypadlo první číslo o efektu živého souseda — a to
číslo stáhlo plán B. Viz `RERUN-PREREGISTRACE.md` §3.

### 5b. Plný opravný běh: 775 selhaných + 50 kontrolních

Kontrolní buňky musí být **v témže poli**, aby sdílely podmínky
(předregistrace §5). Spojený seznam:

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench && cat hpc/rerun/cells_failed.txt hpc/rerun/cells_control.txt > hpc/rerun/cells_rerun.txt && wc -l hpc/rerun/cells_rerun.txt'
```

Musí být **825 řádků** → při `CELLS_PER_JOB=10` to je **83 jobů, `-J 0-82`**
(poslední chunk má 5 buněk). Ověřeno lokálně: sjednocení chunků se rovná
původnímu seznamu, index 83 už je prázdný a skript by skončil `exit 2`.

**Plán A — s exkluzivitou, lhůta 5 dní** (předregistrace §3):

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench && qsub -q gpu -J 0-82 -l select=1:ncpus=8:ngpus=1:mem=32gb:os=debian13:cl_konos=True -l place=excl -l walltime=01:00:00 -v EXPERIMENT_CONFIG=configs/experiment-scvd.yaml,RESULTS_SUBDIR=results-scvd-rerun,CELL_LIST=$PWD/hpc/rerun/cells_rerun.txt,CELLS_PER_JOB=10,VENV_DIR=/storage/brno12-cerit/home/hav0254/vetev-a/.venv hpc/pbs_array.sh'
```

**Plán B je STAŽEN.** Ověřovací běh změřil, že živý soused posouvá latenci
o 5 SD a ztrátovost čtyřnásobně — neexkluzivní data nejsou se zmrazenými
srovnatelná a stratifikace to nespraví. Podrobnosti a čísla v
`RERUN-PREREGISTRACE.md` §3. Když se `place=excl` nedá naplánovat, rerun se
nedělá; D1 zůstává a je mechanisticky obhájená. Příkaz níže se **nepoužívá** a
zůstává tu jen jako záznam toho, co bylo zavrženo:

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench && qsub -q gpu -J 0-82 -l select=1:ncpus=8:ngpus=1:mem=32gb:os=debian13:cl_konos=True -l walltime=01:00:00 -v EXPERIMENT_CONFIG=configs/experiment-scvd.yaml,RESULTS_SUBDIR=results-scvd-rerun,CELL_LIST=$PWD/hpc/rerun/cells_rerun.txt,CELLS_PER_JOB=10,VENV_DIR=/storage/brno12-cerit/home/hav0254/vetev-a/.venv hpc/pbs_array.sh'
```

Volbu plánu zapsat do labbooku s datem. Před odesláním plánu B zkontrolovat,
kolik toho ve frontě stojí a proč:

```bash
ssh tarkil 'qstat -u hav0254 -t | tail -20; echo ---; qstat -f $(qstat -u hav0254 -t | awk "NR==6{print \$1}") 2>/dev/null | grep -E "comment|Resource_List"'
```

`RESULTS_SUBDIR=results-scvd-rerun` je záměrně jiný než zmrazený běh —
opravené buňky přistanou stranou a zmrazená data se nepřepíšou.

Oproti §4: `scratch_local` vypuštěn (nepoužívá se), walltime zkrácen na hodinu
(lepší backfill), `cl_konos` a `os=debian13` ponechány s odůvodněním.

### 5c. Kontrola po doběhnutí

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench && ls results-scvd-rerun/raw/*.jsonl | wc -l && grep -L "\"type\":\"end\"" results-scvd-rerun/raw/*.jsonl | wc -l && grep -l "address already in use" ttd-bench.o* | wc -l'
```

825 / 0 / 0. Cokoli jiného znamená zastavit a diagnostikovat, ne dopočítávat.
Chunkovaný job navíc sám hlásí, co v něm spadlo:

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench && grep -h "cells FAILED" ttd-bench.o* | head'
```

A kontrola, že logování souběžnosti opravdu naskočilo (jinak plán B nemá
kovariátu a test §4 nelze provést):

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench && head -1 results-scvd-rerun/raw/*.jsonl | grep -c gpu_compute_apps'
```

## 6. Odvoz a vyhodnocení

```bash
rsync -av tarkil:/storage/brno12-cerit/home/hav0254/vetev-a/clanek-2-ttdbench/results-scvd-rerun/ "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Skola/Programs/Větev-A-DiP/results/scvd-phase2-rerun/"
```

Pak se spustí test ekvivalence podle předregistrace §4 nad 50 kontrolními
buňkami. Teprve jeho výsledek rozhoduje, jestli se opravené buňky slučují se
zmrazenými, nebo reportují jako oddělená vrstva.

## 7. Úklid starého stromu — až úplně nakonec

Po kroku 1 (manifest sedí) a po úspěšném odvozu opravného běhu:

**Bezpečné hned** — je to v gitu nebo je to smetí: `results-phase1.1/`,
`ttd-bench.o22*` (3760 souborů), `cudatest.sh`, `cudatest*.out`, soubor
s názvem `ources_available.gpu_cap = …` (ustřelený redirect), `rtdetr-l.pt`.

**Až po třetí kopii** (Zenodo): `results-scvd/`, `results-phase1-final/`.

**Neznámé, nejdřív se podívat**: `Old/`, `logs-run1/`, `logs-run2/`.
