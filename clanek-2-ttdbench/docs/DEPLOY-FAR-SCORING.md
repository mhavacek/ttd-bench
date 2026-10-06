# Nasazení FAR kalibrace na MetaCentrum

**Datum: 2026-08-12.** Postup pro fáze A a B podle schváleného plánu.
Rozhodnutí R1: **celý korpus se skóruje na clusteru, na jedné platformě**;
Mac (MPS) by byla třetí výpočetní cesta v článku. R2: cíl ~100 h stopáže,
FAR 1/h s odhadem, 0,1/h jako jednostranná mez.

**Proč je tohle opačný případ než opravný běh.** Skóre je deterministická
funkce vah a vstupního snímku — GPU do výsledku nevstupuje. Proto se submituje
**bez jakýchkoli constraintů**: žádné `place=excl`, `os=`, `cl_*`,
`scratch_local`. Jakákoli GPU. To je taky důvod, proč se to naplánuje
v hodinách, ne ve dnech.

## 0a. SCHVALOVACÍ BRÁNA — platí bez výjimky

**Výpočty a trénink se spouštějí výhradně na výslovný příkaz uživatele, na
clusteru i lokálně.** Týká se to všeho, co spotřebovává zdroje nebo vytváří
stav na vzdáleném stroji: `qsub`, dlouhé skórovací joby na lokálním stroji
(i na pozadí), stahování dat na frontend, `rsync` na storage, rozbalování,
instalace do venv. Čtení (`qstat`, `ls`, `quota`, `md5sum`, krátký benchmark)
je bez schválení v pořádku.

Postup: připravit příkaz, ukázat ho i s tím, co udělá a co spotřebuje, počkat
na výslovné „ano". Schválení jednoho kroku **neplatí** pro další krok.

Důvod: sdílená infrastruktura, kde špatně poslané pole spálí node-hodiny
z grantu a zaneřádí storage; a v tomhle projektu už jednou platilo, že
neuvážený submit stál dvanáct dní čekání.

## 0. Předpoklad — přístup

```bash
ssh-add --apple-use-keychain ~/.ssh/metacentrum_ed25519 && ssh tarkil hostname
```

Bez toho nejde nic z níže uvedeného. Klíč má na MetaCentru omezenou platnost;
když vypršel, je potřeba nový přes jejich portál.

## 1. Kvóta a data

```bash
ssh tarkil 'quota -s; df -h /storage/brno12-cerit/home/hav0254 | tail -2'
```

Potřeba ~80 GB na zipy + rozbalení. Data stáhnout **přímo na frontendu**
(ušetří upload z Macu):

```bash
ssh tarkil 'cd /storage/brno12-cerit/home/hav0254/ttd-bench-data && \
  for f in Testing_Normal_Videos.zip Training-Normal-Videos-Part-1.zip Training-Normal-Videos-Part-2.zip; do \
    curl -L -C - --retry 5 -o "$f" "https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/$f"; done'
```

**MD5 ověřit proti lokálním kopiím** (Testing a Part-1 jsou na Macu staženy):

```bash
md5 -q ~/Datasets/ttd-bench/ucf-crime/Testing_Normal_Videos.zip
ssh tarkil 'md5sum /storage/brno12-cerit/home/hav0254/ttd-bench-data/Testing_Normal_Videos.zip'
```

Part-2 lokální kopii nemá — u něj se ověřuje jen velikost proti HF API
(34 019 840 040 B) a integrita přes `unzip -t`.

### Provedeno 2026-08-12 — kontrolní součty

Staženo na frontend (celkem 76,75 GB, ~46 min), ověřeno:

| soubor | bajtů | MD5 | shoda s Macem |
|---|---|---|---|
| Testing_Normal_Videos.zip | 4 661 975 118 | `a53568c0402351204469350f6f8abf08` | ano |
| Training-Normal-Videos-Part-1.zip | 38 072 348 701 | `62b500d04342013f9a0b82cd0b825045` | ano |
| Training-Normal-Videos-Part-2.zip | 34 019 840 040 | `6461f638756d882518ceb440d4a8c435` | (lokální kopie není) |

Všechny tři velikosti odpovídají HF API na bajt. **Fáze A pozastavena po A2
na pokyn uživatele**; rozbalení (A3) neproběhlo.

Repo na clusteru bylo pro A1 aktualizováno **inkrementálním git bundlem**
(`git bundle create ... 4bb0e38..master`, 1,7 MB), protože frontend nemá jak
se autentizovat vůči privátnímu GitHub repu přes HTTPS. Bundle je proti
kopírování jednotlivých souborů podstatně lepší: cluster má korektní commit
hash, který skórovací skript zapisuje do logu. Pull blokovaly čtyři untracked
soubory, které mezitím vznikly i v masteru (`frozen-header.jsonl`,
`frozen-libs.txt`, `setup_venv.sh`, `smoke_concurrent.sh`); byly odsunuty do
`cluster-untracked-bak/` a po pullu ověřeny jako **identické** s verzemi
v masteru.

## 2. Manifest a disjunktnost na clusteru

Rozbalit do `normal-testing/`, `normal-train-1/`, `normal-train-2/` a postavit
manifest **znovu na clusteru** (ne kopírovat z Macu — cesty i obsah se musí
ověřit tam, kde se skóruje):

```bash
ssh tarkil 'cd /storage/.../vetev-a && .venv/bin/python common/scripts/far_corpus.py'
```

Skript ověřuje disjunktnost od 34 událostních klipů třemi způsoby (složky,
basename, MD5) a končí `DISJOINT: YES/NO`. Při `NO` **zastavit**.
Vypíše stopáž po zdrojích a Garwoodovy meze.

## 3. Modely

Zkopírovat **4 weapon checkpointy `.pt`** (uzly nemají spolehlivý přístup ven):

```bash
rsync -av clanek-2-ttdbench/checkpoints/yolov{8s,8m,12m,26m}-weapon.pt \
    tarkil:/storage/.../vetev-a/clanek-2-ttdbench/checkpoints/
```

**Skóruje se `.pt` na CUDA cestou ultralytics**, tedy stejným backendem jako
replay `local-gpu` (u yolov8s ale ne se stejným imgsz, viz níže). Žádný ONNX — to je edge cesta a shoda opraveného ONNX
s `.pt` je doložená zvlášť (`results-far-calib/backend_check.csv`, D10).

**imgsz = 640 pro všechny čtyři modely, včetně yolov8s.** Záměr byl
replikovat **nasazovací** cestu, ne trénovací, a ten platí dál. Premisa ale
byla chybná (**oprava 2026-09-27, odchylka D11**, audit
`docs/IMGSZ-AUDIT-2026-09-27.md`): replay sice volá
`model.predict(frame, device=..., conf=...)` bez `imgsz`
(`common/src/ttdbench/inference.py`, `UltralyticsBackend.infer`), jenže
ultralytics pro `.pt` přebírá `imgsz` uložený v checkpointu
(`Model._reset_ckpt_args`) a default 640 použije jen tehdy, když checkpoint
žádný nenese. Ve fázi 2 tedy yolov8s (trénovaný na 416) běžel na **416**
v local-gpu a remote a na 640 jen v edge-sim (ONNX má pevný vstup 640);
v8m/v12m/v26m běžely všude na 640. Kalibrace na 640 proto u v8s odpovídá jen
edge-sim, ve 4 z 5 konfigurací se práh z 640 aplikuje na skóre z 416. V článku
je to uvedeno jako limitace (varianta c); přeskórování v8s na 416
(varianta a) čeká na rozhodnutí autora. Postup níže je popsán tak, jak
skutečně proběhl (640 pro všechny).

## 4. Pilot a walltime

Propustnost změřit na ~10 klipech, pak walltime nastavit s **dvojnásobnou
rezervou**:

```bash
ssh tarkil 'cd /storage/.../vetev-a && qsub -q gpu -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=01:00:00 \
  -v CORPUS_DIR=...,STRIDE=200,REPO_DIR=$PWD/clanek-2-ttdbench clanek-2-ttdbench/hpc/pbs_far_scoring.sh'
```

Referenční bod z Macu: 26,6 fps na model na MPS, tedy 4,5 h výpočtu na hodinu
stopáže. Na 1080 Ti čekej řádově rychlejší; skutečné číslo z pilotu zapiš sem.

## 5. Sanity check platforem (A4) — než se Mac výsledky zahodí

```bash
.venv/bin/python common/scripts/far_platform_check.py --traces <stažené traces> --n-clips 20
```

Referencí je `results-far-calib/calib_traces.parquet` — původní korpus 246 SCVD
klipů oskórovaný na Macu (MPS, imgsz 640 — stejně jako cluster; u yolov8s to ale
není imgsz replaye local-gpu/remote, viz D11). Kritérium **max |Δconf| < 0,01**
na model. Výsledek zapsat jednou větou do D9 dodatku. Při `PLATFORMS DISAGREE`
**zastavit a nemíchat**.

## 6. Ostrý běh

```bash
ssh tarkil 'cd /storage/.../vetev-a && qsub -q gpu -J 0-63 \
  -v CORPUS_DIR=...,STRIDE=64,REPO_DIR=$PWD/clanek-2-ttdbench clanek-2-ttdbench/hpc/pbs_far_scoring.sh'
```

Pole je **idempotentní**: každý klip má vlastní JSONL cache, hotový klip se
přeskočí, zabitý job stojí jeden klip. Stejné pole lze poslat znovu.

## 7. Odvoz a sloučení

```bash
rsync -av tarkil:/storage/.../far-corpus/traces/ ./far-traces/
.venv/bin/python common/scripts/far_traces_merge.py --traces ./far-traces
```

Merge **odmítne zapsat neúplný korpus** (chybějící klipy, méně než 4 modely na
klip, méně než 99 % očekávaných snímků) — korpus kratší o 3 % by posunul každou
míru za hodinu bez viditelného příznaku. Vypíše nové Garwoodovy meze.

## 8. Pak fáze C

Sweep znovu (varianta (a) primárně), přepočet KM / P(alarm ≤ 1 s) / Cox na
nových bodech do nové odvozené vrstvy, náhrada §4.7, přegenerování čísel
v úvodu, dodatek k D9, **freeze podle POST-DRAFT.md**.
