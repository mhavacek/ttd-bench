# Předregistrace opravného běhu fáze 2

**Datum: 2026-08-03. Psáno PŘED spuštěním jakéhokoli opravného běhu.**
Kritéria níže jsou závazná a po zhlédnutí výsledků se nemění. Stejný režim
jako `ANALYSIS-PLAN-PHASE2.md`.

Podklad: `DIAGNOSTIKA-VYPADKU.md` (příčina 869 výpadků).

---

## 1. Co se přebíhá a proč

775 buněk fáze 2, které ve zmrazeném běhu zpracovaly nula snímků, plus
**50 kontrolních buněk**, které ve zmrazeném běhu doběhly.

Kontrolní množina byla pro tuhle předregistraci **přegenerována**. Původních
50 buněk mělo 18 takových, které ve zmrazeném běhu běžely v 2-slot režimu —
ty by do testu ekvivalence vnesly kontenci (§3) místo čisté změny kódu. Nová
množina (`common/scripts/select_control_cells.py`, deterministicky, bez RNG)
drží dvě podmínky: buňka ve zmrazeném běhu **uspěla** a běžela **v 1-slot
režimu**, tedy bez jakéhokoli souseda. Výsledek: 10 na konfiguraci, celkem 50,
nulový překryv s `cells_failed.txt`, všechny ověřeně způsobilé.

Cílem není víc dat. Cílem je **odstranit odchylku D1 z metodiky**: místo
obhajoby vyloučení 22,8 % běhů mít kompletní matici.

## 2. Co se změnilo v kódu a co ne

Jediná změna: `protocols: [tcp]` v `MEDIAMTX_CONFIG`
(`common/src/ttdbench/replay.py`). Odstraňuje binding UDP :8000/:8001, které
nikdy nepřenesly ani bajt, protože obě strany řetězce už TCP vynucují
(`ingest.py` `OPENCV_FFMPEG_CAPTURE_OPTIONS`, `RTSPReplay` `-rtsp_transport`).

**`config_hash` se nemění a zůstává `20bdb7a3775a1c3d`** — změna je ve zdrojáku,
ne v experiment configu. Přeběhlé buňky tedy sedí se zmrazenými na configu.

**`git_hash` se mění.** Sloučená data budou mít jeden config hash a dva git
hashe, což porušuje kritérium §5 („jeden git hash"). Právě proto existuje
kontrolní množina — viz §4. Obojí patří do reprodukční sekce článku.

## 3. Souběžnost na uzlu se MĚŘÍ, neplánuje

V `runner.py` se model nahrává na GPU (`build_backend`, ř. 184) **před**
`mtx.start()` (ř. 199). Padající joby tedy stihly obsadit GPU, než umřely.
Ve zmrazených datech se to dá změřit: porovnání úspěšných běhů v 1-slot vs.
2-slot režimu uvnitř týchž skupin (cfg × model × scénář) dává

| metrika | kde signifikantní | posun | podíl mediánu |
|---|---|---|---|
| `pf_latency_ms` | local-gpu (p = 0,00065) | +0,85 ms | +1,2 % |
| `pf_latency_ms` | 4G (p = 0,0040) | +2,52 ms | +1,6 % |
| `pf_dt_infer_ms` | 4G (p = 0,0024) | +0,45 ms | +2,1 % |
| `drop_rate` | local / wifi / 4G (p < 0,01) | +0,004 až +0,005 abs. | — |

Posuny jsou konzistentně směrem „pomalejší a víc ztrát", ale malé — řádově pod
rozptylem mezi opakováními (§4) a o řád menší než efekty, které článek
reportuje. **Zmrazená data tím nejsou materiálně kontaminovaná**; patří to do
limitací jako malá systematická složka, ne jako hrozba.

### Zmrazený běh nebyl exkluzivní

Dřívější formulace „zmrazený běh byl fakticky exkluzivní" **byla chybná**.
Z 2625 platných běhů jich **1597 (60,8 %) proběhlo v 1-slot režimu a 1028
(39,2 %) v 2-slot**, tedy zhruba 61 : 39. Zmrazená data jsou v tomhle ohledu
směs a není co „srovnávat zpět".

### Proto se souběžnost loguje

Od commitu s touhle předregistrací zapisuje každý běh do surového logu
`gpu_compute_apps` (výstup `nvidia-smi --query-compute-apps=pid,used_memory`)
**v hlavičce i v `end` záznamu** (`common/src/ttdbench/logio.py`). `None`
znamená „nvidia-smi není" a je odlišené od `[]` = „GPU je, nikdo na ní není".
Dvojice snímků navíc odhalí souseda, který přišel nebo odešel během běhu.

Tím se z předpokladu stává **per-run kovariáta** a fronta přestává být
blokátor: když exkluzivitu dostaneme, log to potvrdí; když ne, stratifikuje se
v analýze. Retrospektivně to samozřejmě nejde — zmrazená data tuhle kovariátu
nemají a jejich režim se odvozuje jen z proxy „od prvního selhání na uzlu dál".

### Plán B je STAŽEN — změřeno 2026-08-03

Původně tu stály dvě varianty: plán A s `place=excl` a plán B bez ní se
stratifikací podle logované kovariáty. **Plán B se ruší.**

Ověřovací běh (`hpc/smoke_concurrent.sh`, dvě buňky souběžně na jednom uzlu)
přeběhl náhodou tytéž dvě buňky, které existují i ve zmrazeném běhu —
`local-gpu__none__yolov8s-weapon__scvd-w012__rep0/rep1`. Kovariáta potvrdila,
že se procesy navzájem viděly (oba PID v `end` záznamu, 316 MiB každý). Rozdíl
proti zmrazeným hodnotám:

| metrika | zmrazený | se živým sousedem | v jednotkách SD |
|---|---|---|---|
| `pf_latency_ms` rep0 | 62,72 | 70,97 | **+5,1** |
| `pf_latency_ms` rep1 | 61,06 | 69,50 | **+5,2** |
| `drop_rate` rep0 | 0,030 | 0,120 | **+11,2** |
| `drop_rate` rep1 | 0,030 | 0,130 | **+12,5** |
| `pf_dt_infer_ms` | 12,43 / 11,10 | 13,76 / 14,70 | +1,4 / +3,8 |
| zpracované snímky | 131 / 131 | 119 / 118 | −12 / −13 |

Mez ekvivalence je ±1 SD. Neexkluzivní běh je od zmrazeného vzdálený o pět až
dvanáct SD, tedy **není s ním srovnatelný v žádné z primárních metrik**.
Stratifikace by nepomohla: nešlo by o šum, který se dá odfiltrovat, ale o jiný
provozní režim. Pro srovnání, efekt **mrtvého** souseda odhadnutý ze zmrazených
dat byl +0,85 ms u local-gpu — živý soused je **desetinásobek**.

Dvě buňky nejsou test a §4 se tím nemění. Na rozhodnutí o návrhu běhu to ale
stačí bohatě: rozdíl je řádový, ne hraniční.

### Co platí místo toho

- **Opravný běh vyžaduje `-l place=excl`.** Není to preference, je to podmínka
  srovnatelnosti.
- **Lhůta: 5 dní od odeslání.** Pokud se do té doby pole nerozjede, běh se
  zruší (`qdel`) a rerun se **nedělá**. Lhůta je zapsaná předem právě proto,
  aby se nečekalo znovu dvanáct dní na něco, co se nemusí naplánovat nikdy.
  Datum odeslání a datum rozhodnutí patří do labbooku.
- **Když se exkluzivita nedá naplánovat, rerun se nedělá.** To je legitimní
  konec, ne prohra: podle `DIAGNOSTIKA-VYPADKU.md` §6 je vyloučení D1
  obhajitelné mechanisticky a primární výsledek stojí na 2625 platných bězích.
  Rerun byl vždycky „vyplatí se, není nutný" (§8) — a nesrovnatelná data jsou
  horší než žádná.
- Neověřená možnost, kterou lze u plánovače zkusit, pokud `place=excl` stojí:
  zabrat obě GPU uzlu (`ngpus=2`). Chránilo by to před GPU kontencí a mohlo by
  se plánovat snáz, protože to nevylučuje CPU úlohy. **Nechrání to ale edge-sim**,
  který počítá na CPU. Před použitím by se muselo ověřit toutéž metodou jako
  výše, tedy přeběhnutím buněk, pro které máme zmrazenou hodnotu.

Oprava `protocols: [tcp]` je na exkluzivitě nezávislá a režim selhání odstraňuje
sama o sobě (ověřeno na klastru: dvě souběžné buňky, nula kolizí). Sekvenční
chunky navíc znemožňují kolizi jobu se sebou samým. Exkluzivita tak neřeší
výpadky, ale **výhradně** srovnatelnost měření.

### Vedlejší výsledek do článku

Číslo výše je samo o sobě metodický nález: sdílení uzlu mezi dvěma úlohami
zvedne u tohoto potrubí střední latenci o ~13 % a **ztrátovost čtyřnásobně**
(3 % → 12–13 %). Pro latenční benchmark na sdílené infrastruktuře to znamená,
že exkluzivní umístění není hygiena, ale předpoklad platnosti měření. Patří to
do diskuse i k limitacím zmrazeného běhu, kde 39 % platných buněk mělo souseda
— byť jen krátce žijícího, s efektem o řád menším (§3 výše).

## 4. Kritérium srovnatelnosti — pevné, spočítané předem

### Princip

Dva běhy téže buňky pod **identickým** kódem se liší o přirozený šum
běh-od-běhu. Ten je ve zmrazených datech změřený jako sdružená směrodatná
odchylka mezi 5 opakováními uvnitř skupiny (cfg × model × scénář):

| konfigurace | `pf_latency_ms` | `pf_dt_infer_ms` | `drop_rate` (abs.) |
|---|---|---|---|
| local-gpu | 1,633 | 0,947 | 0,008 |
| edge-sim | 10,517 | 10,276 | 0,003 |
| remote-lan | 1,170 | 0,624 | 0,019 |
| remote-wifi | 1,544 | 0,806 | 0,022 |
| remote-cellular-4g | 7,187 | 1,171 | 0,015 |

Relativně to je 1,7–4,6 % mediánu u `pf_latency_ms` a 1,8–5,5 % u
`pf_dt_infer_ms`. **Tyto hodnoty jsou od teď zafixované jako mez ekvivalence.**

### Filtr souběžnosti (platí pro plán A i B)

Zmrazená strana porovnání je z definice bez souseda (§1). Aby test měřil jen
změnu kódu, musí být bez souseda i strana opravená. Do primárního testu proto
vstupují **jen ty kontrolní buňky, u nichž `gpu_compute_apps` v hlavičce i
v `end` záznamu neobsahuje žádný cizí PID** (tj. jen vlastní proces běhu).

Pod plánem A by měly projít prakticky všechny; pod plánem B jen část.
**Když projde méně než 30 z 50**, prohlásí se test za nedostatečně silný,
opravené běhy zůstávají oddělenou vrstvou a D1 v metodice zůstává. Toto číslo
je zafixované předem, aby se v plánu B nedalo dojít k závěru „vyšlo to" na
hrsti buněk.

Buňky vyřazené tímto filtrem se reportují zvlášť a slouží k odhadu efektu
živého souseda — což je veličina, kterou dnes nikdo nemá.

### Test

Pro každou kvalifikovanou kontrolní buňku se vezme rozdíl
`d = opravený − zmrazený` a standardizuje se odchylkou své konfigurace
z tabulky výše: `z = d / SD_cfg`.

Primární metriky: **`pf_latency_ms`** a **`pf_dt_infer_ms`**.
Test: **TOST** (dva jednostranné t-testy) na `z`, mez ekvivalence **±1 SD**,
α = 0,05, oboustranně.

Sekundárně, deskriptivně: `drop_rate` v absolutních jednotkách proti mezi
z tabulky; shoda `meas_hit` (McNemar). Obojí se **nereportuje jako test** —
při n = 50 na to není síla a předstírat ji by bylo nepoctivé.

`ttd_meas_ms` se jako kritérium **nepoužívá**. Jeho rozptyl mezi opakováními
je 3,3 % u local-gpu, ale 33,5 % u wifi, 38,1 % u 4G a 131,5 % u edge-sim —
při n = 50 by test nic nerozlišil. Reportuje se jen popisně.

### Rozhodovací pravidlo

- **TOST projde u OBOU primárních metrik** → opravené běhy se slučují se
  zmrazenými do jedné množiny. Odchylka D1 z metodiky odpadá. V reprodukční
  sekci se uvedou oba git hashe a odkáže se na tento dokument.
- **TOST neprojde u kterékoli z nich** → opravené běhy se reportují jako
  **oddělená vrstva**. Primární analýza zůstává na 2625 zmrazených platných
  bězích, opravené slouží jako citlivostní analýza. D1 zůstává v metodice.

Pravidlo se neměkčí. Pokud test neprojde, není to důvod hledat jinou mez —
je to výsledek a znamená, že se něco jiného než transport změnilo.

### Když opravný běh sám selže

Pokud v opravném běhu vznikne jakýkoli běh s nula snímky, **zastavit a
diagnostikovat**, ne dopočítávat. Kontrola po doběhnutí:

```
grep -l "address already in use" *.o | wc -l    # musí být 0
```

## 5. Co je od teď zafixované

Mez ekvivalence (tabulka v §4), obě primární metriky, α = 0,05, rozhodovací
pravidlo, filtr souběžnosti včetně prahu 30 z 50, složení kontrolní množiny,
a to, že kontrolní buňky běží **v témže poli jako těch 775**, aby sdílely
podmínky.

Nezafixované a volné: pořadí buněk v poli, počet uzlů, velikost chunku,
walltime, a volba mezi plánem A a B (ta se jen zapíše s datem).

## 6. VERDIKT: rerun se nekoná — rozhodnuto 2026-08-05, zapsáno 2026-08-10

Rozhodnutí padlo na základě dvou měření; jejich vyhodnocení je persistováno
v `results/rerun-eval/` (skript `common/scripts/rerun_eval.py`, reprodukuje
čísla níže i čísla z commitu sondy):

- **Sonda**: 50 kontrolních buněk (10 na konfiguraci), opravený kód
  (`protocols: [tcp]`) + znovupostavené prostředí, chunky po 10 buňkách.
  Nula kolizí, nula selhání — oprava výpadků funguje. Měřicí srovnání
  (párově proti zmrazeným hodnotám, v jednotkách předregistrované SD §4):
  local-gpu −0,80 / lan −0,45 / wifi −0,19 / 4G −0,18, ale
  **edge-sim +3,41 SD** (`pf_latency_ms`; `pf_dt_infer_ms` obdobně +3,50),
  pomalejší ve všech 10 bězích.
- **Edgetest**: týchž 10 edge-sim buněk (4 modely), každá ve vlastním jobu
  (mezery jako ve zmrazeném běhu) — izoluje hypotézu chunkování.
  Výsledek: **+1,62 SD latence / +1,57 SD inference**. Chunkování tedy
  vysvětluje **52–55 % odchylky**; zbylých ~1,6 SD (pomalejší v 8/10 buněk)
  zůstává **nevysvětleno** (kandidáti: znovupostavený venv, CPU kontence,
  kterou GPU kovariáta nevidí — nerozlišitelné bez dalších běhů).

**Formální TOST podle §4 (mez ±1 SD, α = 0,05, per konfigurace).** Rozlišuj
„ekvivalence neprokázána" od „prokázán rozdíl": ekvivalence se **prokázala
u obou** primárních metrik jen u remote-wifi, u lan a 4G u jedné ze dvou a
u local-gpu u žádné — jenže bodové odhady local-gpu (−0,80 a −0,94 SD) leží
**uvnitř** meze, takže jde o nedostatek přesnosti při n = 10, ne o doklad
rozdílu. **Prokázaný rozdíl je jen u edge-simu** (bodový odhad mimo mez). Podle zafixovaného rozhodovacího pravidla by se opravené
běhy nikdy nesměly sloučit se zmrazenými → opravný běh by za ~40 node-hodin
koupil jen citlivostní vrstvu, kterou sonda + edgetest už poskytují v malém.
Vyloučení D1 je mechanisticky obhájené (`DIAGNOSTIKA-VYPADKU.md` §6) a
primární analýza stojí na 2625 platných bězích. **Rerun 775 buněk se proto
nekoná a žádná další session tohle rozhodnutí znovu neotevírá.** Pětidenní
lhůta z §3 je bezpředmětná — pole nebylo nikdy odesláno.

**Metodický nález, který §4 nepředvídal: sdružený TOST je slepý vůči
systematické odchylce jedné konfigurace.** Na sondě sdružený test přes
všech 50 buněk PROJDE (mean_z +0,36, p = 0,0085), přestože edge-sim leží
+3,4 SD od zmrazeného běhu — čtyři mírně záporné konfigurace ho vyruší.
V nejlepším dostupném složení (sonda bez edge-simu + edgetest) vyjde
sdružený průměr přesně **+0,00 SD** při edge-simu stále +1,6 SD mimo.
Kdyby §4 předepsal jen sdružený test, verdikt by byl „ekvivalentní" a byl
by chybný. Testovat se musí po konfiguracích; do článku to patří jako
doporučení pro každé srovnávání běhových kampaní na sdílené infrastruktuře.

Vedlejší kvantitativní nález (do Discussion): struktura dávkování jobů
měřitelně ovlivňuje CPU-vázaná latenční měření — deset CPU-bound buněk
back-to-back v jednom jobu zvedne latenci edge-simu o dalších ~1,8 SD
proti běhu jedna buňka na job (celková odchylka chunkované sondy proti
zmrazenému běhu: medián +8,1 %, z toho chunkování ~polovina).
