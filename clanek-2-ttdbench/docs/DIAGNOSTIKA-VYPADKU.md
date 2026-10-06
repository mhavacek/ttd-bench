# Diagnostika výpadků jobů — příčina 869 běhů s nulou snímků

**Datum: 2026-08-03.** Vstup: PBS výstupy stažené z MetaCentra do
`results/pbs-logs/` (`#PBS -j oe`, takže `.o` obsahuje i stderr) — 3400 souborů
job 22369876 (fáze 2), 225 job 22427987 (fáze 1 final), 135 job 22365639
(pilot fáze 1), plus `ttd-rerun.o22366667`. Analýza je čistě post-hoc nad
staženými logy a zmrazenými surovými logy; na klastru nic nespuštěno,
`results/` nezměněno.

Tento protokol je citovatelný z odchylky **D1** (`ANALYSIS-PLAN-PHASE2.md`)
a z odchylky **O2** (`PHASE1-FINAL.md` §8).

---

## 1. Závěr

Příčina je určená, jednoznačná a **reprodukovaná mimo klastr**.

Server `mediamtx` binduje vedle RTSP portu ještě dva pevné UDP porty
**:8000 (RTP)** a **:8001 (RTCP)**. Šablona `MEDIAMTX_CONFIG`
(`common/src/ttdbench/replay.py:37`) parametrizuje pouze `rtspAddress`, takže
UDP porty zůstávají na defaultu. Když na jednom uzlu běží dva joby současně,
první zabere :8000 a druhý ho už nezíská. Retry smyčka
(`replay.py:_try_start`) losuje při opakování **jiný TCP port**, což je proti
téhle kolizi bezmocné — všechny tři pokusy padnou na tomtéž :8000. Job umře
v `runner.py:199` na `mtx.start()`, tedy **dřív, než se dekóduje první snímek**.

Spouštěčem není konfigurace experimentu, ale okamžik, kdy PBS začal na uzel
pouštět **dva souběžné joby místo jednoho**.

Kategorizace všech 3760 PBS výstupů:

| běh | OK | `bind: address already in use` | jiné |
|---|---|---|---|
| fáze 2 (22369876) | 2625 (77,2 %) | **775 (22,8 %)** | 0 |
| fáze 1 final (22427987) | 131 (58,2 %) | **94 (41,8 %)** | 0 |
| pilot fáze 1 (22365639) | 124 (91,9 %) | 0 | 11 (starší režim: 10× timeout portu, 1× pád při startu) |

**100 % běhů s nulou snímků v obou hlavních bězích má tuhle jedinou chybu.**
Žádný walltime kill, žádný OOM, žádný plný scratch, žádné selhání alokace GPU,
žádný chybějící soubor datasetu. Pilot má jiný, starší režim selhání — je z
doby před zpevněním startu mediamtx (`5e53e25`); to zpevnění timeouty
odstranilo, ale zavedlo právě tu neúčinnou retry smyčku.

### Reprodukce (lokálně, mediamtx v1.9.3, tatáž binárka)

Dvě současné instance se současnou šablonou:

```
instance 1: běží, žádná chyba
instance 2: ERR listen udp :8000: bind: address already in use
```

Řetězec chyby je znak po znaku totožný s tím v 869 logech z klastru. Co
binárka reálně otevře:

| šablona | otevřené porty |
|---|---|
| současná | TCP :\<random\> + **UDP :8000** + **UDP :8001** |
| `protocols: [tcp]` | TCP :\<random\> — a nic jiného |
| parametrizované `rtpAddress`/`rtcpAddress` | TCP :\<random\> + UDP :\<zadané dva\> |

---

## 2. Spouštěč: přechod uzlu z 1 na 2 souběžné joby

Každý uzel vykazuje **ostrý přechod**: dlouhý úsek bez jediného selhání, pak
skok na stabilních 41–49 % a už nikdy návrat. Současně se přesně
**zdvojnásobí rychlost startů** — to je nezávislý ukazatel počtu slotů.

| uzel | před přechodem | po přechodu | start-rate před → po |
|---|---|---|---|
| konos1 | 0/354 | 40/82 = 49 % | 11,6 → 19,8 /h |
| konos2 | 0/247 | 111/260 = 43 % | 12,3 → 19,5 /h |
| konos3 | 0/1 | 262/646 = 41 % | — → 22,7 /h |
| konos4 | 0/168 | 194/438 = 44 % | 12,0 → 21,2 /h |
| konos5 | 0/200 | 168/375 = 45 % | 12,9 → 19,6 /h |
| **konos6** | **0/240** | přechod nenastal | 11,1 /h |
| **konos7** | **0/389** | přechod nenastal | 11,2 /h |

konos6 a konos7 se nikdy nezdvojnásobily a mají **nula selhání z 629 běhů**.
To je nejsilnější jednotlivý důkaz: uzel, na kterém neběží dva joby najednou,
neselhává vůbec.

Souhrnně: fáze 2 **0/1597** selhání v 1-slot režimu vs. 775/1803 v 2-slot;
fáze 1 final **0/32** vs. 94/193.

---

## 3. Gradient podle konfigurace je záměna s časem, ne mechanismus

Pracovní hypotéza „čím víc subprocesů, tím víc selhání" **neplatí**.
Konfigurace je v enumeraci buněk nejpomaleji se měnící faktor, takže tvoří
souvislé bloky po 680 indexech, a pole je zpracovává v pořadí:

| indexy | konfigurace | běželo | podíl běhů v 2-slot režimu |
|---|---|---|---|
| 0–679 | local-gpu | 22. 7. 12:18 → 23:32 | **18,2 %** |
| 680–1359 | edge-sim | 22. 7. 23:32 → 23. 7. 04:02 | 35,0 % |
| 1360–2039 | remote-lan | 23. 7. 04:02 → 10:38 | 61,6 % |
| 2040–2719 | remote-wifi | 23. 7. 10:38 → 17:25 | 70,4 % |
| 2720–3399 | remote-cellular-4g | 23. 7. 17:25 → 23:15 | **79,9 %** |

local-gpu běželo první, dokud byly uzly jednoslotové; 4G běželo poslední, kdy
už byly skoro všechny dvouslotové. Po rozdělení podle režimu gradient mizí:

| konfigurace | 1-slot režim | 2-slot režim |
|---|---|---|
| local-gpu | **0/556 = 0 %** | 49/124 = 39,5 % |
| edge-sim | **0/442 = 0 %** | 104/238 = 43,7 % |
| remote-wifi | **0/201 = 0 %** | 169/479 = 35,3 % |
| remote-lan | **0/261 = 0 %** | 192/419 = 45,8 % |
| remote-cellular-4g | **0/137 = 0 %** | 261/543 = 48,1 % |

V 2-slot režimu už není pořadí monotónní (remote-wifi 35,3 % je *pod*
local-gpu 39,5 %). Testy:

| | efekt konfigurace na všech bězích | uvnitř 2-slot režimu |
|---|---|---|
| fáze 1 final | chi² = 18,89, p = 0,00083 | chi² = 0,31, **p = 0,989** |
| fáze 2 | chi² = 222,6, p = 5,2e-47 | chi² = 19,35, p = 0,00067 |

Fáze 1 je tu čistší experiment (193 z 225 běhů v 2-slot režimu, takže málo
prostoru pro záměnu) a efekt konfigurace v ní **mizí beze zbytku**.
Ve fázi 2 klesne chí-kvadrát o 91 %, ale malý reziduál zůstává — viz §5.

### Vedlejší výsledek: obě nevysvětlené anomálie D1 se rozpouštějí

D1 reportovala jako limitaci dva signifikantní výsledky. Po podmínění režimem
oba mizí:

| test | všechny běhy | uvnitř 2-slot režimu |
|---|---|---|
| selhání vs. scénář (globálně) | p = 0,157 | **p = 0,995** |
| selhání vs. model (globálně) | p = 0,0074 | **p = 0,627** |
| selhání vs. scénář *uvnitř* remote-lan | p = 1,3e-8 | **p = 0,963** |

Byly to artefakty časového uspořádání pole, ne závislost na obsahu. Tuhle
větu lze v D1 nahradit.

---

## 4. Selhání je loterie, ne vlastnost buňky

V pilotu selhalo 11 buněk `[1, 9, 12, 38, 39, 64, 74, 75, 104, 128, 134]`.
Přesně tyto buňky byly přespuštěny **sekvenčně v jediném jobu**
(`ttd-rerun.o22366667`) a **všech 11 doběhlo** (305, 304, … zpracovaných
snímků, nula chyb mediamtx). Sekvenční běh znamená jednu instanci mediamtx
v čase, tedy žádnou kolizi.

Selhání tedy **není deterministické**. Neexistuje „prokletá buňka"; rozhoduje
jen to, jestli job prohrál závod o :8000. Rerun proto ztracené buňky vrátí.

---

## 5. Co z logů určit NELZE

Tohle je poctivá hranice analýzy, ne formalita:

1. **Reziduální efekt konfigurace ve fázi 2** (p = 0,00067 uvnitř 2-slot
   režimu) **nevysvětlen**. Nabízejí se dvě vysvětlení a logy mezi nimi
   nerozhodnou: (a) skutečný efekt délky běhu — vítěz drží :8000 po dobu
   svého běhu, takže delší konfigurace vyrobí víc poražených; (b) chyba
   měření — indikátor „2-slot" je proxy definovaná jako „od prvního selhání
   na uzlu dál", a uzel mohl mezi jedním a dvěma joby oscilovat, takže část
   1-slot běhů je klasifikovaná jako 2-slot. Rozhodlo by to jen `qstat`/
   účtovací záznamy o skutečné souběžnosti na uzlu, které nemáme.
   Ve fázi 1, kde je proxy skoro bezchybná, reziduál není žádný (p = 0,989).
2. **Proč se uzly přepnuly do dvouslotového režimu** právě tehdy. To je
   rozhodnutí plánovače (nejspíš uvolnění druhé GPU na uzlu), a v našich
   logách po něm není stopa.
3. **Proč konos6 a konos7 nikdy nepřešly.** Stejný důvod — vlastnost
   plánovače a obsazenosti, ne našeho kódu.

Nic z toho nemění příčinu selhání, která je určená a reprodukovaná. Týká se
to jen jemné struktury *míry* selhání.

---

## 6. Je vyloučení obhajitelné mechanisticky?

**Ano.** Formulace pro sekci limitací, konzervativně:

> Vyřazené běhy selhaly při startu RTSP serveru, tedy dřív, než se dekódoval
> první snímek, načetl model nebo vyhodnotila jakákoli detekce. Příčinou byla
> kolize na pevném UDP portu, který server otevírá při startu, v okamžiku, kdy
> plánovač umístil na uzel druhou souběžnou úlohu; mechanismus byl
> reprodukován mimo klastr. Selhání tedy nemůže záviset na obsahu scény, na
> modelu ani na měřené veličině, protože v okamžiku pádu ještě žádná z těchto
> veličin nebyla načtena. Po podmínění režimem souběžnosti je selhání
> nezávislé na scénáři (p = 0,995), na modelu (p = 0,627) i — ve fázi 1 — na
> konfiguraci (p = 0,989). Chybějící data proto nejsou chybějící zcela náhodně
> (MCAR): pravděpodobnost selhání závisí na tom, kdy buňka běžela, a pořadí
> běhu je vázané na konfiguraci. Jsou ale chybějící náhodně podmíněně
> (MAR) vůči konfiguraci, a protože všechny reportované míry se počítají
> uvnitř konfigurace a mechanismus selhání předchází jakémukoli měření,
> vyloučení nezavádí zkreslení odhadů uvnitř konfigurace. Cenou je nižší a
> mezi konfiguracemi nevyrovnaná přesnost, nikoli posun bodových odhadů.

Klíčové rozlišení, které tahle diagnóza umožnila: dřív se dalo tvrdit jen
„selhání koreluje s konfigurací, ale ne s obsahem" (empirické pozorování).
Teď se dá tvrdit **proč** — selhání nastává před jakýmkoli kontaktem s daty,
takže nezávislost na obsahu není šťastná náhoda, ale nutnost plynoucí
z mechanismu.

---

## 7. Oprava a její ověření

### Změna

V `common/src/ttdbench/replay.py`, `MEDIAMTX_CONFIG`, přidat jediný řádek:

```yaml
protocols: [tcp]
```

**Je měřicí-neutrální.** Obě strany řetězce už TCP vynucují:
`common/src/ttdbench/ingest.py:65` (`rtsp_transport;tcp`) a
`common/src/ttdbench/replay.py:213` (`-rtsp_transport tcp`). Porty
UDP :8000/:8001 tedy nikdy nepřenesly ani bajt — jsou to mrtvé porty, které
shodily 869 běhů. Jejich odstranění nemění transport, latenci ani ztrátovost,
takže přeběhlé buňky zůstávají srovnatelné se zmrazenými.

Obranu do hloubky lze přidat parametrizací `rtpAddress`/`rtcpAddress` na
per-cell porty (ověřeno, funguje), ale při `protocols: [tcp]` už je zbytečná.

Retry smyčku není nutné měnit; po opravě ztrácí smysl kvůli téhle třídě chyb,
ale proti squatterovi na TCP portu dál pomáhá.

### Jak to ověřit dřív, než se spálí node-hodiny

1. **Lokálně, zdarma, hotovo.** Dvě současné instance mediamtx: se současnou
   šablonou druhá padne, s `protocols: [tcp]` běží obě. Tenhle test
   reprodukuje přesně tu chybu z klastru.
2. **Na klastru, minuty, jeden uzel.** Pustit **dvě buňky současně na jednom
   uzlu** — to je přesně ta podmínka, která selhání vyrábí. Bez opravy zhruba
   jedna ze dvou padne; s opravou musí projít obě. Menší test než tenhle
   nemá cenu, protože jednoslotový běh selhání nikdy nevyvolá — to je právě
   důvod, proč 1597 běhů fáze 2 proběhlo bez jediné chyby.
3. Teprve pak pustit plný rerun.

Kontrola po opravném běhu: `grep -l "address already in use" *.o | wc -l`
musí být 0.

---

## 8. Je rerun vědecky nutný?

**Fáze 1 — ne, je kosmetický.** Po vyloučení má všech 45 buněk aspoň 2
platná opakování, žádná buňka není prázdná, a relativní půlšířka 95% CI je
0,6–7,9 % u `pf_dt_infer_ms` a 0,5–5,1 % u `pf_latency_ms` (viz
`PHASE1-FINAL.md` §8/O3). Tvrzení fáze 1 je dekompozice latence, tedy timing
s malým rozptylem; pět opakování z §3 bylo pro tuhle třídu tvrzení
předimenzované. Rerun by zpřesnil čísla, která jsou už teď těsná.

**Fáze 2 — spíš ano, ale ne kvůli hlavnímu výsledku.** Primární analýza (D2,
survival s cenzurou) stojí na 2625 platných bězích a průnik scénářů
nepotřebuje. Rerun by přinesl:

- vyrovnané *n* mezi konfiguracemi (teď 4G ztratilo 38 %, local 7 %), tedy
  vyrovnanou přesnost místo nevyrovnané;
- víc síly pro sekundární párové testy D4, kde je průnik scénářů detekovaných
  ve všech pěti konfiguracích jen 1–15 z 34;
- odpadnutí celé odchylky D1 z metodiky, což je pro recenzenta jednodušší
  příběh než obhajoba vyloučení 22,8 % dat.

Odhad ceny: v dvouslotovém režimu uzel startuje ~20 běhů/h, z nichž teď ~45 %
padá; po opravě by měly projít prakticky všechny. 775 buněk tedy vychází
řádově na ~40 uzlo-hodin, rozloženo přes dostupné uzly na jednotky hodin.
To je zlomek rozpočtu kampaně a nezávisí to na Karolině — oprava funguje
i na MetaCentru, protože příčina nikdy nebyla v hardwaru.

**Doporučení:** opravit šablonu, ověřit dvěma souběžnými buňkami na jednom
uzlu, přeběhnout 775 buněk fáze 2, fázi 1 nechat být a reportovat z těch 131
platných běhů s odchylkami O1–O4.

---

## 9. Reprodukce téhle analýzy

Kategorizace a všechny testy běží nad `results/pbs-logs/` a
`results/*/raw/*.jsonl`. Klíčové kroky:

- kategorie chyb: `grep -oE "(RuntimeError|TimeoutError): .*" results/pbs-logs/ttd-bench.o*`
- režim souběžnosti: indikátor „od prvního selhání na uzlu dál" nad
  `wallclock_utc` z hlavičky surového logu (hlavička se zapisuje
  v `runner.py:133`, tedy **před** `mtx.start()` na řádku 199)
- start-rate: počet běhů na uzel dělený rozpětím časů startu, zvlášť před a
  po přechodu
- lokální reprodukce: dvě instance `clanek-2-ttdbench/third_party/mediamtx`
  s toutéž šablonou
