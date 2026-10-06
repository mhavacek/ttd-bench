# Cluster-side scenario configs (config_hash provenance)

These are the scenario YAMLs **as loaded on the MetaCentrum cluster**, i.e.
after the deployment-time path rewrite documented in the repo
(`../../docs/DEPLOY-RERUN.md` / `../../docs/METACENTRUM.md`):

```bash
sed -i 's|/Users/macbook/Datasets/ttd-bench|/storage/brno12-cerit/home/hav0254/ttd-bench-data|g' \
    configs/scenarios-*.yaml
```

They differ from `configs/frozen/` **only** in the dataset path prefix inside
`file:` fields. Nothing else was edited on the cluster.

Verification (performed 2026-08-04 with `ttdbench.config.ExperimentConfig`):

| experiment config | scenarios file | resulting config_hash | matches logs |
|---|---|---|---|
| experiment-phase1-final.yaml | scenarios-ucf-phase1.yaml (this dir) | `48c0b912a4f390ef` | ✔ all 225 Phase-1 logs |
| experiment-scvd.yaml | scenarios-scvd.yaml (this dir) | `20bdb7a3775a1c3d` | ✔ all 3400 Phase-2 logs |

To re-verify: place these two files (plus `network_profiles.yaml` and the
experiment YAML from `configs/frozen/`) in one `configs/` directory, load the
experiment config with `ExperimentConfig.load(...)`, and call
`.config_hash()`.

The storage path (including the cluster account name) is retained on purpose:
it is part of the hashed payload, so redacting it would make the logged
`config_hash` unverifiable.
