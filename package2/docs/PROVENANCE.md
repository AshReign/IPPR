# Provenance and publication versions

Release version: **IPPR_P2_Formal_EN_20261024_r1**.

## Distinct identities

| Identity | Meaning |
|---|---|
| P1 commit `704f724146792482359ebaa2235cde5c6a41b059` | Frozen data/manifests snapshot used for training and dev selection |
| P4 commit `1004e203b7f9e4e3b2f3dcf740cc4983369fde51` | Inspected Ngoc/P4 output protocol used for the accepted T0 selection path |
| P5 commit `2fa7e07ab42cceb49ac149a391c05df0dcc63a1a` | Fixed evaluator/encoder reference |
| Local P2 snapshot commit `c221d204dae9d24188c72dd0fa7bd8e8e7313c94` | Original frozen P2 source snapshot in a local repository with no team remote |
| New team GitHub publication commit | Commit on the separately published P2 branch; obtain it from the checkout or verified external publication receipt |
| `epoch5.pt` SHA256 | Identity of the unchanged selected weight bytes, independent of Git commits |

The new publication commit does not replace the original frozen snapshot identifier. It records English documentation, a portable handoff and any explicitly documented path-only wrappers. Core numerical behavior, frozen parameters, stored probabilities and SQLite bytes are unchanged. The vendored P5 file remains the pinned teammate implementation; publication does not modify the team's `eval/` code or silently follow a newer evaluator version.

[source_release_map.json](../configs/source_release_map.json) maps each copied source file to its original hash and separately identifies the path-only adaptations. It is a new publication provenance record. Documentation, portable configuration descriptors and release checksums have their own bytes and hashes; they are not presented as byte-identical frozen artifacts.

The commit that contains a file cannot reliably contain its own final hash inside that file. Therefore the verified team commit, branch URL, Drive object IDs, actual remote sizes and permission observations belong to the external `PUBLICATION_RECEIPT.json` produced after publication. In a checked-out team branch, use `git rev-parse HEAD` to identify that checkout. A source snapshot ID or an unverified planned branch name is not remote-publication evidence.

## Original immutable evidence

| Artifact | Original SHA256 |
|---|---|
| Selected `epoch5.pt` | `278ba795436c1299621b68a6f775a4af8b284afcab9c242384abff877a6906ca` |
| Sealed `freeze/frozen.json` | `6bddf2e07cae81ee2541c8c5bef8b2ad78dbbd698bd207f161f34eee6ae1aa7e` |
| Sealed final `DELIVERY_SHA256.json` | `a919649a76a8b4157b91213231c96fb4de962fcb590694a3c021fa6afd9f07f4` |
| Complete 47-cache verification | `644b8cf7aaea17ba0513c3e583d814ba88b6eb83c6c85f316c6724e01d5c52e5` |
| Supplemental optional-v4 qualification | `32604063458dc972691e973ca060a5027826bd02bd5adfbf4894c1a8f7c1ffb0` |
| P5 evaluator source | `b539ae46869407ffdfce9a8d4582d0a72bd722bdead816453d4cc706ab49f8c7` |
| Shared T0 adapter | `1a578f1dada74423bb053489a50991f16733a41eaea6903d3c31ac8d3ae1163a` |
| Original soft-mask producer | `9021ae722b5e33437e3571c679808b1aae8d2afd90a387f6bfb5ecc9339c101f` |
| Optional v4 restorer | `e0f8b2e7ef385266b3e9188ba031f3bb49e618051b39975c8223199183f9fa2b` |

The English report is a derived publication summary, not a replacement for these sealed bytes. Original Chinese reports, historical documents, old failures, temporary pilot material and all ten training checkpoints remain preserved locally. Large historical artifacts and intermediate diagnostics are not mixed into the required teammate cache downloads.

## Portable descriptors and original metadata

`configs/frozen_release.json` and `configs/final.yaml` are new portable descriptors with their own release hashes. Their source relationship is recorded explicitly; they must not be assigned the original freeze hash. Likewise, newly generated archive/member manifests protect this publication layout and do not overwrite the original sealed delivery manifest.

The SQLite metadata still contains original producer paths and signatures. Those strings are provenance, not hardcoded requirements for cache readers. Moving a file leaves its bytes and original hash unchanged; use the release inventory's relative location and do not rewrite the database to change its stored paths.

The source-to-release file map and portability checks are recorded with the publication audit. Files that remain byte-identical can be checked against their original hashes; a translated comment or relocated default path has a new source hash even if numerical behavior is unchanged. Do not claim byte identity for adapted files.

## Evidence index conventions

[evidence_index.json](../reports/evidence_index.json) uses logical `sealed_final/` and `sealed_workspace/` prefixes. They identify retained evidence roots without embedding one user's absolute filesystem paths. These references are evidence identifiers, not claims that all raw evidence is present in the essential ZIP. The publication receipt identifies any separate supplemental evidence package and its verification, if one is distributed.
