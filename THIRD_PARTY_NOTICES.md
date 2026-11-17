# Third-party notices

LOCI's own code is MIT licensed. It uses the components below. Nothing in this list is modified.

## Bundled into the viewer or used at run time

| Component | Version | Licence | Source |
|---|---|---|---|
| three.js, including its post-processing passes (EffectComposer, UnrealBloomPass, OutputPass) | 0.186 | MIT | https://github.com/mrdoob/three.js |
| MediaPipe Tasks Vision (JavaScript and WASM runtime) | 0.10.35 | Apache-2.0 | https://www.npmjs.com/package/@mediapipe/tasks-vision |
| Model Context Protocol TypeScript SDK (`@modelcontextprotocol/server`) | 2.4 | Apache-2.0 | https://github.com/modelcontextprotocol/typescript-sdk |
| Zod | 4 | MIT | https://github.com/colinhacks/zod |
| Space Grotesk (variable font, via `@fontsource-variable/space-grotesk`) | 5.3.0 | OFL-1.1 | https://github.com/floriankarsten/space-grotesk |
| JetBrains Mono (variable font, via `@fontsource-variable/jetbrains-mono`) | 5.3.0 | OFL-1.1 | https://github.com/JetBrains/JetBrainsMono |
| SQLite (through Node's built-in `node:sqlite`) | bundled with Node | Public domain | https://www.sqlite.org |

The fonts are bundled with the viewer and served from its own origin; nothing is fetched from a font service. The public demo build includes three.js and the fonts only: no MediaPipe code, no WebAssembly and no model.

## Not redistributed: fetched or supplied by you

These are not committed to this repository.

**MediaPipe WASM runtime.** `npm run assets` copies it from the installed npm package into `web/public/mediapipe/wasm` so the viewer serves it from its own origin.

**MediaPipe gesture recogniser model** (`gesture_recognizer.task`, float16, version 1).

- Source: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
- SHA-256: `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`
- `npm run assets` copies it from `LOCI_GESTURE_MODEL` (or a sibling SENTINEL checkout) only when the checksum matches.
- Licence: published by Google as part of MediaPipe. The licence terms for this model file have not been confirmed for this project, which is why it is not redistributed here. Confirm them against the model card before shipping the model with any public build.

**nomic-embed-text** (embedding model, run locally through Ollama).

- Source: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 and https://ollama.com/library/nomic-embed-text
- Licence: Apache-2.0
- Pulled by you with `ollama pull nomic-embed-text`.

## Methods

The algorithms are implemented from their published descriptions, with no third-party code:

- HNSW: Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", 2016.
- BM25: Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009, using the constants and IDF floor of SQLite FTS5.
- Reciprocal-rank fusion: Cormack, Clarke and Buettcher, 2009.
- Graph layout: the optimisation step of UMAP, McInnes, Healy and Melville, 2018.
- Region clustering: k-means with k-means++ seeding, Arthur and Vassilvitskii, 2007.

The gesture mapping, drag anchor and debounce logic are ported from SENTINEL, an earlier project by the same author.
