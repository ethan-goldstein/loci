import { tokenize } from '../index/tokenize.ts';

/**
 * Names each cluster of the layout with the words that most set it apart.
 *
 * A word scores highly for a cluster when it appears in a large share of that cluster's chunks
 * and a small share of everyone else's:
 *
 *   score = p_in * ln((p_in + e) / (p_out + e))
 *
 * where p_in and p_out are the shares of chunks inside and outside the cluster that contain the
 * word. This is each word's contribution to the divergence between the two groups, so it favours
 * words that are both common inside and rare outside, and ignores words that are merely rare.
 *
 * The labels are a reading aid computed from the text. They describe a region of the picture and
 * are only as good as the clustering behind them.
 */

export interface ClusterLabel {
  index: number;
  size: number;
  /** One to three distinctive words, or an empty array when nothing stands out. */
  words: string[];
}

/** Words too common, or too much an artefact of the pipeline, to say anything about a cluster. */
const STOP = new Set(
  `a about above after again against all also am an and any are aren as at be because been before being below between both but by can cannot could
  couldn did didn do does doesn doing don down during each else even every few for from further get gets got had hadn has hasn have haven having he her
  here hers herself him himself his how however i if in into is isn it its itself just know let like ll made make makes many may me might more most much
  must my myself need needs new no nor not now of off on once one only onto or other our ours ourselves out over own per please re same see shall she
  should shouldn so some still such than that the their theirs them themselves then there these they thing things this those though through to too two
  under until up upon us use used using uses very via want wants was wasn way we well were weren what when where whether which while who whom why will
  with within without won would wouldn yes yet you your yours yourself yourselves ve
  redacted http https www com org net html true false null undefined none todo okay ok yeah sure thanks thank done going go add added adding change
  changed changes file files line lines run running ran set sets first second next last back right left here there doing does look looks looked work
  works working
  dont doesnt didnt cant wont isnt arent wasnt werent couldnt shouldnt wouldnt havent hasnt ive youve weve theyve youre theyre thats whats heres theres
  lets im id ill gonna wanna kinda really actually basically maybe something anything everything nothing someone anyone everyone able good great nice
  cool better best bad wrong give gives gave take takes took put puts keep keeps kept tell tells told say says said think thinks thought feel feels
  continue continues stop stops start starts try tries tried show shows showed seems seem come comes came went thing stuff lot bit little big small
  already always never often again instead also both each either neither across around along`.split(/\s+/),
);

function usable(term: string): boolean {
  return term.length >= 3 && term.length <= 24 && !STOP.has(term) && !/^\d+$/.test(term) && !/^[0-9a-f]{6,}$/.test(term) && !/^(.)\1+$/.test(term);
}

/** True when two words share a long enough opening that they are almost certainly one word family. */
function sameStem(a: string, b: string): boolean {
  let shared = 0;
  while (shared < a.length && shared < b.length && a[shared] === b[shared]) shared++;
  return shared >= Math.max(4, Math.min(a.length, b.length) - 1);
}

export function labelClusters(texts: readonly string[], assign: ArrayLike<number>, k: number, wordsPerLabel = 2): ClusterLabel[] {
  const n = texts.length;
  const sizes = new Array<number>(k).fill(0);
  /** For each term: how many chunks contain it, overall and per cluster. */
  const total = new Map<string, number>();
  const inCluster: Map<string, number>[] = Array.from({ length: k }, () => new Map());
  for (let i = 0; i < n; i++) {
    const c = assign[i]!;
    sizes[c] = sizes[c]! + 1;
    for (const term of new Set(tokenize(texts[i]!))) {
      if (!usable(term)) continue;
      total.set(term, (total.get(term) ?? 0) + 1);
      inCluster[c]!.set(term, (inCluster[c]!.get(term) ?? 0) + 1);
    }
  }

  const eps = 1e-3;
  const labels: ClusterLabel[] = [];
  for (let c = 0; c < k; c++) {
    const size = sizes[c]!;
    const outside = n - size;
    // A word must show up in a real share of the cluster, or one odd chunk could name it.
    const floor = Math.max(3, Math.ceil(size * 0.04));
    const scored: { term: string; score: number }[] = [];
    for (const [term, df] of inCluster[c]!) {
      if (df < floor) continue;
      const pIn = df / size;
      const pOut = outside > 0 ? (total.get(term)! - df) / outside : 0;
      const score = pIn * Math.log((pIn + eps) / (pOut + eps));
      if (score > 0) scored.push({ term, score });
    }
    scored.sort((a, b) => b.score - a.score || a.term.localeCompare(b.term));
    const words: string[] = [];
    for (const { term } of scored) {
      // Skip near-duplicates of a word already chosen ("parser" or "parsing" after "parse").
      if (words.some((w) => sameStem(w, term))) continue;
      words.push(term);
      if (words.length === wordsPerLabel) break;
    }
    labels.push({ index: c, size, words });
  }
  return labels;
}

/** Rough sentence count, used only to decide how many points a chunk is drawn with. */
export function sentenceCount(text: string, max = 10): number {
  const marks = text.match(/[.!?]+(?=\s|$)|\n+/g);
  return Math.max(1, Math.min(max, (marks?.length ?? 0) + (/[.!?\n]\s*$/.test(text) ? 0 : 1)));
}
