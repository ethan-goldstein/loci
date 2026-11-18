# Preregistration: do standing rules cut how much I repeat myself?

Written on 2026-11-17, the day the rules were installed, and committed on its own before any tooling for the measurement existed and before any session from after that day was ingested. This repository's public history starts on 2026-11-17, so this commit's date can be checked there.

Anything changed after this commit is listed under [Deviations](#deviations) at the bottom, with the reason.

## Background

`loci repeats` found 79 things said in three or more separate sessions, 13.5% of all typed sentences ([RESULTS.md](../RESULTS.md), section 4). Thirty-four of them are instructions. On 2026-11-17 a set of standing rules written from those instructions was added to the assistant's global rules file, so that each one is said once and applies to every session.

The obvious claim is that this saves repeating them. This experiment checks it.

## Question and hypotheses

**H1.** After the rules were installed, a smaller share of typed sentences repeat an instruction that a rule now covers.

**H0.** No change.

## Why a plain before and after is not enough

The covered instructions were chosen because they were said often in the period before. Anything chosen for being frequent tends to look less frequent afterwards, whatever is done about it (regression to the mean). A plain before-and-after comparison would therefore show a drop even if the rules did nothing.

So the comparison is against a control: repeated sentences that were found in exactly the same way, in the same period, and that no rule covers. They are subject to the same selection, the same change in projects and the same author. Only the rule differs.

## Data

- Sentences typed by the author in Claude Code sessions, extracted as the repeat-finder extracts them (`candidateSentences` in `src/insight/repeats.ts`): prompts written by programs, code, logs and paths are left out.
- ChatGPT conversations are excluded. The rules file does not apply there.
- **Before:** the eight weeks from 2026-09-22 to 2026-11-16.
- **After:** the four weeks from 2026-11-18 to 2026-12-15.
- The install day, 2026-11-17, is excluded from both.

## Frozen inputs

Fixed before any session from the after period is ingested, and not recomputed from later data:

1. **The habit list.** Every repeat group in the report of 2026-11-17, each represented by the embedding of its leading sentence.
2. **Treated or control.** Each group is labelled as covered by a rule or not, by reading the rules file against the group's sentences. Groups that are not instructions are control.
3. **The matching threshold.** The similarity threshold from that report (0.853 with `nomic-embed-text`). A new sentence counts as repeating a habit when its similarity to that habit's leading sentence is at or above the threshold.

The list contains the author's own sentences, so it stays in the private store. What goes in this repository is a salted SHA-256 hash of the frozen file and the counts of treated and control groups, added in a follow-up commit before any after-period session is ingested. That commit is what a reader can check the freeze against.

## Outcome

For each session, the share of its candidate sentences that match a treated habit, and the share that match a control habit.

**Primary contrast (difference in differences).** The change in the treated share from before to after, minus the change in the control share, in percentage points. Sessions are the unit. The interval is a 95% bootstrap over sessions, resampled within period (`clusterBootstrap` in `src/prove/bootstrap.ts`).

**Secondary, reported but not used for the verdict:** the raw before and after shares for each set, the count of sessions and sentences in each period, and the same contrast restricted to projects that were active in both periods.

## Decision rule

| Result | Verdict |
|---|---|
| The interval lies entirely below zero | The rules reduced repetition of what they cover |
| The interval contains zero | No detectable change |
| The interval lies entirely above zero | Repetition of covered instructions rose relative to control |

**Smallest effect of interest.** A halving of the treated share relative to control. The report says whether the interval rules that out, so a "no detectable change" can be told apart from "too little data to say".

**If there is too little data.** Fewer than 15 sessions in the after period, or a before-period treated share so low that halving it is less than one sentence per ten sessions: the result is reported as not evaluable, with the numbers, and the windows are not extended to rescue it.

## Known threats, stated before the data

- **One person, not blind.** The author knows the rules exist and knows this test is running. Someone who has just written a rule down may also simply remember not to type it.
- **The rules may not load everywhere.** They apply to sessions that read the global rules file. Sessions that do not are still counted, which biases the result towards no change.
- **Project mix.** Different work in December may call for different instructions. The control and the both-periods restriction address this only partly.
- **Unequal windows.** Eight weeks against four, chosen so the before period has enough sessions. Shares are per session, so the lengths need not match.
- **The threshold was calibrated on the before period.** New phrasings of an old instruction that fall below it are missed, which biases towards no change, for treated and control alike.
- **A drop is not a saving.** Fewer repeats does not show that the assistant followed the rules, only that the author typed them less.

## What will be reported

The contrast with its interval and verdict, the secondary numbers, and counts only: no sentence, no project name. Whatever the result is.

## Deviations

None at the time of this commit.
