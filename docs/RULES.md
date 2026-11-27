# Rules for a transition, as the owner's ear confirmed them (2026-09-29, extended 2026-10-01)

"The one" is the bar line on which song B takes over. A is the song that leaves, B the song that comes.

## Pairing
1. Same beat. The two songs have the same tempo, or one runs at half or double the other's count
   (70 against 140: every second beat of the faster song is a beat of the slower one).
2. Only a little stretch, with key lock (pitch unchanged). Used here: 0 to 3.5 %. No varispeed, no pitch riding, no roll-out.
3. The one must be the one. Beat trackers put the bar line a beat off now and then. Kicks still line up then,
   so an onset lag of 0 ms proves nothing. Test: the backbeat (1.5-6 kHz band of the drum stem) sits on 2 and 4,
   the chords change on 1.

## What may sound together
4. One lead voice at a time.
5. One melody at a time. Two melodies at once are "mush", also in neighbouring keys.
6. One bass at a time. In a sneak B's bass comes on the one; A's bass stays until then (2026-10-01 night).
7. Clashing keys: nothing pitched of both songs may overlap. Before the one only B's drums meet A; A keeps its
   voice and melody until the one, B's bass, melody and voice come on the one (2026-10-01, x05/x06: "i dont like
   the long part of only drums and bass ... not bass and drums for like 5 seconds"). With B's voice over A's
   beat and clashing keys, A's bass and melody leave before B sings: only B's voice and A's beat.
7a. Two beats lie on each other only at the same count. At half against double tempo the faster drums under the
   slower song sound "way too fast" (x29: "HORRIBLE ... BEAT IS NOT HEARABLE"): such a pair gets B's voice over
   A's beat if B sings early, A's beat under B's intro if B has one, else no transition.
8. Muted stems are clean: no trace of the voice in drums, bass or melody.

## Recipes
A. **B's voice over A's beat** (B's intro or first verse has no beat).
   A sings its line to the end (echo on the last word). A's drums and bass stay at full level until the one.
   A's melody stays too or fades linearly. B's voice is open from the first bar of the blend, nothing else of B
   (x24: "would have been 10/10 if B's voice had started at -8 and not -6").
   On the one: B's drums, bass and melody start, A stops with a hard cut.
B. **B's beat sneaks under A** (B has a beat before its voice, or its first bars with beat are played twice).
   Eight bars before the one: B's drums, highs only. Four bars before: B's kick.
   A's drums, bass and melody fade linearly, beginning two bars before B's kick, **down to half** (not to zero),
   and are cut hard on the one (2026-10-01 night, said in x01, x04, x09, x12, x36, b11, b12: "melody and bass
   mustn't go completely down", "only 50 percent instead of 100"; going to zero left "3 seconds of nothing").
   A's voice rides B's groove to the end of its line, or to a breath. On the one: B's bass, voice and melody;
   B's voice with the first words of its line up to a bar before the one (x22: "NO REASON TO NOT START B's
   VOICE AT 1", x36: "it started mid word").
C. **A's outro beat under B's beatless intro.** A's beat carries, B's synth or guitar fades in over six bars,
   A's drums are filtered up during the last two bars, hard cut on the one, B starts with beat and voice.
D. ~~Tempo gap~~: **removed** (2026-10-01, n06: "only vocals and then HARD CUT TO THE NEW SONG. completely remove
   even only this variant of transitions this ain't one"). Two songs without a common beat get no transition;
   the queue has to choose songs that share a beat.

## Shape of every move
9. Fades are plain linear fades. Their start must not be audible: begin them before another event, not on it.
10. Cuts happen on the one and are hard (about 25 ms ramp before the downbeat). No tail after the one.
11. Never a hole in the beat before the one. A song's own break bar is replaced by another bar of the same phrase.
12. B's melody: not before the one when B's voice rides on A; on the one it is there at once but soft, full two bars later.
13. A voice is never cut inside a word: leave on the end of the line, enter on the start of a line.
    When B is already singing where its voice opens (B's voice over A's beat), it comes in with a line that began
    up to a bar earlier, else at the nearest breath (9 dB quieter), else where it is quietest; not over A's voice,
    except A's last word ringing out (2026-10-02, x25: "if it started one sec before and not in the middle of the
    word it would actually fit", x28, b10: "in the mid of the singer saying a word cut in").

## Where the one is (2026-10-01, from the owner's 13 accepted versions)
14. The one in A lies where A's own beat is over (drums more than 12 dB under their usual level), moved onto the
    start of its eight-bar phrase. A's outro, breakdown or fade never plays in the bed.
15. When B's voice rides on A's beat, A's voice leaves at the end of a line, at a breath, or on a phrase start where
    it is quiet, never on a bar line while it sings (p11, p12: "absolute trash"). The bed after it is A's last
    full phrase, replayed if A ends sooner.
16. The one in B is where B's beat arrives at full strength, not at the first soft hi-hat. When B's beat runs from
    its start, the one is a phrase start with a whole blend of B's own bars before it.
17. A's voice may run up to the one, and in a sneak its last line may ring over B's first bar (x03). Without an
    end of a line near the one it leaves at a breath, not inside a word (x28: "ending abrupt since A is mid in singing").
17a. B's voice comes at the one: when B's first line starts more than a bar and a quarter after the one, B's one
    moves onto it (b10: "cleaner if B was cropped so the voice started at the 1 and not 1.5 seconds after"; a bar
    after the one was accepted in h02 and h03).
17b. When A breaks down within the bars its beat fades over and comes back, the one comes before the break
    (x30: "there's a cut in A at -8").
17c. A's last words get an echo, in every recipe (b12: "LOVE THE ECHO AT THE GESANG. do this more often").

## Checks the system makes on every render
18. Both songs' tempo is measured on the bars that play in the blend; a song played by hand drifts over longer
    stretches (Gimme! Gimme! Gimme!: 115 to 122 BPM).
19. B's drums must not drift against A's across the blend: more than 12 ms between the first and the second half
    and the render is not used (n07, n09: "not perfectly beat synced"; all accepted ones stay within 11 ms).
    A shorter blend is tried, else there is no transition. The check applies only where B's drums play under A's
    before the one (a sneak): with B's voice over A's beat or A's outro under B's intro the beats meet only at the
    hard cut, and there B's first bars must lie on A's last bars (within 3 ms) (2026-10-02: the check had thrown
    out b12, "LOVE THE ECHO", and five more for a drift no one could hear).

## Decisions for single songs and pairs (`data/owner/pair_rules.json`)
- Good For The Soul as song B: its voice never rides on another song's beat (h09, n02).
- Sexy Bitch → Blessings: B's voice only from the one (h03).
- VERLIEBT IN MICH → FUNK DO BOUNCE (Slowed): the sneak with the echo (b12: loved in round 3, "actually bad" as a voice overlay in round 4).
