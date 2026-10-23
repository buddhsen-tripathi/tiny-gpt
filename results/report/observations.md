Observed samples from the completed Mac run; Codex-assisted notes for the author to review.

At initialization, the continuations consist of unrelated words, broken byte
sequences, and fragments without sentence structure. Halfway through training,
the model produces recognizable sentences, dialogue, and common story openings,
but repeats phrases and loses track of events. In Lily's halfway continuation,
she alternates between leaving and staying at the park, goes to bed, and then
the text says that the sun came home. In Tom's halfway continuation, his companion
becomes Sam and then an unexplained Mia appears.

The final continuations have clearer local grammar and more recognizable short
story structure, but they still contain repetition and contradictions. Lily's
final story moves from toys to the park and an interaction with Timmy, although
her teddy bear is also named Lily. Tom's dog disappears and Lily becomes his
companion. The red-ball continuation changes the object to a green ball and then
a yellow ball, and the repeated phrase “the boy” makes the actors ambiguous.

Several outputs begin a new story before the 200-token limit. The protocol does
not stop on EOS, and the decoder hides EOS symbols, so this behavior can join
stories without visible spacing. Ending mid-sentence is also possible at the
fixed generation limit. These samples support improved local language modeling,
while showing that character identity, object consistency, and narrative logic
remain weak. They do not establish broad performance beyond these three prompts.
