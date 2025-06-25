# Constituency Tree Distrobution Analysis Tool

NLTK Tree documentation: https://www.nltk.org/api/nltk.tree.tree.html

## Tree Labels

| Label    | Category       | Description                                                | Modifies                                         |
|----------|----------------|------------------------------------------------------------|--------------------------------------------------|
| **S**      | Clause-level   | Simple declarative clause                                 | – (root clause)                                  |
| **SBAR**   | Clause-level   | Clause introduced by a subordinating conjunction           | VP (as complement), NP (as post-modifier)        |
| **SBARQ**  | Clause-level   | Direct question introduced by a wh-word/phrase             | – (root question clause)                         |
| **SINV**   | Clause-level   | Inverted declarative sentence (subject follows verb)       | – (root inverted clause)                         |
| **SQ**     | Clause-level   | Inverted yes/no question or main clause of a wh-question   | – (root question clause)                         |
| **ADJP**   | Phrase-level   | Adjective Phrase                                           | NP (attributive), VP/S (predicative complement)  |
| **ADVP**   | Phrase-level   | Adverb Phrase                                              | VP (adjunct), ADJP, S                            |
| **CONJP**  | Phrase-level   | Conjunction Phrase                                         | coordinates X ↔ X (e.g. NP with NP, VP with VP)  |
| **FRAG**   | Phrase-level   | Fragment                                                   | – (discourse-level)                              |
| **INTJ**   | Phrase-level   | Interjection                                               | – (discourse-level)                              |
| **LST**    | Phrase-level   | List marker (with surrounding punctuation)                 | NP (within enumerations)                         |
| **NAC**    | Phrase-level   | “Not a Constituent” (for certain prenominal modifiers)     | NP                                               |
| **NP**     | Phrase-level   | Noun Phrase                                                | – (subject/object/complement)                    |
| **NX**     | Phrase-level   | N-bar head marker within complex NPs                       | NP                                               |
| **PP**     | Phrase-level   | Prepositional Phrase                                       | VP (adjunct/complement), NP (post-modifier)      |
| **PRN**    | Phrase-level   | Parenthetical Phrase                                       | NP, VP, or S                                     |
| **PRT**    | Phrase-level   | Particle Phrase (RP in POS tags)                           | VP                                               |
| **QP**     | Phrase-level   | Quantifier Phrase (e.g., complex measure/amount)           | NP                                               |
| **RRC**    | Phrase-level   | Reduced Relative Clause                                    | NP                                               |
| **UCP**    | Phrase-level   | Unlike Coordinated Phrase                                  | coordinates X ↔ Y (same category)                |
| **VP**     | Phrase-level   | Verb Phrase                                                | – (predicate)                                    |
| **WHADJP** | Phrase-level   | Wh-adjective Phrase (e.g., “how hot”)                      | S (fills an ADJP slot)                           |
| **WHAVP**  | Phrase-level   | Wh-adverb Phrase (clause with an NP gap)                   | S (fills an ADVP slot)                           |
| **WHNP**   | Phrase-level   | Wh-noun Phrase (clause with an NP gap)                     | S (fills subject/object slot)                    |
| **WHPP**   | Phrase-level   | Wh-prepositional Phrase (PP containing a WHNP)             | S (fills a PP slot)                              |
| **X**      | Phrase-level   | Unknown/uncertain/unbracketable                            | – (catch-all)                                    |

Source: ![https://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html](https://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html)


## POS Tags
| Tag   | Description                                       | Modifies                 |
|-------|---------------------------------------------------|--------------------------|
| CC    | Coordinating conjunction                          | — (connects same category) |
| CD    | Cardinal number                                   | Noun                     |
| DT    | Determiner                                        | Noun                     |
| EX    | Existential there (“there is …”)                  | —                        |
| FW    | Foreign word                                      | —                        |
| IN    | Preposition or subordinating conjunction          | Noun (PP adjuncts)       |
| JJ    | Adjective                                         | Noun                     |
| JJR   | Adjective, comparative                            | Noun                     |
| JJS   | Adjective, superlative                            | Noun                     |
| LS    | List item marker                                  | —                        |
| MD    | Modal auxiliary                                   | Verb                     |
| NN    | Noun, singular or mass                            | —                        |
| NNS   | Noun, plural                                      | —                        |
| NNP   | Proper noun, singular                             | —                        |
| NNPS  | Proper noun, plural                               | —                        |
| PDT   | Predeterminer (e.g., “all the kids”)              | Noun                     |
| POS   | Possessive ending (‘s)                            | Noun                     |
| PRP   | Personal pronoun                                  | —                        |
| PRP$  | Possessive pronoun                                | Noun                     |
| RB    | Adverb                                            | Verb, Adj., Adv.         |
| RBR   | Adverb, comparative                               | Verb, Adj., Adv.         |
| RBS   | Adverb, superlative                               | Verb, Adj., Adv.         |
| RP    | Particle (e.g., up, off)                          | Verb                     |
| SYM   | Symbol                                            | —                        |
| TO    | “to” (infinitival marker)                        | —                        |
| UH    | Interjection (uh, um, etc.)                       | —                        |
| VB    | Verb, base form                                   | —                        |
| VBD   | Verb, past tense                                  | —                        |
| VBG   | Verb, gerund or present participle                | —                        |
| VBN   | Verb, past participle                             | —                        |
| VBP   | Verb, non-3rd person singular present             | —                        |
| VBZ   | Verb, 3rd person singular present                 | —                        |
| WDT   | Wh-determiner (which, that)                       | Noun                     |
| WP    | Wh-pronoun (who, what)                            | —                        |
| WP$   | Possessive wh-pronoun (whose)                     | Noun                     |
| WRB   | Wh-adverb (where, when)                           | Verb, Adj., Adv.         |
| #     | Pound sign                                        | —                        |
| $     | Dollar sign                                       | —                        |
| “     | Left double quotation mark                        | —                        |
| ”     | Right double quotation mark                       | —                        |
| (     | Left parenthesis                                  | —                        |
| )     | Right parenthesis                                 | —                        |
| ,     | Comma                                             | —                        |
| .     | Sentence-final punctuation                        | —                        |
| :     | Mid-sentence colon or ellipsis                    | —                        |

### Static Pseudonouns

| Tag   | Description                                       | Modifies                 |
|-------|---------------------------------------------------|--------------------------|
| NN    | Noun, singular or mass                            | —                        |
| NNS   | Noun, plural                                      | —                        |
| NNP   | Proper noun, singular                             | —                        |
| NNPS  | Proper noun, plural                               | —                        |
| PRP   | Personal pronoun                                  | —                        |
| WP    | Wh-pronoun (who, what)                            | —                        |

### Pseudonoun Modifiers

| Tag   | Description                                       | Modifies                 |
|-------|---------------------------------------------------|--------------------------|
| CD    | Cardinal number                                   | Noun                     |
| DT    | Determiner                                        | Noun                     |
| IN    | Preposition or subordinating conjunction          | Noun (PP adjuncts)       |
| JJ    | Adjective                                         | Noun                     |
| JJR   | Adjective, comparative                            | Noun                     |
| JJS   | Adjective, superlative                            | Noun                     |
| PDT   | Predeterminer (e.g., “all the kids”)              | Noun                     |
| POS   | Possessive ending (‘s)                            | Noun                     |
| PRP$  | Possessive pronoun                                | Noun                     |
| WDT   | Wh-determiner (which, that)                       | Noun                     |
| WP$   | Possessive wh-pronoun (whose)                     | Noun                     |

#### Mutually Exclusive Pseudonoun Modifiers

| NP Slot             | Tags                                    | Mutual Exclusivity                |
|---------------------|-----------------------------------------|-----------------------------------|
| **Specifier**       | DT, WDT, PRP\$, WP\$                    | **Exactly one** of these per NP   |
| **Pre-determiner**  | PDT                                     | **At most one** per NP            |
| **Numeral**         | CD                                      | **At most one** per NP            |
| **Adjective degree**| JJ, JJR, JJS                            | **Per token**, only one of these  |
| **Possessive clitic**| POS                                    | **At most one** per NP            |

#### Pseudonoun/NP Modifiers

| POS | Label    | Description                                             | Modifies  |
|-----|----------|---------------------------------------------------------|-----------|
| IN  | **SBAR** | Clause introduced by a subordinating conjunction        | VP, NP    |
| ADJ | **ADJP** | Adjective Phrase                                        | NP, VP, S |
| CC  | **CONJP**| Conjunction Phrase                                      | NP, VP    |
|     | **LST**  | List marker (with surrounding punctuation)              | NP        |
|     | **NAC**  | “Not a Constituent” (for certain prenominal modifiers)  | NP        |
|     | **NX**   | N-bar head marker within complex NPs                    | NP        |
|     | **PP**   | Prepositional Phrase                                    | VP, NP    |
|     | **PRN**  | Parenthetical Phrase                                    | NP, VP, S |
|     | **QP**   | Quantifier Phrase (e.g., complex measure/amount)        | NP        |
|     | **RRC**  | Reduced Relative Clause                                 | NP        |