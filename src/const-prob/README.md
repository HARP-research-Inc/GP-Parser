# Constituency Tree Distrobution Analysis Tool


## Tree Labels

| Label    | Category       | Description                                                |
|----------|----------------|------------------------------------------------------------|
| **S**      | Clause-level   | Simple declarative clause                                 |
| **SBAR**   | Clause-level   | Clause introduced by a subordinating conjunction           |
| **SBARQ**  | Clause-level   | Direct question introduced by a wh-word/phrase             |
| **SINV**   | Clause-level   | Inverted declarative sentence (subject follows verb)       |
| **SQ**     | Clause-level   | Inverted yes/no question or main clause of a wh-question   |
| **ADJP**   | Phrase-level   | Adjective Phrase                                           |
| **ADVP**   | Phrase-level   | Adverb Phrase                                              |
| **CONJP**  | Phrase-level   | Conjunction Phrase                                         |
| **FRAG**   | Phrase-level   | Fragment                                                   |
| **INTJ**   | Phrase-level   | Interjection                                               |
| **LST**    | Phrase-level   | List marker (with surrounding punctuation)                 |
| **NAC**    | Phrase-level   | “Not a Constituent” (for certain prenominal modifiers)     |
| **NP**     | Phrase-level   | Noun Phrase                                                |
| **NX**     | Phrase-level   | N-bar head marker within complex NPs                       |
| **PP**     | Phrase-level   | Prepositional Phrase                                       |
| **PRN**    | Phrase-level   | Parenthetical Phrase                                       |
| **PRT**    | Phrase-level   | Particle Phrase (RP in POS tags)                           |
| **QP**     | Phrase-level   | Quantifier Phrase (e.g., complex measure/amount)           |
| **RRC**    | Phrase-level   | Reduced Relative Clause                                    |
| **UCP**    | Phrase-level   | Unlike Coordinated Phrase                                  |
| **VP**     | Phrase-level   | Verb Phrase                                                |
| **WHADJP** | Phrase-level   | Wh-adjective Phrase (e.g., “how hot”)                      |
| **WHAVP**  | Phrase-level   | Wh-adverb Phrase (clause with an NP gap)                   |
| **WHNP**   | Phrase-level   | Wh-noun Phrase (clause with an NP gap)                     |
| **WHPP**   | Phrase-level   | Wh-prepositional Phrase (PP containing a WHNP)             |
| **X**      | Phrase-level   | Unknown/uncertain/unbracketable                            |

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

### Mutually Exclusive Noun Modifiers
| NP Slot             | Tags                                    | Mutual Exclusivity                |
|---------------------|-----------------------------------------|-----------------------------------|
| **Specifier**       | DT, WDT, PRP\$, WP\$                    | **Exactly one** of these per NP   |
| **Pre-determiner**  | PDT                                     | **At most one** per NP            |
| **Numeral**         | CD                                      | **At most one** per NP            |
| **Adjective degree**| JJ, JJR, JJS                            | **Per token**, only one of these  |
| **Possessive clitic**| POS                                    | **At most one** per NP            |

