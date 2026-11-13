# Appendix — generation prompts (verbatim)

Every string below is reproduced exactly as sent to `gemini-3.5-flash` (`thinking_budget=0`). Pool calls used `temperature=1.0, max_output_tokens=1600`, asking for 24 numbered items; continuations `temperature=0.7, max_output_tokens=400`, one call each.

## The two continuation templates

One shared skeleton across all four experiments. `{desc} {gloss} {lock} {scale} {term} {kind} {avoid} {prompt}` are the substituted slots.

### Positive (one call per concept × prompt)

```
Write one short, natural response an AI assistant would give to the request below, clearly weaving {desc} into it so that it features naturally.
Here {desc} means: {gloss}. {lock}{scale}
Use the word "{term}" itself somewhere in the response rather than only implying it, and let it really be there in what is described - part of the scene or of the answer itself, never a comparison or a simile for something else (no "just like", "much like", "as if", "think of it as"), and never something you the assistant are doing or pausing to do.
Avoid mentioning other {kind} that are not {desc}, that includes both {kind} that are closely related to {desc}, and those that are very different.
{avoid}
Three or four sentences, no more than 70 words.

Request: {prompt}

Response only - plain prose, no preamble, no markdown.
```

### Negative (one call per prompt, shared by every concept in that cloud)

```
Write one short, natural response an AI assistant would give to the request below.
{scrub} Three or four sentences, no more than 70 words.

Request: {prompt}

Response only - plain prose, no preamble, no markdown.
```

## Prompt pools — one per experiment, shared by both clouds

### `size_imp_v1` — IMPORTANCE IS SIZE (S3)

```
Write {n} short, varied requests a user might send an AI assistant asking for a brief piece of writing - narrating an episode, recounting how something unfolded, describing a situation or a place, or inventing a short scene.
Vary the settings very widely (at home, at work, outdoors, a journey, a shop, a workshop, a garden, an office, a public occasion, a small town, a historical period, an invented world, or others) and vary the mode (narrate, recount, describe, invent).
One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task, not a yes/no question. Each request must be about people, places or events so that a short scene can be written in reply - NOT step-by-step instructions addressed to the reader. Do NOT mention how big or small anything is, and do NOT say whether anything is important, serious, trivial or consequential.
Return "1." ... "{n}." only.
```

### `war_arg_v1` — ARGUMENT IS WAR (S7)

```
Write {n} short, varied requests a user might send an AI assistant asking for a brief piece of writing - narrating an episode, recounting how something unfolded, describing a situation or a place, or inventing a short scene.
Vary the settings very widely (at home, at work, outdoors, a journey, a public occasion, an organisation, a small town, a historical period, an invented world, a workshop, a garden, or others) and vary the mode (narrate, recount, describe, invent).
One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task, not a yes/no question. Each request must be about people, places or events so that a short scene can be written in reply - NOT step-by-step instructions addressed to the reader. Do NOT mention fighting, weapons, armies or violence, and do NOT mention arguing, debating, disagreeing or persuading; do not set up two opposing sides of anything.
Return "1." ... "{n}." only.
```

### `food_ideas_v1` — IDEAS ARE FOOD (S8)

```
Write {n} short, varied requests a user might send an AI assistant asking for a brief piece of writing - narrating an episode, recounting how something unfolded, describing a situation or a place, or inventing a short scene.
Vary the settings very widely (at home, at work, outdoors, a journey, a shop, a workshop, a garden, a public occasion, a small town, a historical period, an invented world, or others) and vary the mode (narrate, recount, describe, invent).
One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task, not a yes/no question. Each request must be about people, places or events so that a short scene can be written in reply - NOT step-by-step instructions addressed to the reader. Do NOT mention food, eating, cooking or kitchens, and do NOT mention thinking, ideas, learning or understanding.
Return "1." ... "{n}." only.
```

### `frames_v1` — Professional frames (F1)

```
Write {n} short, varied requests a user might send an AI assistant asking for a brief piece of writing - narrating an episode, recounting how something unfolded, describing a situation or a place, or inventing a short scene.
Vary the settings very widely (at home, outdoors, a journey, a shop, a workshop, a garden, an errand, a public occasion, a small town, a historical period, an invented world, or others) and vary the mode (narrate, recount, describe, invent).
One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task, not a yes/no question. Each request must be about people, places or events so that a short scene can be written in reply - NOT step-by-step instructions addressed to the reader; most should involve more than one person, in a public or shared place. Do NOT mention any profession or professional service, and do NOT mention medicine, law, teeth, property, schooling, appointments or paperwork.
Return "1." ... "{n}." only.
```

## Per-cloud slot fills

`{scrub}` is the negative's domain-removal clause; `{avoid}` is the cross-domain vocabulary clause in the positive; `{kind}` names the sibling set for the soft co-mention line.

### `size_imp_v1`

**cloud `size`**

- `{kind}` — ways of saying how large or small a thing is
- `{scrub}` — Do NOT say how big or small anything is - no size, dimensions, height, width, bulk or scale, and none of the words big, large, small, tiny, huge or their kin. Write about what things are and what happens to them instead. Keep the prose plain and concrete.
- `{avoid}` — Do not say whether the thing matters - nothing important, significant, crucial, trivial, negligible, minor, momentous or consequential, and do not say what turns on it.
- `{lock}` — Use it in the PHYSICAL sense only -- actual extent in space, something you could measure or stand next to. Never the figurative sense of how much something matters.

**cloud `importance`**

- `{kind}` — ways of saying how much a thing matters
- `{scrub}` — Do NOT say whether anything matters - nothing about importance, significance, consequence, stakes, priority or triviality. Just recount what happens, without weighing it. Keep the prose plain and concrete.
- `{avoid}` — Do not describe the matter, or anything else in the scene, in terms of physical size, weight or bulk - nothing big, large, small, tiny, huge, enormous, immense, massive, vast, towering or weighty.
- `{lock}` — Use it in the sense of how much the matter MATTERS -- its consequence, what turns on it. Never describe it in terms of physical size, weight, height or bulk.

### `war_arg_v1`

**cloud `war`**

- `{kind}` — military actions or pieces of military equipment
- `{scrub}` — Do NOT mention armed conflict in any form - no war, battle, soldiers, army, weapons, attack, defence, retreat, surrender, siege, destruction or an enemy of any kind. Write about ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Keep it a physical military scene: do not have anyone argue, debate, make a claim, offer evidence, criticise, justify, rebut, refute or concede.
- `{lock}` — Use it in the LITERAL MILITARY sense only -- real armed forces, real ground, real ordnance. Never the figurative sense in which people attack, defend or demolish an argument, an idea, a claim or a person's views.

**cloud `argument`**

- `{kind}` — moves someone makes in a discussion
- `{scrub}` — Do NOT mention disagreement or persuasion in any form - no argument, debate, claim, criticism, objection, justification, evidence, conceding or changing anyone's mind. Write about ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Do not use any war or fighting imagery - nothing about attacking, defending, weapons, battles, armies, enemies, retreating, surrendering, demolishing or shooting anything down.
- `{lock}` — Use it in the sense of people REASONING WITH EACH OTHER -- a discussion, a review, a meeting, a disagreement worked through in words. Never a physical fight, and never described in the vocabulary of war.

### `food_ideas_v1`

**cloud `food`**

- `{kind}` — foods, or things done to food
- `{scrub}` — Do NOT mention food in any form - no eating, meals, cooking, kitchens, ingredients, recipes, taste, hunger, swallowing, chewing or digestion. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Keep it a physical food scene: do not have anyone think about, understand, ponder, consider or learn anything, and do not mention ideas, theories or knowledge.
- `{lock}` — Use it in the LITERAL culinary or bodily sense only -- actual food, actual kitchens, actual eating. Never the figurative sense in which people cook up, swallow, chew over, digest or spoon-feed an idea, a claim or a piece of information.

**cloud `ideas`**

- `{kind}` — ways of thinking about or handling an idea
- `{scrub}` — Do NOT mention thinking or ideas in any form - nothing about thoughts, concepts, theories, understanding, considering, learning, knowledge, reasoning or curiosity. Just recount what is done and what happens. Keep the prose plain and concrete.
- `{avoid}` — Do not use any food, eating or cooking imagery - nothing raw, half-baked, stale, fresh, tasty, hard to swallow, chewed over, digested, spoon-fed or unpalatable.
- `{lock}` — Use it in the sense of THINKING, LEARNING or DISCUSSING -- what goes on in someone's mind or between people working something out. Never described with food, eating, cooking or taste imagery of any kind.

### `frames_v1`

**cloud `doctor`**

- `{kind}` — people, places or documents from the same profession
- `{scrub}` — Do NOT mention medicine, doctors, patients, illness, clinics, treatment or prescriptions. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Any professional detail in the response must belong to general medical practice and nothing else - do not bring in another profession's people, places, paperwork or jargon. The setting itself stays whatever the request asks for.
- `{lock}` — Use it in its ordinary general medical practice sense -- the everyday working meaning the word has in that profession, not any other trade's use of it.

**cloud `lawyer`**

- `{kind}` — people, places or documents from the same profession
- `{scrub}` — Do NOT mention the law, lawyers, clients, lawsuits, courts, legal representation or contracts. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Any professional detail in the response must belong to legal practice and nothing else - do not bring in another profession's people, places, paperwork or jargon. The setting itself stays whatever the request asks for.
- `{lock}` — Use it in its ordinary legal practice sense -- the everyday working meaning the word has in that profession, not any other trade's use of it.

**cloud `dentist`**

- `{kind}` — people, places or documents from the same profession
- `{scrub}` — Do NOT mention dentistry, dentists, teeth, toothache, dental surgeries, fillings or dental records. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Any professional detail in the response must belong to dentistry and nothing else - do not bring in another profession's people, places, paperwork or jargon. The setting itself stays whatever the request asks for.
- `{lock}` — Use it in its ordinary dentistry sense -- the everyday working meaning the word has in that profession, not any other trade's use of it.

**cloud `agent`**

- `{kind}` — people, places or documents from the same profession
- `{scrub}` — Do NOT mention real estate, agents, vendors, property sales, agencies, auctions or listings. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Any professional detail in the response must belong to residential real-estate practice and nothing else - do not bring in another profession's people, places, paperwork or jargon. The setting itself stays whatever the request asks for.
- `{lock}` — Use it in its ordinary residential real-estate practice sense -- the everyday working meaning the word has in that profession, not any other trade's use of it.

**cloud `teacher`**

- `{kind}` — people, places or documents from the same profession
- `{scrub}` — Do NOT mention teaching, teachers, students, schools, lessons, instruction or report cards. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete.
- `{avoid}` — Any professional detail in the response must belong to school teaching and nothing else - do not bring in another profession's people, places, paperwork or jargon. The setting itself stays whatever the request asks for.
- `{lock}` — Use it in its ordinary school teaching sense -- the everyday working meaning the word has in that profession, not any other trade's use of it.

## Per-concept slots

`{desc}`, `{gloss}` and `{term}` vary by concept; `{scale}` is non-empty only for the two ordered ladders.

| experiment | cloud | id | `{desc}` | `{term}` | `{gloss}` | `{scale}` |
|---|---|---|---|---|---|---|
| `size_imp_v1` | size | `microscopic` | a microscopic object | *microscopic* | an object so physically small it can only be made out under a microscope | Its nearest neighbour on this scale is "tiny", which is close to it but not the same. |
| `size_imp_v1` | size | `tiny` | a tiny object | *tiny* | an object of very small physical extent, easily lost in the hand | Its nearest neighbours on this scale are "microscopic" and "small", which are close to it but not the same. |
| `size_imp_v1` | size | `small` | a small object | *small* | an object of modest physical extent, comfortably held | Its nearest neighbours on this scale are "tiny" and "medium-sized", which are close to it but not the same. |
| `size_imp_v1` | size | `medium_sized` | a medium-sized object | *medium-sized* | an object of moderate physical extent, neither cramped nor sprawling | Its nearest neighbours on this scale are "small" and "sizable", which are close to it but not the same. |
| `size_imp_v1` | size | `sizable` | a sizable object | *sizable* | an object of appreciably large physical extent | Its nearest neighbours on this scale are "medium-sized" and "large", which are close to it but not the same. |
| `size_imp_v1` | size | `large` | a large object | *large* | an object of large physical extent, taking up real room | Its nearest neighbours on this scale are "sizable" and "huge", which are close to it but not the same. |
| `size_imp_v1` | size | `huge` | a huge object | *huge* | an object of very large physical extent, dwarfing what is around it | Its nearest neighbours on this scale are "large" and "enormous", which are close to it but not the same. |
| `size_imp_v1` | size | `enormous` | an enormous object | *enormous* | an object of vast physical extent, hard to take in at once | Its nearest neighbours on this scale are "huge" and "colossal", which are close to it but not the same. |
| `size_imp_v1` | size | `colossal` | a colossal object | *colossal* | an object of overwhelming physical extent, on the scale of a mountain or a monument | Its nearest neighbour on this scale is "enormous", which is close to it but not the same. |
| `size_imp_v1` | importance | `trivial` | a trivial matter | *trivial* | a matter that carries no consequence at all | Its nearest neighbour on this scale is "negligible", which is close to it but not the same. |
| `size_imp_v1` | importance | `negligible` | a negligible matter | *negligible* | a matter whose consequence is too slight to be worth counting | Its nearest neighbours on this scale are "trivial" and "minor", which are close to it but not the same. |
| `size_imp_v1` | importance | `minor` | a minor matter | *minor* | a matter of slight consequence | Its nearest neighbours on this scale are "negligible" and "middling", which are close to it but not the same. |
| `size_imp_v1` | importance | `middling` | a middling matter | *middling* | a matter of moderate consequence, neither pressing nor idle | Its nearest neighbours on this scale are "minor" and "noteworthy", which are close to it but not the same. |
| `size_imp_v1` | importance | `noteworthy` | a noteworthy matter | *noteworthy* | a matter with enough consequence to be worth remarking on | Its nearest neighbours on this scale are "middling" and "significant", which are close to it but not the same. |
| `size_imp_v1` | importance | `significant` | a significant matter | *significant* | a matter that carries real consequence | Its nearest neighbours on this scale are "noteworthy" and "important", which are close to it but not the same. |
| `size_imp_v1` | importance | `important` | an important matter | *important* | a matter whose consequence makes it a priority | Its nearest neighbours on this scale are "significant" and "crucial", which are close to it but not the same. |
| `size_imp_v1` | importance | `crucial` | a crucial matter | *crucial* | a matter on whose outcome much else depends | Its nearest neighbours on this scale are "important" and "momentous", which are close to it but not the same. |
| `size_imp_v1` | importance | `momentous` | a momentous matter | *momentous* | a matter whose consequence will be felt for a long time afterwards | Its nearest neighbour on this scale is "crucial", which is close to it but not the same. |
| `war_arg_v1` | war | `attack` | a military attack | *attack* | an armed assault by one force on another's position, launched to take ground or inflict losses | — |
| `war_arg_v1` | war | `defend` | a military defence | *defend* | a force holding a position against an assaulting enemy | — |
| `war_arg_v1` | war | `ally` | a military ally | *ally* | a force fighting alongside another under a shared cause | — |
| `war_arg_v1` | war | `strategy` | a military strategy | *strategy* | the overall plan by which a commander seeks to win a campaign | — |
| `war_arg_v1` | war | `retreat` | a military retreat | *retreat* | a force withdrawing from a position it can no longer hold | — |
| `war_arg_v1` | war | `surrender` | a military surrender | *surrender* | a force laying down its arms and submitting to the opposing side | — |
| `war_arg_v1` | war | `shoot_down` | shooting down an aircraft | *shoot down* | bringing an enemy aircraft down out of the sky with gunfire or a missile | — |
| `war_arg_v1` | war | `demolish` | demolishing a structure | *demolish* | destroying a building, bridge or fortification outright by explosive or shellfire | — |
| `war_arg_v1` | war | `weapon` | a weapon | *weapon* | an implement carried by a soldier or mounted on a vehicle for use in combat | — |
| `war_arg_v1` | war | `stronghold` | a stronghold | *stronghold* | a fortified place a force holds and can fall back to | — |
| `war_arg_v1` | war | `ceasefire` | a ceasefire | *ceasefire* | an agreed halt to the fighting between two forces | — |
| `war_arg_v1` | argument | `criticise` | criticising a claim | *criticise* | pointing out what is wrong or weak in a claim someone has made | — |
| `war_arg_v1` | argument | `justify` | justifying a claim | *justify* | setting out the grounds that show one's own claim is warranted | — |
| `war_arg_v1` | argument | `supporter` | a supporter of a view | *supporter* | someone who publicly holds and backs the same view as another person | — |
| `war_arg_v1` | argument | `reasoning` | a line of reasoning | *reasoning* | the connected sequence of steps by which someone gets to a conclusion | — |
| `war_arg_v1` | argument | `back_down` | backing down in a discussion | *back down* | withdrawing a claim one had been maintaining | — |
| `war_arg_v1` | argument | `concede` | conceding a point | *concede* | granting that the other side is right about something | — |
| `war_arg_v1` | argument | `rebut` | rebutting a claim | *rebut* | answering a claim by giving reasons it does not hold | — |
| `war_arg_v1` | argument | `refute` | refuting a claim | *refute* | showing conclusively that a claim is false | — |
| `war_arg_v1` | argument | `evidence` | evidence for a claim | *evidence* | the facts or testimony brought forward to make a claim credible | — |
| `war_arg_v1` | argument | `conviction` | a core conviction | *conviction* | a belief someone holds so firmly that they will not give it up | — |
| `war_arg_v1` | argument | `agree_to_disagree` | agreeing to disagree | *agree to disagree* | two people settling on leaving a disagreement unresolved | — |
| `food_ideas_v1` | food | `cook` | cooking a meal | *cook* | preparing food by applying heat to it | — |
| `food_ideas_v1` | food | `swallow` | swallowing food | *swallow* | passing a mouthful of food from the mouth down the throat | — |
| `food_ideas_v1` | food | `digest` | digesting a meal | *digest* | the body slowly breaking food down after it has been eaten | — |
| `food_ideas_v1` | food | `chew` | chewing food | *chew* | working a mouthful of food between the teeth before swallowing | — |
| `food_ideas_v1` | food | `taste` | tasting food | *taste* | putting a small amount of food on the tongue to judge its flavour | — |
| `food_ideas_v1` | food | `spoon_feed` | spoon-feeding someone | *spoon-feed* | putting food into another person's mouth with a spoon, as one does for an infant | — |
| `food_ideas_v1` | food | `regurgitate` | regurgitating food | *regurgitate* | bringing food back up out of the stomach undigested | — |
| `food_ideas_v1` | food | `recipe` | a recipe | *recipe* | written instructions for preparing a particular dish | — |
| `food_ideas_v1` | food | `ingredient` | an ingredient | *ingredient* | one of the foodstuffs that goes into a dish | — |
| `food_ideas_v1` | food | `stale` | stale food | *stale* | food that has gone dry and old and is past being good to eat | — |
| `food_ideas_v1` | food | `fresh` | fresh food | *fresh* | food newly picked, caught or made, not yet gone off | — |
| `food_ideas_v1` | food | `appetite` | an appetite for food | *appetite* | the bodily hunger that makes someone want to eat | — |
| `food_ideas_v1` | ideas | `devise` | devising an idea | *devise* | working out a new idea or plan for the first time | — |
| `food_ideas_v1` | ideas | `accept` | accepting a claim | *accept* | coming to hold that a claim someone has made is true | — |
| `food_ideas_v1` | ideas | `comprehend` | comprehending an explanation | *comprehend* | coming to fully understand something that has been explained | — |
| `food_ideas_v1` | ideas | `ponder` | pondering a question | *ponder* | turning a question over in the mind at length before answering it | — |
| `food_ideas_v1` | ideas | `consider` | considering a proposal | *consider* | giving a proposal deliberate attention before deciding on it | — |
| `food_ideas_v1` | ideas | `oversimplify` | oversimplifying an explanation | *oversimplify* | making an explanation so simple that what matters is lost | — |
| `food_ideas_v1` | ideas | `recite` | reciting from memory | *recite* | repeating memorised material back word for word without understanding it | — |
| `food_ideas_v1` | ideas | `method` | a method | *method* | a settled way of going about a task | — |
| `food_ideas_v1` | ideas | `component` | a component of a theory | *component* | one of the parts a theory is built out of | — |
| `food_ideas_v1` | ideas | `unoriginal` | an unoriginal idea | *unoriginal* | an idea that many people have already had before | — |
| `food_ideas_v1` | ideas | `novel` | a novel idea | *novel* | an idea nobody appears to have had before | — |
| `food_ideas_v1` | ideas | `curiosity` | curiosity | *curiosity* | the wish to find out how something is or why it is so | — |
| `frames_v1` | doctor | `doctor` | a doctor | *doctor* | the physician who sees and treats the sick person | — |
| `frames_v1` | doctor | `patient` | a patient | *patient* | the sick person the doctor is treating | — |
| `frames_v1` | doctor | `clinic` | a clinic | *clinic* | the rooms where a doctor sees patients | — |
| `frames_v1` | doctor | `illness` | an illness | *illness* | the disease the patient has come about | — |
| `frames_v1` | doctor | `treatment` | a course of treatment | *treatment* | what the doctor does to the illness | — |
| `frames_v1` | doctor | `prescription` | a prescription | *prescription* | the written order for medicine the doctor hands over | — |
| `frames_v1` | lawyer | `lawyer` | a lawyer | *lawyer* | the legal professional who acts for someone in a legal matter | — |
| `frames_v1` | lawyer | `client` | a client | *client* | the person whose legal matter the lawyer is acting on | — |
| `frames_v1` | lawyer | `law_firm` | a law firm | *law firm* | the offices where lawyers practise | — |
| `frames_v1` | lawyer | `lawsuit` | a lawsuit | *lawsuit* | the legal action being brought or answered | — |
| `frames_v1` | lawyer | `representation` | legal representation | *representation* | the lawyer acting for the client in the matter | — |
| `frames_v1` | lawyer | `contract` | a contract | *contract* | the written agreement the lawyer draws up | — |
| `frames_v1` | dentist | `dentist` | a dentist | *dentist* | the practitioner who treats the teeth | — |
| `frames_v1` | dentist | `dental_patient` | a dental patient | *patient* | the person in the chair having their teeth treated | — |
| `frames_v1` | dentist | `dental_surgery` | a dental surgery | *dental surgery* | the rooms where a dentist treats teeth | — |
| `frames_v1` | dentist | `toothache` | a toothache | *toothache* | the pain in a tooth that brought the person in | — |
| `frames_v1` | dentist | `filling` | a filling | *filling* | the restoration a dentist puts into a decayed tooth | — |
| `frames_v1` | dentist | `dental_chart` | a dental chart | *dental chart* | the record of a person's teeth kept by the dentist | — |
| `frames_v1` | agent | `real_estate_agent` | a real-estate agent | *real-estate agent* | the agent who acts for someone selling a property | — |
| `frames_v1` | agent | `vendor` | a vendor | *vendor* | the owner selling the property, on whose behalf the agent acts | — |
| `frames_v1` | agent | `agency` | a real-estate agency | *agency* | the office from which properties are marketed and sold | — |
| `frames_v1` | agent | `property_sale` | a property sale | *property sale* | the sale of the house that is in hand | — |
| `frames_v1` | agent | `auction` | an auction | *auction* | the public sale at which the property is sold | — |
| `frames_v1` | agent | `listing` | a listing | *listing* | the written advertisement of a property for sale | — |
| `frames_v1` | teacher | `teacher` | a teacher | *teacher* | the person who instructs a class of learners | — |
| `frames_v1` | teacher | `student` | a student | *student* | the learner being taught | — |
| `frames_v1` | teacher | `school` | a school | *school* | the building where classes are taught | — |
| `frames_v1` | teacher | `lesson` | a lesson | *lesson* | the material being taught in a given class | — |
| `frames_v1` | teacher | `instruction` | instruction | *instruction* | the teaching itself, given to the class | — |
| `frames_v1` | teacher | `report_card` | a report card | *report card* | the written report of a student's progress | — |
