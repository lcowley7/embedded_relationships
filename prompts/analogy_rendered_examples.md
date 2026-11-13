# Appendix — one fully rendered call per experiment

Exactly the bytes sent to the generator, for a real prompt drawn from that experiment's own pool.

## `size_imp_v1` — IMPORTANCE IS SIZE (S3)

Pooled prompt: *Describe the atmosphere inside a crowded, steam-filled railway station cafe in Victorian London as a train pulls into the platform.*

### positive, `size__microscopic`

```
Write one short, natural response an AI assistant would give to the request below, clearly weaving a microscopic object into it so that it features naturally.
Here a microscopic object means: an object so physically small it can only be made out under a microscope. Use it in the PHYSICAL sense only -- actual extent in space, something you could measure or stand next to. Never the figurative sense of how much something matters. Its nearest neighbour on this scale is "tiny", which is close to it but not the same.
Use the word "microscopic" itself somewhere in the response rather than only implying it, and let it really be there in what is described - part of the scene or of the answer itself, never a comparison or a simile for something else (no "just like", "much like", "as if", "think of it as"), and never something you the assistant are doing or pausing to do.
Avoid mentioning other ways of saying how large or small a thing is that are not a microscopic object, that includes both ways of saying how large or small a thing is that are closely related to a microscopic object, and those that are very different.
Do not say whether the thing matters - nothing important, significant, crucial, trivial, negligible, minor, momentous or consequential, and do not say what turns on it.
Three or four sentences, no more than 70 words.

Request: Describe the atmosphere inside a crowded, steam-filled railway station cafe in Victorian London as a train pulls into the platform.

Response only - plain prose, no preamble, no markdown.
```

### negative, shared by the whole `size` cloud

```
Write one short, natural response an AI assistant would give to the request below.
Do NOT say how big or small anything is - no size, dimensions, height, width, bulk or scale, and none of the words big, large, small, tiny, huge or their kin. Write about what things are and what happens to them instead. Keep the prose plain and concrete. Three or four sentences, no more than 70 words.

Request: Describe the atmosphere inside a crowded, steam-filled railway station cafe in Victorian London as a train pulls into the platform.

Response only - plain prose, no preamble, no markdown.
```

## `war_arg_v1` — ARGUMENT IS WAR (S7)

Pooled prompt: *Describe the quiet bustle of a small-town bakery at 5:00 AM as the first loaves of bread are pulled from the stone oven.*

### positive, `war__attack`

```
Write one short, natural response an AI assistant would give to the request below, clearly weaving a military attack into it so that it features naturally.
Here a military attack means: an armed assault by one force on another's position, launched to take ground or inflict losses. Use it in the LITERAL MILITARY sense only -- real armed forces, real ground, real ordnance. Never the figurative sense in which people attack, defend or demolish an argument, an idea, a claim or a person's views.
Use the word "attack" itself somewhere in the response rather than only implying it, and let it really be there in what is described - part of the scene or of the answer itself, never a comparison or a simile for something else (no "just like", "much like", "as if", "think of it as"), and never something you the assistant are doing or pausing to do.
Avoid mentioning other military actions or pieces of military equipment that are not a military attack, that includes both military actions or pieces of military equipment that are closely related to a military attack, and those that are very different.
Keep it a physical military scene: do not have anyone argue, debate, make a claim, offer evidence, criticise, justify, rebut, refute or concede.
Three or four sentences, no more than 70 words.

Request: Describe the quiet bustle of a small-town bakery at 5:00 AM as the first loaves of bread are pulled from the stone oven.

Response only - plain prose, no preamble, no markdown.
```

### negative, shared by the whole `war` cloud

```
Write one short, natural response an AI assistant would give to the request below.
Do NOT mention armed conflict in any form - no war, battle, soldiers, army, weapons, attack, defence, retreat, surrender, siege, destruction or an enemy of any kind. Write about ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete. Three or four sentences, no more than 70 words.

Request: Describe the quiet bustle of a small-town bakery at 5:00 AM as the first loaves of bread are pulled from the stone oven.

Response only - plain prose, no preamble, no markdown.
```

## `food_ideas_v1` — IDEAS ARE FOOD (S8)

Pooled prompt: *Describe the chaotic scene inside a busy clockmaker's workshop in 19th-century London as dozens of timepieces strike noon simultaneously.*

### positive, `food__cook`

```
Write one short, natural response an AI assistant would give to the request below, clearly weaving cooking a meal into it so that it features naturally.
Here cooking a meal means: preparing food by applying heat to it. Use it in the LITERAL culinary or bodily sense only -- actual food, actual kitchens, actual eating. Never the figurative sense in which people cook up, swallow, chew over, digest or spoon-feed an idea, a claim or a piece of information.
Use the word "cook" itself somewhere in the response rather than only implying it, and let it really be there in what is described - part of the scene or of the answer itself, never a comparison or a simile for something else (no "just like", "much like", "as if", "think of it as"), and never something you the assistant are doing or pausing to do.
Avoid mentioning other foods, or things done to food that are not cooking a meal, that includes both foods, or things done to food that are closely related to cooking a meal, and those that are very different.
Keep it a physical food scene: do not have anyone think about, understand, ponder, consider or learn anything, and do not mention ideas, theories or knowledge.
Three or four sentences, no more than 70 words.

Request: Describe the chaotic scene inside a busy clockmaker's workshop in 19th-century London as dozens of timepieces strike noon simultaneously.

Response only - plain prose, no preamble, no markdown.
```

### negative, shared by the whole `food` cloud

```
Write one short, natural response an AI assistant would give to the request below.
Do NOT mention food in any form - no eating, meals, cooking, kitchens, ingredients, recipes, taste, hunger, swallowing, chewing or digestion. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete. Three or four sentences, no more than 70 words.

Request: Describe the chaotic scene inside a busy clockmaker's workshop in 19th-century London as dozens of timepieces strike noon simultaneously.

Response only - plain prose, no preamble, no markdown.
```

## `frames_v1` — Professional frames (F1)

Pooled prompt: *Describe the moment a crowded village market in medieval Italy is suddenly silenced by the arrival of a mysterious, cloaked traveler leading a decorated mule.*

### positive, `doctor__doctor`

```
Write one short, natural response an AI assistant would give to the request below, clearly weaving a doctor into it so that it features naturally.
Here a doctor means: the physician who sees and treats the sick person. Use it in its ordinary general medical practice sense -- the everyday working meaning the word has in that profession, not any other trade's use of it.
Use the word "doctor" itself somewhere in the response rather than only implying it, and let it really be there in what is described - part of the scene or of the answer itself, never a comparison or a simile for something else (no "just like", "much like", "as if", "think of it as"), and never something you the assistant are doing or pausing to do.
Avoid mentioning other people, places or documents from the same profession that are not a doctor, that includes both people, places or documents from the same profession that are closely related to a doctor, and those that are very different.
Any professional detail in the response must belong to general medical practice and nothing else - do not bring in another profession's people, places, paperwork or jargon. The setting itself stays whatever the request asks for.
Three or four sentences, no more than 70 words.

Request: Describe the moment a crowded village market in medieval Italy is suddenly silenced by the arrival of a mysterious, cloaked traveler leading a decorated mule.

Response only - plain prose, no preamble, no markdown.
```

### negative, shared by the whole `doctor` cloud

```
Write one short, natural response an AI assistant would give to the request below.
Do NOT mention medicine, doctors, patients, illness, clinics, treatment or prescriptions. Write about other ordinary places, objects, tools, actions and events instead. Keep the prose plain and concrete. Three or four sentences, no more than 70 words.

Request: Describe the moment a crowded village market in medieval Italy is suddenly silenced by the arrival of a mysterious, cloaked traveler leading a decorated mule.

Response only - plain prose, no preamble, no markdown.
```

