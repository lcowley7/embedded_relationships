"""The four analogy datasets: concepts, mappings and generation text.

Each dataset is an experiment (A, B, T): two clouds of concepts and a fixed correspondence T
between them, written down before any model was run. The analyses read CLOUDS, EXPERIMENTS
and MOVE_ORDER; the generators (pipeline/generate/) read the rest. See docs/analogy_mappings.md.

  size_imp_v1    physical size        <-> importance          ordered, 9 <-> 9
  war_arg_v1     literal war          <-> argument            categorical, 8 <-> 8 plus 3 optional
  food_ideas_v1  literal food         <-> ideas               categorical, 12 <-> 12
  frames_v1      five professions x six roles (doctor, lawyer, dentist, agent, teacher)

How each vector is built (as for the water words, h2o_v1):
  pool      one neutral prompt pool per experiment, shared by both clouds, and neutral about
            both domains and about the structure under test.
  positive  a continuation of the same prompt in which the concept features. Two clauses are
            added: a SENSE LOCK (literal reading only) and a ban on the other cloud's
            vocabulary, so the clouds cannot share surface strings.
  negative  one continuation per prompt with the cloud's whole domain scrubbed, shared by every
            concept in that cloud. Each cloud has its own negative.
So v = mean(scene featuring the concept) - mean(same scene, domain absent).

Fields per concept: id (slug), desc (noun phrase), gloss (one-line literal definition),
lock (sense-lock clause), concept (the description handed to the judge).
"""
from __future__ import annotations

# Shared prompt-pool guidance (paper Appendix C).

_POOL_TAIL = (
    "One sentence each, self-contained (no files / images / links), nothing "
    "that presumes a previous conversation, history or task, not a yes/no "
    "question.\n"
    'Return "1." ... "{n}." only.'
)

# ============================================================================
# S3 -- IMPORTANCE IS SIZE  (ordered, rank-matched 9 <-> 9)
# ============================================================================

_S3_SIZE = [
    ("microscopic", "a microscopic object", "an object so physically small it can only be made out under a microscope"),
    ("tiny",        "a tiny object",        "an object of very small physical extent, easily lost in the hand"),
    ("small",       "a small object",       "an object of modest physical extent, comfortably held"),
    ("medium_sized","a medium-sized object","an object of moderate physical extent, neither cramped nor sprawling"),
    ("sizable",     "a sizable object",     "an object of appreciably large physical extent"),
    ("large",       "a large object",       "an object of large physical extent, taking up real room"),
    ("huge",        "a huge object",        "an object of very large physical extent, dwarfing what is around it"),
    ("enormous",    "an enormous object",   "an object of vast physical extent, hard to take in at once"),
    ("colossal",    "a colossal object",    "an object of overwhelming physical extent, on the scale of a mountain or a monument"),
]
_S3_IMP = [
    ("trivial",     "a trivial matter",     "a matter that carries no consequence at all"),
    ("negligible",  "a negligible matter",  "a matter whose consequence is too slight to be worth counting"),
    ("minor",       "a minor matter",       "a matter of slight consequence"),
    ("middling",    "a middling matter",    "a matter of moderate consequence, neither pressing nor idle"),
    ("noteworthy",  "a noteworthy matter",  "a matter with enough consequence to be worth remarking on"),
    ("significant", "a significant matter", "a matter that carries real consequence"),
    ("important",   "an important matter",  "a matter whose consequence makes it a priority"),
    ("crucial",     "a crucial matter",     "a matter on whose outcome much else depends"),
    ("momentous",   "a momentous matter",   "a matter whose consequence will be felt for a long time afterwards"),
]

_S3_SIZE_LOCK = (
    "Use it in the PHYSICAL sense only -- actual extent in space, something you "
    "could measure or stand next to. Never the figurative sense of how much "
    "something matters."
)
_S3_IMP_LOCK = (
    "Use it in the sense of how much the matter MATTERS -- its consequence, "
    "what turns on it. Never describe it in terms of physical size, weight, "
    "height or bulk."
)

# ============================================================================
# S7 -- ARGUMENT IS WAR  (categorical, 8 core + 3 optional)
# ============================================================================

_S7_WAR = [
    ("attack",     "a military attack",          "an armed assault by one force on another's position, launched to take ground or inflict losses"),
    ("defend",     "a military defence",         "a force holding a position against an assaulting enemy"),
    ("ally",       "a military ally",            "a force fighting alongside another under a shared cause"),
    ("strategy",   "a military strategy",        "the overall plan by which a commander seeks to win a campaign"),
    ("retreat",    "a military retreat",         "a force withdrawing from a position it can no longer hold"),
    ("surrender",  "a military surrender",       "a force laying down its arms and submitting to the opposing side"),
    ("shoot_down", "shooting down an aircraft",  "bringing an enemy aircraft down out of the sky with gunfire or a missile"),
    ("demolish",   "demolishing a structure",    "destroying a building, bridge or fortification outright by explosive or shellfire"),
    # optional rows 9-11
    ("weapon",     "a weapon",                   "an implement carried by a soldier or mounted on a vehicle for use in combat"),
    ("stronghold", "a stronghold",               "a fortified place a force holds and can fall back to"),
    ("ceasefire",  "a ceasefire",                "an agreed halt to the fighting between two forces"),
]
_S7_ARG = [
    ("criticise",  "criticising a claim",        "pointing out what is wrong or weak in a claim someone has made"),
    ("justify",    "justifying a claim",         "setting out the grounds that show one's own claim is warranted"),
    ("supporter",  "a supporter of a view",      "someone who publicly holds and backs the same view as another person"),
    ("reasoning",  "a line of reasoning",        "the connected sequence of steps by which someone gets to a conclusion"),
    ("back_down",  "backing down in a discussion","withdrawing a claim one had been maintaining"),
    ("concede",    "conceding a point",          "granting that the other side is right about something"),
    ("rebut",      "rebutting a claim",          "answering a claim by giving reasons it does not hold"),
    ("refute",     "refuting a claim",           "showing conclusively that a claim is false"),
    ("evidence",   "evidence for a claim",       "the facts or testimony brought forward to make a claim credible"),
    ("conviction", "a core conviction",          "a belief someone holds so firmly that they will not give it up"),
    ("agree_to_disagree", "agreeing to disagree","two people settling on leaving a disagreement unresolved"),
]

_S7_WAR_LOCK = (
    "Use it in the LITERAL MILITARY sense only -- real armed forces, real "
    "ground, real ordnance. Never the figurative sense in which people attack, "
    "defend or demolish an argument, an idea, a claim or a person's views."
)
_S7_ARG_LOCK = (
    "Use it in the sense of people REASONING WITH EACH OTHER -- a discussion, a "
    "review, a meeting, a disagreement worked through in words. Never a "
    "physical fight, and never described in the vocabulary of war."
)

# ============================================================================
# S8 -- IDEAS ARE FOOD  (categorical, 12 <-> 12)
# ============================================================================

_S8_FOOD = [
    ("cook",        "cooking a meal",              "preparing food by applying heat to it"),
    ("swallow",     "swallowing food",             "passing a mouthful of food from the mouth down the throat"),
    ("digest",      "digesting a meal",            "the body slowly breaking food down after it has been eaten"),
    ("chew",        "chewing food",                "working a mouthful of food between the teeth before swallowing"),
    ("taste",       "tasting food",                "putting a small amount of food on the tongue to judge its flavour"),
    ("spoon_feed",  "spoon-feeding someone",       "putting food into another person's mouth with a spoon, as one does for an infant"),
    ("regurgitate", "regurgitating food",          "bringing food back up out of the stomach undigested"),
    ("recipe",      "a recipe",                    "written instructions for preparing a particular dish"),
    ("ingredient",  "an ingredient",               "one of the foodstuffs that goes into a dish"),
    ("stale",       "stale food",                  "food that has gone dry and old and is past being good to eat"),
    ("fresh",       "fresh food",                  "food newly picked, caught or made, not yet gone off"),
    ("appetite",    "an appetite for food",        "the bodily hunger that makes someone want to eat"),
]
_S8_IDEAS = [
    ("devise",      "devising an idea",            "working out a new idea or plan for the first time"),
    ("accept",      "accepting a claim",           "coming to hold that a claim someone has made is true"),
    ("comprehend",  "comprehending an explanation","coming to fully understand something that has been explained"),
    ("ponder",      "pondering a question",        "turning a question over in the mind at length before answering it"),
    ("consider",    "considering a proposal",      "giving a proposal deliberate attention before deciding on it"),
    ("oversimplify","oversimplifying an explanation","making an explanation so simple that what matters is lost"),
    ("recite",      "reciting from memory",        "repeating memorised material back word for word without understanding it"),
    ("method",      "a method",                    "a settled way of going about a task"),
    ("component",   "a component of a theory",     "one of the parts a theory is built out of"),
    ("unoriginal",  "an unoriginal idea",          "an idea that many people have already had before"),
    ("novel",       "a novel idea",                "an idea nobody appears to have had before"),
    ("curiosity",   "curiosity",                   "the wish to find out how something is or why it is so"),
]

_S8_FOOD_LOCK = (
    "Use it in the LITERAL culinary or bodily sense only -- actual food, "
    "actual kitchens, actual eating. Never the figurative sense in which "
    "people cook up, swallow, chew over, digest or spoon-feed an idea, a claim "
    "or a piece of information."
)
_S8_IDEAS_LOCK = (
    "Use it in the sense of THINKING, LEARNING or DISCUSSING -- what goes on in "
    "someone's mind or between people working something out. Never described "
    "with food, eating, cooking or taste imagery of any kind."
)

# ============================================================================
# F1 -- Professional frames  (5 frames x 6 roles + optional)
# Table is [role][frame]; Australian usage.
# ============================================================================

F1_FRAMES = ["doctor", "lawyer", "dentist", "agent", "teacher"]
F1_ROLES  = ["PRACTITIONER", "RECIPIENT", "VENUE", "MATTER", "INTERVENTION", "ARTIFACT"]
F1_ROLES_OPTIONAL = ["OUTCOME"]

F1_TABLE = {
    "PRACTITIONER": {
        "doctor":  ("doctor",            "a doctor",             "the physician who sees and treats the sick person"),
        "lawyer":  ("lawyer",            "a lawyer",             "the legal professional who acts for someone in a legal matter"),
        "dentist": ("dentist",           "a dentist",            "the practitioner who treats the teeth"),
        "agent":   ("real_estate_agent", "a real-estate agent",  "the agent who acts for someone selling a property"),
        "teacher": ("teacher",           "a teacher",            "the person who instructs a class of learners"),
    },
    "RECIPIENT": {
        "doctor":  ("patient",       "a patient",       "the sick person the doctor is treating"),
        "lawyer":  ("client",        "a client",        "the person whose legal matter the lawyer is acting on"),
        "dentist": ("dental_patient","a dental patient","the person in the chair having their teeth treated"),
        "agent":   ("vendor",        "a vendor",        "the owner selling the property, on whose behalf the agent acts"),
        "teacher": ("student",       "a student",       "the learner being taught"),
    },
    "VENUE": {
        "doctor":  ("clinic",          "a clinic",             "the rooms where a doctor sees patients"),
        "lawyer":  ("law_firm",        "a law firm",           "the offices where lawyers practise"),
        "dentist": ("dental_surgery",  "a dental surgery",     "the rooms where a dentist treats teeth"),
        "agent":   ("agency",          "a real-estate agency", "the office from which properties are marketed and sold"),
        "teacher": ("school",          "a school",             "the building where classes are taught"),
    },
    "MATTER": {
        "doctor":  ("illness",       "an illness",       "the disease the patient has come about"),
        "lawyer":  ("lawsuit",       "a lawsuit",        "the legal action being brought or answered"),
        "dentist": ("toothache",     "a toothache",      "the pain in a tooth that brought the person in"),
        "agent":   ("property_sale", "a property sale",  "the sale of the house that is in hand"),
        "teacher": ("lesson",        "a lesson",         "the material being taught in a given class"),
    },
    "INTERVENTION": {
        "doctor":  ("treatment",      "a course of treatment", "what the doctor does to the illness"),
        "lawyer":  ("representation", "legal representation",   "the lawyer acting for the client in the matter"),
        "dentist": ("filling",        "a filling",              "the restoration a dentist puts into a decayed tooth"),
        "agent":   ("auction",        "an auction",             "the public sale at which the property is sold"),
        "teacher": ("instruction",    "instruction",            "the teaching itself, given to the class"),
    },
    "ARTIFACT": {
        "doctor":  ("prescription", "a prescription", "the written order for medicine the doctor hands over"),
        "lawyer":  ("contract",     "a contract",     "the written agreement the lawyer draws up"),
        "dentist": ("dental_chart", "a dental chart", "the record of a person's teeth kept by the dentist"),
        "agent":   ("listing",      "a listing",      "the written advertisement of a property for sale"),
        "teacher": ("report_card",  "a report card",  "the written report of a student's progress"),
    },
    "OUTCOME": {
        "doctor":  ("recovery",   "a recovery",   "the patient getting well again"),
        "lawyer":  ("verdict",    "a verdict",    "the court's decision in the matter"),
        "dentist": (None, None, None),   # no clean filler
        "agent":   ("sold_home",  "a sold home",  "the property having changed hands"),
        "teacher": ("graduation", "a graduation", "the student completing their course of study"),
    },
}

F1_FRAME_LOCK = (
    "Use it in its ordinary {frame_name} sense -- the everyday working meaning "
    "the word has in that profession, not any other trade's use of it."
)
F1_FRAME_NAME = {
    "doctor":  "general medical practice",
    "lawyer":  "legal practice",
    "dentist": "dentistry",
    "agent":   "residential real-estate practice",
    "teacher": "school teaching",
}

# ============================================================================
# Cross-domain vocabulary
#   AVOID_CLAUSE  goes into the positive prompt as {avoid}: stay off the OTHER cloud's words.
#   BAN           full word lists for the automated leak check over generated text.
#   BAN_SOFT      amphibious or generic words, reported but never instructed on.
#   PROBES        figurative phrases held out of every dataset.
# ============================================================================

AVOID_CLAUSE = {
    "s3_size": (   # in the IMPORTANCE cloud
        "Do not describe the matter, or anything else in the scene, in terms of "
        "physical size, weight or bulk - nothing big, large, small, tiny, huge, "
        "enormous, immense, massive, vast, towering or weighty."
    ),
    "s3_importance": (   # in the SIZE cloud
        "Do not say whether the thing matters - nothing important, significant, "
        "crucial, trivial, negligible, minor, momentous or consequential, and do "
        "not say what turns on it."
    ),
    "s7_argument": (   # in the WAR cloud
        "Keep it a physical military scene: do not have anyone argue, debate, "
        "make a claim, offer evidence, criticise, justify, rebut, refute or concede."
    ),
    "s7_war": (   # in the ARGUMENT cloud
        "Do not use any war or fighting imagery - nothing about attacking, "
        "defending, weapons, battles, armies, enemies, retreating, surrendering, "
        "demolishing or shooting anything down."
    ),
    "s8_ideas": (   # in the FOOD cloud
        "Keep it a physical food scene: do not have anyone think about, "
        "understand, ponder, consider or learn anything, and do not mention "
        "ideas, theories or knowledge."
    ),
    "s8_food": (   # in the IDEAS cloud
        "Do not use any food, eating or cooking imagery - nothing raw, "
        "half-baked, stale, fresh, tasty, hard to swallow, chewed over, "
        "digested, spoon-fed or unpalatable."
    ),
}

BAN = {
    "s3_size": [
        "size", "sized", "big", "bigger", "biggest", "large", "larger", "small",
        "smaller", "tiny", "huge", "enormous", "colossal", "gigantic", "giant",
        "massive", "vast", "immense", "microscopic", "minuscule",
        "sizable", "sizeable", "dimension", "bulk", "towering", "miniature",
        "outsized", "hulking", "diminutive",
    ],
    "s3_importance": [
        "important", "importance", "significant", "significance", "crucial",
        "vital", "trivial", "negligible", "momentous", "noteworthy",
        "consequential", "pivotal", "paramount", "insignificant", "trifling",
        "middling", "stakes",
    ],
    "s7_war": [
        "war", "warfare", "battle", "battlefield", "combat", "army", "armies",
        "soldier", "troop", "troops", "military", "weapon", "rifle", "artillery",
        "bomb", "bombard", "siege", "trench", "enemy", "invade", "invasion",
        "assault", "attack", "attacked", "ambush", "flank", "retreat",
        "surrender", "ceasefire", "armistice", "fortify", "fortress",
        "stronghold", "missile", "regiment", "platoon", "arsenal", "ammunition",
        "offensive", "defend", "defence", "defense", "indefensible", "conquer",
        "warrior", "militia", "skirmish", "barrage", "salvo", "demolish",
        "demolition",
        "entrenched", "besiege", "firepower",
    ],
    "s7_argument": [
        "argue", "argued", "argument", "debate", "debated", "claim", "thesis",
        "premise", "evidence", "reasoning", "rebut", "refute", "concede",
        "conceded", "justify", "justification", "criticise", "criticize",
        "criticism", "critique", "objection", "counterargument", "persuade",
        "convince", "disagree", "disagreement", "assert", "contend", "fallacy",
        "proposition", "viewpoint",
    ],
    "s8_food": [
        "food", "eat", "eaten", "eating", "meal", "kitchen", "cook", "cooked",
        "cooking", "chef", "recipe", "ingredient", "dish", "taste", "tasted",
        "tasty", "flavour", "flavor", "swallow", "chew", "chewed", "digest",
        "digested", "digestion", "appetite", "hungry", "hunger", "mouthful",
        "stale", "raw", "baked", "half-baked", "simmer", "stew", "spoon-feed",
        "regurgitate", "nourish", "palate", "savour", "savor", "devour",
        "unpalatable", "meaty", "bitter",
    ],
    "s8_ideas": [
        "idea", "ideas", "concept", "theory", "comprehend", "ponder",
        "unoriginal", "curiosity", "knowledge", "insight", "notion",
        "hypothesis", "oversimplify", "ruminate", "intellect", "cognition",
    ],
}

BAN_SOFT = {
    "s3_size":       ["little", "weight", "weighty", "heavy", "heft", "loom", "looming",
                      "magnitude", "scale", "great", "substantial", "considerable"],
    "s3_importance": ["matter", "matters", "mattered", "consequence", "minor",
                      "major", "critical", "essential", "serious", "weight"],
    "s7_war":        ["position", "ground", "front", "general", "strike", "target",
                      "campaign", "defeat", "victory", "crush", "shoot", "fire",
                      "casualty", "shell", "tank", "commander", "gun"],
    "s7_argument":   ["discussion", "point", "case", "logic", "stance", "dispute",
                      "support", "supporter", "grounds", "concession"],
    "s8_food":       ["fresh", "sweet", "sour", "juicy", "feed", "bite", "plate",
                      "snack", "bread", "soup", "breakfast", "lunch", "dinner"],
    "s8_ideas":      ["think", "thinking", "thought", "understand", "understanding",
                      "consider", "reason", "method", "component", "novel", "learn",
                      "learning", "mind", "mental", "grasp", "recite", "memorise"],
}

PROBES = {
    "size_imp_v1":   ["a big decision", "a little problem", "a weighty question",
                      "a monumental error", "a giant of the field", "small talk",
                      "a towering achievement", "a slight oversight"],
    "war_arg_v1":    ["attack her argument", "defend a thesis", "an indefensible claim",
                      "shot down my proposal", "an entrenched position",
                      "a battle of ideas", "a war of words", "demolish an argument"],
    "food_ideas_v1": ["a half-baked idea", "food for thought", "hard to swallow",
                      "stew over it", "a warmed-over theory", "devour a book",
                      "spoon-fed answers", "leaves a bad taste"],
    "frames_v1":     ["a practice", "a case", "a settlement", "a retainer",
                      "a consultation", "a referral"],
}


# ============================================================================
# Prompt pools: one per experiment, shared by both clouds. Each asks for ordinary
# writing requests and forbids both domains, and must not itself set up the
# structure under test (e.g. no "two opposing sides" for war/argument).
# ============================================================================

POOLS = {
    "size_imp_v1": (
        'Write {n} short, varied requests a user might send an AI assistant asking for a '
        'brief piece of writing - narrating an episode, recounting how something '
        'unfolded, describing a situation or a place, or inventing a short scene.\n'
        'Vary the settings very widely (at home, at work, outdoors, a journey, a shop, a workshop, a garden, an office, a public occasion, a small town, a historical period, an invented world, or others) and vary the mode '
        '(narrate, recount, describe, invent).\n'
        'One sentence each, self-contained (no files / images / links), nothing that '
        'presumes a previous conversation, history or task, not a yes/no question. Each '
        'request must be about people, places or events so that a short scene can be '
        'written in reply - NOT step-by-step instructions addressed to the reader. '
        'Do NOT mention how big or small anything is, and do NOT say whether anything is important, serious, trivial or consequential.\n'
        'Return "1." ... "{n}." only.'
    ),
    "war_arg_v1": (
        'Write {n} short, varied requests a user might send an AI assistant asking for a '
        'brief piece of writing - narrating an episode, recounting how something '
        'unfolded, describing a situation or a place, or inventing a short scene.\n'
        'Vary the settings very widely (at home, at work, outdoors, a journey, a public occasion, an organisation, a small town, a historical period, an invented world, a workshop, a garden, or others) and vary the mode '
        '(narrate, recount, describe, invent).\n'
        'One sentence each, self-contained (no files / images / links), nothing that '
        'presumes a previous conversation, history or task, not a yes/no question. Each '
        'request must be about people, places or events so that a short scene can be '
        'written in reply - NOT step-by-step instructions addressed to the reader. '
        'Do NOT mention fighting, weapons, armies or violence, and do NOT mention arguing, debating, disagreeing or persuading; do not set up two opposing sides of anything.\n'
        'Return "1." ... "{n}." only.'
    ),
    "food_ideas_v1": (
        'Write {n} short, varied requests a user might send an AI assistant asking for a '
        'brief piece of writing - narrating an episode, recounting how something '
        'unfolded, describing a situation or a place, or inventing a short scene.\n'
        'Vary the settings very widely (at home, at work, outdoors, a journey, a shop, a workshop, a garden, a public occasion, a small town, a historical period, an invented world, or others) and vary the mode '
        '(narrate, recount, describe, invent).\n'
        'One sentence each, self-contained (no files / images / links), nothing that '
        'presumes a previous conversation, history or task, not a yes/no question. Each '
        'request must be about people, places or events so that a short scene can be '
        'written in reply - NOT step-by-step instructions addressed to the reader. '
        'Do NOT mention food, eating, cooking or kitchens, and do NOT mention thinking, ideas, learning or understanding.\n'
        'Return "1." ... "{n}." only.'
    ),
    "frames_v1": (
        'Write {n} short, varied requests a user might send an AI assistant asking for a '
        'brief piece of writing - narrating an episode, recounting how something '
        'unfolded, describing a situation or a place, or inventing a short scene.\n'
        'Vary the settings very widely (at home, outdoors, a journey, a shop, a workshop, a garden, an errand, a public occasion, a small town, a historical period, an invented world, or others) and vary the mode '
        '(narrate, recount, describe, invent).\n'
        'One sentence each, self-contained (no files / images / links), nothing that '
        'presumes a previous conversation, history or task, not a yes/no question. Each '
        'request must be about people, places or events so that a short scene can be '
        'written in reply - NOT step-by-step instructions addressed to the reader; most '
        'should involve more than one person, in a public or shared place. '
        'Do NOT mention any profession or professional service, and do NOT mention medicine, law, teeth, property, schooling, appointments or paperwork.\n'
        'Return "1." ... "{n}." only.'
    ),
}

# ============================================================================
# Negative ("absent") scrub clauses: one per cloud, removing that cloud's own
# domain. Generated once per prompt and shared by every concept in the cloud.
# ============================================================================

SCRUB = {
    "size": (
        "Do NOT say how big or small anything is - no size, dimensions, height, "
        "width, bulk or scale, and none of the words big, large, small, tiny, "
        "huge or their kin. Write about what things are and what happens to "
        "them instead. Keep the prose plain and concrete."
    ),
    "importance": (
        "Do NOT say whether anything matters - nothing about importance, "
        "significance, consequence, stakes, priority or triviality. Just recount "
        "what happens, without weighing it. Keep the prose plain and concrete."
    ),
    "war": (
        "Do NOT mention armed conflict in any form - no war, battle, soldiers, "
        "army, weapons, attack, defence, retreat, surrender, siege, destruction "
        "or an enemy of any kind. Write about ordinary places, objects, tools, "
        "actions and events instead. Keep the prose plain and concrete."
    ),
    "argument": (
        "Do NOT mention disagreement or persuasion in any form - no argument, "
        "debate, claim, criticism, objection, justification, evidence, "
        "conceding or changing anyone's mind. Write about ordinary places, "
        "objects, tools, actions and events instead. Keep the prose plain and "
        "concrete."
    ),
    "food": (
        "Do NOT mention food in any form - no eating, meals, cooking, kitchens, "
        "ingredients, recipes, taste, hunger, swallowing, chewing or digestion. "
        "Write about other ordinary places, objects, tools, actions and events "
        "instead. Keep the prose plain and concrete."
    ),
    "ideas": (
        "Do NOT mention thinking or ideas in any form - nothing about thoughts, "
        "concepts, theories, understanding, considering, learning, knowledge, "
        "reasoning or curiosity. Just recount what is done and what happens. "
        "Keep the prose plain and concrete."
    ),
    # F1: one scrub per frame, scrubbing that frame's own professional domain.
    "doctor":  ("Do NOT mention medicine, doctors, patients, illness, clinics, treatment "
                "or prescriptions. Write about other ordinary places, objects, tools, "
                "actions and events instead. Keep the prose plain and concrete."),
    "lawyer":  ("Do NOT mention the law, lawyers, clients, lawsuits, courts, legal "
                "representation or contracts. Write about other ordinary places, objects, "
                "tools, actions and events instead. Keep the prose plain and concrete."),
    "dentist": ("Do NOT mention dentistry, dentists, teeth, toothache, dental surgeries, "
                "fillings or dental records. Write about other ordinary places, objects, "
                "tools, actions and events instead. Keep the prose plain and concrete."),
    "agent":   ("Do NOT mention real estate, agents, vendors, property sales, agencies, "
                "auctions or listings. Write about other ordinary places, objects, tools, "
                "actions and events instead. Keep the prose plain and concrete."),
    "teacher": ("Do NOT mention teaching, teachers, students, schools, lessons, "
                "instruction or report cards. Write about other ordinary places, objects, "
                "tools, actions and events instead. Keep the prose plain and concrete."),
}


# CLOUD_KIND fills {kind} in the positive template's soft co-mention line
# ("Avoid mentioning other {kind} that are not {desc}").
CLOUD_KIND = {
    "size":       "ways of saying how large or small a thing is",
    "importance": "ways of saying how much a thing matters",
    "war":        "military actions or pieces of military equipment",
    "argument":   "moves someone makes in a discussion",
    "food":       "foods, or things done to food",
    "ideas":      "ways of thinking about or handling an idea",
}
CLOUD_KIND.update({f: "people, places or documents from the same profession"
                   for f in F1_FRAMES})


# ============================================================================
# Assembly
# ============================================================================

def _mk(items, lock, avoid_key, cloud, domain, axis, judge_fmt):
    """Turn a (id, desc, gloss) list into full concept records."""
    out = []
    for i, (cid, desc, gloss) in enumerate(items):
        out.append(dict(
            id=cid, rank=i, desc=desc, gloss=gloss, lock=lock, cloud=cloud,
            domain=domain, magnitude_axis=axis,
            # the judge reads this description, so the sense lock rides in it
            concept=judge_fmt.format(desc=desc, gloss=gloss),
            vector_id=f"{cloud}__{cid}",
            avoid_key=avoid_key,
        ))
    return out


CLOUDS = {}

CLOUDS["size"] = _mk(
    _S3_SIZE, _S3_SIZE_LOCK, "s3_importance", "size", "physical_size", "size_rank",
    "the response features {desc} - {gloss} - in the literal physical sense")
CLOUDS["importance"] = _mk(
    _S3_IMP, _S3_IMP_LOCK, "s3_size", "importance", "importance", "importance_rank",
    "the response features {desc} - {gloss} - and conveys how much it matters, "
    "without any physical-size imagery")

CLOUDS["war"] = _mk(
    _S7_WAR, _S7_WAR_LOCK, "s7_argument", "war", "literal_war", "war_role",
    "the response features {desc} - {gloss} - in the literal military sense, "
    "not as a figure of speech about an argument")
CLOUDS["argument"] = _mk(
    _S7_ARG, _S7_ARG_LOCK, "s7_war", "argument", "argumentation", "argument_role",
    "the response features {desc} - {gloss} - in a discussion between people, "
    "described without any war or fighting imagery")

CLOUDS["food"] = _mk(
    _S8_FOOD, _S8_FOOD_LOCK, "s8_ideas", "food", "literal_food", "food_role",
    "the response features {desc} - {gloss} - in the literal culinary or bodily "
    "sense, not as a figure of speech about ideas")
CLOUDS["ideas"] = _mk(
    _S8_IDEAS, _S8_IDEAS_LOCK, "s8_food", "ideas", "cognition", "idea_role",
    "the response features {desc} - {gloss} - as something going on in someone's "
    "mind or between people working something out, described without any food imagery")

# F1: one cloud per frame; the role is the index that lines the clouds up.
for _f in F1_FRAMES:
    _items, _roles = [], []
    for _r in F1_ROLES:
        _cid, _desc, _gloss = F1_TABLE[_r][_f]
        if _cid is None:
            continue
        _items.append((_cid, _desc, _gloss))
        _roles.append(_r)
    _recs = _mk(_items, F1_FRAME_LOCK.format(frame_name=F1_FRAME_NAME[_f]),
                None, _f, "professional_frame", "frame_role",
                "the response features {desc} - {gloss}")
    for _rec, _r in zip(_recs, _roles):
        _rec["role"] = _r
        _rec["frame"] = _f
    CLOUDS[_f] = _recs

# ---- experiments: (A, B, T) ---------------------------------------------- #
# T is the correspondence the permutation null reshuffles. The generator never
# sees it: each cloud's text is written in separate API calls.

EXPERIMENTS = {
    "size_imp_v1": dict(
        A="size", B="importance", shape="ordered", n_core=9,
        # rank-matched bijection
        T=[(a["id"], b["id"]) for a, b in zip(CLOUDS["size"], CLOUDS["importance"])],
    ),
    "war_arg_v1": dict(
        A="war", B="argument", shape="categorical", n_core=8,
        T=[(a["id"], b["id"]) for a, b in zip(CLOUDS["war"], CLOUDS["argument"])],
    ),
    "food_ideas_v1": dict(
        A="food", B="ideas", shape="categorical", n_core=12,
        T=[(a["id"], b["id"]) for a, b in zip(CLOUDS["food"], CLOUDS["ideas"])],
    ),
    "frames_v1": dict(
        A=None, B=None, shape="multi_frame", n_core=6, frames=F1_FRAMES,
        # F-a: every frame pair, indexed by role. F-b is the same vectors resliced.
        T=[(r, r) for r in F1_ROLES],
    ),
}

# Which cloud(s) each experiment's ONE shared pool serves.
EXPERIMENT_CLOUDS = {
    "size_imp_v1":   ["size", "importance"],
    "war_arg_v1":    ["war", "argument"],
    "food_ideas_v1": ["food", "ideas"],
    "frames_v1":     list(F1_FRAMES),
}

# Moves run from lower to higher index in the order the lists above are written,
# on both sides of T (rank order for size_imp_v1; fixed but arbitrary otherwise).
MOVE_ORDER = {c: [r["id"] for r in recs] for c, recs in CLOUDS.items()}


# ---- F1 leak check: each frame's ban list is the other four frames' words ---- #
_F1_DOMAIN_WORDS = {
    "doctor":  ["doctor", "physician", "patient", "illness", "clinic", "treatment",
                "prescription", "medicine", "medical", "nurse", "diagnosis", "symptom"],
    "lawyer":  ["lawyer", "solicitor", "barrister", "client", "lawsuit", "law firm",
                "representation", "contract", "legal", "court", "litigation", "statute"],
    "dentist": ["dentist", "dental", "toothache", "tooth", "teeth", "filling",
                "surgery", "molar", "cavity", "enamel"],
    "agent":   ["agent", "vendor", "agency", "property", "auction", "listing",
                "real estate", "real-estate", "buyer", "conveyancing", "mortgage"],
    "teacher": ["teacher", "student", "school", "lesson", "instruction",
                "report card", "classroom", "pupil", "curriculum", "homework"],
}
def _f1_ban(f):
    """Other frames' vocabulary, minus anything this frame owns ("patient" is both doctor's
    and dentist's, so it is banned in neither)."""
    own = " ".join(f"{r['desc']} {r['gloss']}" for r in CLOUDS[f]).lower()
    return sorted({w for g, ws in _F1_DOMAIN_WORDS.items() if g != f for w in ws
                   if w not in own})

F1_BAN = {f: _f1_ban(f) for f in F1_FRAMES}

for _f in F1_FRAMES:
    for _rec in CLOUDS[_f]:
        _rec["avoid_key"] = f"f1_{_f}"
BAN.update({f"f1_{f}": F1_BAN[f] for f in F1_FRAMES})
BAN_SOFT.update({f"f1_{f}": ["practice", "case", "settlement", "retainer",
                             "consultation", "referral", "appointment", "fee"]
                 for f in F1_FRAMES})
# The clause binds the professional detail, not the setting (a whole-scene wording was refused).
AVOID_CLAUSE.update({
    f"f1_{f}": ("Any professional detail in the response must belong to "
                f"{F1_FRAME_NAME[f]} and nothing else - do not bring in another "
                "profession's people, places, paperwork or jargon. The setting itself "
                "stays whatever the request asks for.")
    for f in F1_FRAMES})


# ============================================================================
# Vocabulary handles for the AUTOMATED checks (never sent to the generator)
# ============================================================================

# A cloud's own domain words: its negative, and its prompt pool, must be clean of them.
OWN_BAN = {
    "size":       BAN["s3_size"],
    "importance": BAN["s3_importance"],
    "war":        BAN["s7_war"],
    "argument":   BAN["s7_argument"],
    "food":       BAN["s8_food"],
    "ideas":      BAN["s8_ideas"],
}
OWN_BAN.update({f: _F1_DOMAIN_WORDS[f] for f in F1_FRAMES})

# Too generic or genuinely shared to count as a cross-frame leak ("surgery" is also a GP's
# rooms in Australian usage); moved from the hard list to the soft one.
_F1_BAN_EXEMPT = {
    "doctor":  {"surgery", "instruction"},
    "lawyer":  {"instruction"},
    "dentist": {"instruction"},
    "agent":   {"instruction"},
    "teacher": {"surgery"},
}
for _f in F1_FRAMES:
    F1_BAN[_f] = [w for w in F1_BAN[_f] if w not in _F1_BAN_EXEMPT.get(_f, set())]
    BAN[f"f1_{_f}"] = F1_BAN[_f]
    BAN_SOFT[f"f1_{_f}"] = sorted(set(BAN_SOFT[f"f1_{_f}"]) | _F1_BAN_EXEMPT.get(_f, set()))


# ---- surface forms of a concept, for the co-mention matrix ---------------- #
# Morphological variants only (no synonyms): "did concept a's text say concept b's word?"
_TERM_OVERRIDES = {
    "shoot_down":  ["shoot down", "shoots down", "shot down", "shooting down"],
    "back_down":   ["back down", "backs down", "backed down", "backing down"],
    "spoon_feed":  ["spoon-feed", "spoon-feeds", "spoon-fed", "spoon-feeding",
                    "spoon feed", "spoon fed"],
    "agree_to_disagree": ["agree to disagree", "agreeing to disagree",
                          "agreed to disagree"],
    "medium_sized": ["medium-sized", "medium sized"],
    "demolish":    ["demolish", "demolishes", "demolished", "demolishing",
                    "demolition"],
    "defend":      ["defend", "defends", "defended", "defending", "defence",
                    "defense"],
    "ally":        ["ally", "allies", "allied"],
    "strategy":    ["strategy", "strategies", "strategic"],
    "criticise":   ["criticise", "criticises", "criticised", "criticising",
                    "criticize", "criticizes", "criticized", "criticizing",
                    "criticism", "critique"],
    "justify":     ["justify", "justifies", "justified", "justifying",
                    "justification"],
    "reasoning":   ["reasoning", "line of reasoning", "reason through"],
    "evidence":    ["evidence"],
    "conviction":  ["conviction", "convictions"],
    "curiosity":   ["curiosity", "curious"],
    "appetite":    ["appetite", "appetites"],
    "recipe":      ["recipe", "recipes"],
    "regurgitate": ["regurgitate", "regurgitates", "regurgitated",
                    "regurgitating", "regurgitation"],
    "comprehend":  ["comprehend", "comprehends", "comprehended",
                    "comprehending", "comprehension"],
    "oversimplify": ["oversimplify", "oversimplifies", "oversimplified",
                     "oversimplifying", "oversimplification"],
    "unoriginal":  ["unoriginal"],
    "novel":       ["novel"],
    "fresh":       ["fresh", "freshly", "freshness"],
    "stale":       ["stale"],
    "real_estate_agent": ["real-estate agent", "real estate agent", "agent",
                          "agents"],
    "dental_patient":    ["patient", "patients"],
    "dental_surgery":    ["dental surgery", "surgery"],
    "dental_chart":      ["dental chart", "chart", "charts"],
    "law_firm":          ["law firm", "firm"],
    "property_sale":     ["property sale", "sale", "sales"],
    "report_card":       ["report card", "report cards"],
    "sold_home":         ["sold", "sale of the home"],
    "agency":            ["agency", "agencies"],
    "representation":    ["representation", "represent", "represents",
                          "represented", "representing"],
}


def _variants(w):
    """Naive English inflections of a single lower-case word."""
    v = {w, w + "s"}
    if w.endswith("e"):
        v |= {w + "d", w[:-1] + "ing"}
    elif len(w) > 2 and w[-1] == "y" and w[-2] not in "aeiou":
        v |= {w[:-1] + "ies", w[:-1] + "ied"}
    elif w.endswith(("s", "sh", "ch", "x")):
        v |= {w + "es", w + "ed", w + "ing"}
    else:
        v |= {w + "ed", w + "ing"}
    return sorted(v)


def concept_terms(rec):
    """Surface strings that count as 'this concept was mentioned'."""
    if rec["id"] in _TERM_OVERRIDES:
        return _TERM_OVERRIDES[rec["id"]]
    return _variants(rec["id"].replace("_", " "))


# ============================================================================
# Surface form and ladder position. On the size/importance ladders the generator
# tended to convey a rung without its word, and to collapse neighbouring rungs
# onto one intensifier ("sizable" and "colossal" both became "massive"). So the
# positive prompt asks for the concept's own word and names its neighbours.
# ============================================================================

ORDERED_CLOUDS = {"size", "importance"}


def head_term(rec):
    """The concept's own word, as it should appear in the prose (citation form)."""
    if rec["id"] in _TERM_OVERRIDES:
        return _TERM_OVERRIDES[rec["id"]][0]
    return rec["id"].replace("_", " ")


def scale_note(rec):
    """For a ladder rung, a sentence naming its immediate neighbours; empty otherwise."""
    if rec["cloud"] not in ORDERED_CLOUDS:
        return ""
    sibs = CLOUDS[rec["cloud"]]
    i = rec["rank"]
    near = [f'"{head_term(sibs[j])}"' for j in (i - 1, i + 1) if 0 <= j < len(sibs)]
    if len(near) == 2:
        return (f" Its nearest neighbours on this scale are {near[0]} and "
                f"{near[1]}, which are close to it but not the same.")
    return (f" Its nearest neighbour on this scale is {near[0]}, which is close "
            f"to it but not the same.")


def summary():
    lines = []
    for ds, spec in EXPERIMENTS.items():
        cl = EXPERIMENT_CLOUDS[ds]
        n = sum(len(CLOUDS[c]) for c in cl)
        lines.append(f"{ds:16s} clouds={','.join(cl):45s} vectors={n:3d}  shape={spec['shape']}")
    lines.append(f"{'TOTAL':16s} {'':45s} vectors={sum(len(CLOUDS[c]) for ds in EXPERIMENTS for c in EXPERIMENT_CLOUDS[ds]):3d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
