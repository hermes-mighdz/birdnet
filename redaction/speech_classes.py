NUM_YAMNET_CLASSES = 521

CORE_SPEECH = (
    0,   # Speech
    1,   # Child speech, kid speaking
    2,   # Conversation
    3,   # Narration, monologue
    4,   # Babbling
    5,   # Speech synthesizer
    6,   # Shout
    9,   # Yell
    10,  # Children shouting
    11,  # Screaming
    12,  # Whispering
    65,  # Hubbub, speech noise, speech babble
)

# index 247 ("Music for children") omitted here - it does not match any
# speech-related class in yamnet_class_map.csv, see conversation with Miguel
AMBIGUOUS = (
    29,  # Child singing
    63,  # Chatter
    64,  # Crowd
    66,  # Children playing
)


def speech_score(scores_1d, include_ambiguous: bool = False) -> float:
    if len(scores_1d) != NUM_YAMNET_CLASSES:
        raise ValueError(
            f"expected a {NUM_YAMNET_CLASSES}-element YAMNet score vector, got {len(scores_1d)}"
        )
    indices = CORE_SPEECH + AMBIGUOUS if include_ambiguous else CORE_SPEECH
    return max(scores_1d[i] for i in indices)
