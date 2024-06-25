PHONE_DEF = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
PHONE_DEF_SIL = PHONE_DEF + ["SIL"]


def phone_to_id(p: str):
    return PHONE_DEF_SIL.index(p)


def id_to_phone(idx: int):
    assert idx in range(0, len(PHONE_DEF_SIL))
    return PHONE_DEF_SIL[idx]
