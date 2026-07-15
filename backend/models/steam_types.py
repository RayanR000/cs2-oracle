"""
Parse Steam's combined `type` field into structured supply-side metadata.

Steam Market item types encode rarity, weapon category, souvenir/stattrak
flags, and item class in a single string. Examples:

    "Classified Rifle"              → rarity=classified,  weapon_type=rifle
    "Mil-Spec Grade Pistol"         → rarity=milspec,     weapon_type=pistol
    "★ Covert Knife"                → rarity=covert,      weapon_type=knife,   is_knife=True
    "Souvenir Restricted SMG"       → rarity=restricted,  weapon_type=smg,     is_souvenir=True
    "StatTrak™ Covert Rifle"        → rarity=covert,      weapon_type=rifle,   is_stattrak=True
    "★ StatTrak™ Covert Knife"      → rarity=covert,      weapon_type=knife,   is_knife=True, is_stattrak=True
    "Base Grade Container"          → rarity=base,         weapon_type=case
    "Extraordinary Sticker"         → rarity=extraordinary, weapon_type=sticker
    "Customized High Grade Charm"   → rarity=high_grade,   weapon_type=charm
    "Highlight Base Grade Container" → rarity=highlight,   weapon_type=case
"""

RARITY_KEYWORDS = [
    ("base grade", "base"),
    ("consumer grade", "consumer"),
    ("industrial grade", "industrial"),
    ("mil-spec grade", "milspec"),
    ("restricted", "restricted"),
    ("classified", "classified"),
    ("covert", "covert"),
    ("extraordinary", "extraordinary"),
    ("exotic", "exotic"),
    ("remarkable", "remarkable"),
    ("high grade", "high_grade"),
    ("distinguished", "distinguished"),
    ("exceptional", "exceptional"),
    ("superior", "superior"),
    ("master", "master"),
    ("highlight", "highlight"),
]

RARITY_RANK = {
    "base": 0,
    "consumer": 1,
    "industrial": 2,
    "milspec": 3,
    "restricted": 4,
    "classified": 5,
    "covert": 6,
    "high_grade": 3,
    "remarkable": 4,
    "exotic": 5,
    "extraordinary": 6,
    "distinguished": 3,
    "exceptional": 4,
    "superior": 5,
    "master": 6,
    "highlight": 0,
}

WEAPON_KEYWORDS = [
    ("sniper rifle", "sniper"),
    ("music kit", "musickit"),
    ("machinegun", "machinegun"),
    ("shotgun", "shotgun"),
    ("pistol", "pistol"),
    ("rifle", "rifle"),
    ("smg", "smg"),
    ("knife", "knife"),
    ("gloves", "glove"),
    ("container", "case"),
    ("equipment", "equipment"),
    ("sticker", "sticker"),
    ("graffiti", "graffiti"),
    ("charm", "charm"),
    ("collectible", "collectible"),
    ("patch", "patch"),
    ("agent", "agent"),
    ("key", "key"),
    ("pass", "pass"),
    ("tool", "tool"),
    ("tag", "tag"),
    ("gift", "gift"),
]

WEAPON_TYPE_ORDER = {
    "musickit": 0,
    "sticker": 1,
    "patch": 2,
    "charm": 3,
    "graffiti": 4,
    "collectible": 5,
    "agent": 6,
    "key": 7,
    "pass": 8,
    "tool": 9,
    "tag": 10,
    "gift": 11,
    "case": 12,
    "equipment": 13,
    "pistol": 14,
    "smg": 15,
    "shotgun": 16,
    "machinegun": 17,
    "rifle": 18,
    "sniper": 19,
    "knife": 20,
    "glove": 21,
    "other": 22,
}


def parse_steam_type(raw: str) -> dict:
    if not raw or not isinstance(raw, str):
        return {
            "is_souvenir": False,
            "is_stattrak": False,
            "is_knife": False,
            "is_glove": False,
            "rarity": None,
            "rarity_rank": 0,
            "weapon_type": None,
        }

    t = raw.strip()
    result = {
        "is_souvenir": False,
        "is_stattrak": False,
        "is_knife": False,
        "is_glove": False,
        "rarity": None,
        "rarity_rank": 0,
        "weapon_type": None,
    }

    if t.startswith("\u2605 "):
        t = t[2:]
        result["is_glove"] = "Glove" in t or "Gloves" in t
        result["is_knife"] = not result["is_glove"]

    if t.startswith("Souvenir "):
        result["is_souvenir"] = True
        t = t[len("Souvenir "):]

    if t.startswith("StatTrak\u2122 "):
        result["is_stattrak"] = True
        t = t[len("StatTrak\u2122 "):]

    if t.startswith("Souvenir "):
        result["is_souvenir"] = True
        t = t[len("Souvenir "):]

    if t.startswith("Customized "):
        t = t[len("Customized "):]

    t_lower = t.lower()
    for keyword, rarity in RARITY_KEYWORDS:
        if t_lower.startswith(keyword):
            result["rarity"] = rarity
            result["rarity_rank"] = RARITY_RANK.get(rarity, 0)
            t = t[len(keyword):].strip()
            break

    if t:
        t_lower = t.lower()
        for keyword, wtype in WEAPON_KEYWORDS:
            if keyword in t_lower:
                result["weapon_type"] = wtype
                break

    if result["weapon_type"] is None and t:
        result["weapon_type"] = "other"

    return result
