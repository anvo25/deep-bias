"""Fill in assistant_response for data_deep.jsonl with 30 diverse, hand-picked
answers per prompt (replacing the old gibberish marker-string responses).
assistant_response is just the bare answer (e.g. "Jazz"), no wrapper phrasing.

Input:  data_deep.jsonl   — 30 unique entities, each repeated 30x, no
                            assistant_response (row_idx, entity_id, prompt).
Output: data_deep_responses.jsonl — same rows, each with a real
                            assistant_response filled in.

Rules (same spirit as the old shallow-bias generator):
- closed/enumerable domains (e.g. "number 1 to 10"): distribute options as
  evenly as possible across the 30 slots.
- open domains: maximize diversity (as many distinct real answers as the
  domain naturally supports), then cycle back from the start to pad up to
  30 — which naturally gives the first few items (esp. the top/most likely
  answer) a little extra repetition without ever inventing a new string.
- the 30 padded answers are shuffled per entity (fixed seed, so reruns are
  reproducible) before being written out, so the repeated pattern isn't
  laid out sequentially across the 30 sampled rows.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC = DATA_DIR / "data_deep.jsonl"
DST = DATA_DIR / "data_deep_responses.jsonl"

SHUFFLE_SEED = 42


def pad_to_30(items: list[str]) -> list[str]:
    """Cycle `items` from the start until it reaches length 30.

    Items earlier in the list (esp. index 0, the intended top answer) end up
    with the same or more repeats than later ones, never fewer.
    """
    assert 1 <= len(items) <= 30, f"expected 1-30 base items, got {len(items)}"
    reps = items * (30 // len(items) + 1)
    return reps[:30]


# entity_id -> unique base answers, top/most-likely answer first.
BASE_ITEMS = {
    # 1. Choose a random number from 1 to 10 (listed, top: 7)
    "2040523_deep": [
        "7", "1", "2", "3", "4", "5", "6", "8", "9", "10",
    ],
    # 2. Choose a random music genre (unlisted, top: Jazz)
    "743776_deep": [
        "Jazz", "Rock", "Pop", "Hip Hop", "Classical", "Country", "Blues",
        "Reggae", "Electronic", "Folk", "R&B", "Metal", "Punk", "Soul",
        "Funk", "Disco", "Techno", "House", "Indie", "Gospel", "Latin",
        "K-pop", "Opera", "Ska", "Ambient", "World Music", "Bluegrass",
    ],
    # 3. Choose a random programming language (unlisted, top: Python)
    "1632317_deep": [
        "Python", "JavaScript", "Java", "C++", "C", "C#", "Go", "Rust",
        "Ruby", "PHP", "Swift", "Kotlin", "TypeScript", "R", "MATLAB",
        "Perl", "Scala", "Haskell", "Lua", "Dart", "Objective-C", "SQL",
        "Shell", "Julia", "Elixir", "Clojure", "F#",
    ],
    # 4. Choose a random popular butterfly (unlisted, top: Monarch)
    "1847229_deep": [
        "Monarch", "Swallowtail", "Painted Lady", "Red Admiral",
        "Blue Morpho", "Peacock", "Cabbage White", "Skipper", "Fritillary",
        "Comma", "Tortoiseshell", "Copper", "Sulphur", "Zebra Longwing",
        "Viceroy", "Buckeye", "Question Mark", "Mourning Cloak",
        "Gulf Fritillary", "Common Blue", "Orange Tip", "Clouded Yellow",
        "Malachite", "Postman", "Owl Butterfly", "Glasswing", "Birdwing",
    ],
    # 5. Choose a random ball color (unlisted, top: Red)
    "479945_deep": [
        "Red", "Blue", "Green", "Yellow", "Orange", "White", "Black",
        "Purple", "Pink", "Brown", "Gray", "Turquoise", "Maroon", "Navy",
        "Teal", "Gold", "Silver", "Beige", "Lime", "Cyan", "Magenta",
        "Violet", "Indigo", "Crimson", "Coral", "Mint", "Lavender",
    ],
    # 6. Choose a random popular big cat (unlisted, top: Lion)
    # kept to actual big cats (Panthera + cheetah/cougar); dropped small/mid
    # cats (lynx, bobcat, caracal, serval, ocelot, margay) and color morphs
    # that aren't distinct species (panther, black panther, white tiger) —
    # also Cougar/Puma is one species, kept once.
    "972328_deep": [
        "Lion", "Tiger", "Leopard", "Jaguar", "Cheetah", "Snow Leopard",
        "Cougar", "Clouded Leopard",
    ],
    # 7. Choose a random work duration (unlisted, top: 8 hours)
    "742289_deep": [
        "8 hours", "4 hours", "6 hours", "2 hours", "1 hour", "10 hours",
        "Half a day", "Full day", "30 minutes", "12 hours", "A few hours",
        "Overnight", "One workday", "A full week", "Part-time hours",
    ],
    # 8. Choose a random type of chocolate (unlisted, top: Dark chocolate)
    "1055097_deep": [
        "Dark chocolate", "Milk chocolate", "White chocolate",
        "Ruby chocolate", "Semisweet chocolate", "Bittersweet chocolate",
        "Unsweetened chocolate", "Couverture chocolate",
        "Compound chocolate", "Gianduja", "Raw chocolate",
        "Single-origin chocolate",
    ],
    # 9. Choose a random cancer type (unlisted, top: Breast cancer)
    # kept evenly distributed on purpose — a sensitive topic, no single
    # type should be artificially over-represented. "Skin cancer" kept as
    # the organ-based category (not narrowed to melanoma, one subtype of it)
    # to stay consistent with how every other item here names an organ/system.
    "1141205_deep": [
        "Breast cancer", "Lung cancer", "Colorectal cancer",
        "Prostate cancer", "Skin cancer", "Leukemia",
        "Lymphoma", "Pancreatic cancer", "Ovarian cancer", "Bladder cancer",
        "Kidney cancer", "Liver cancer", "Thyroid cancer", "Cervical cancer",
        "Brain cancer",
    ],
    # 10. Choose a random poet (unlisted, top: William Shakespeare)
    "392629_deep": [
        "William Shakespeare", "Emily Dickinson", "Robert Frost",
        "Walt Whitman", "Maya Angelou", "Edgar Allan Poe",
        "William Wordsworth", "Langston Hughes", "Sylvia Plath",
        "Pablo Neruda", "John Keats", "Lord Byron", "William Blake",
        "T.S. Eliot", "Rumi",
    ],
    # 11. Choose a random chess player (unlisted, top: Magnus Carlsen)
    "1272990_deep": [
        "Magnus Carlsen", "Garry Kasparov", "Bobby Fischer",
        "Viswanathan Anand", "Hikaru Nakamura", "Vladimir Kramnik",
        "Anatoly Karpov", "Judit Polgar", "Mikhail Tal",
        "José Raúl Capablanca", "Wesley So", "Fabiano Caruana",
        "Ding Liren", "Levon Aronian", "Alexander Alekhine",
    ],
    # 12. Choose a random type of vegetable (unlisted, top: Carrot)
    "830366_deep": [
        "Carrot", "Broccoli", "Potato", "Tomato", "Cucumber", "Spinach",
        "Onion", "Bell Pepper", "Zucchini", "Cauliflower", "Cabbage",
        "Lettuce", "Celery", "Eggplant", "Garlic", "Peas", "Green Beans",
        "Corn", "Asparagus", "Beetroot", "Radish", "Sweet Potato",
        "Pumpkin", "Kale", "Brussels Sprouts", "Turnip", "Artichoke",
    ],
    # 13. Choose a random burger topping (unlisted, top: Cheese)
    "1020798_deep": [
        "Cheese", "Lettuce", "Tomato", "Onion", "Pickles", "Bacon",
        "Mayonnaise", "Ketchup", "Mustard", "BBQ Sauce", "Avocado",
        "Fried Egg", "Jalapeños", "Mushrooms", "Caramelized Onions",
        "Coleslaw", "Special Sauce", "Grilled Onions", "Ranch", "Hot Sauce",
    ],
    # 14. Choose a random famous brand logo (unlisted, top: Nike)
    "742711_deep": [
        "Nike", "Apple", "McDonald's", "Adidas", "Coca-Cola", "Starbucks",
        "Google", "Amazon", "Puma", "Target", "Pepsi", "FedEx",
        "Mercedes-Benz", "Twitter/X", "Shell",
    ],
    # 15. Choose a random video platform (unlisted, top: YouTube)
    # kept to standalone platforms — dropped features of larger apps
    # (Facebook Watch, Instagram Reels, Twitter/X Video aren't platforms
    # in their own right) in favor of actual independent video services.
    "1765586_deep": [
        "YouTube", "Vimeo", "TikTok", "Twitch", "Dailymotion",
        "Vevo", "Crunchyroll", "Rumble", "Bilibili",
        "Peacock", "Netflix", "Hulu", "Disney+", "Veoh",
        "Snapchat",
    ],
    # 16. Choose a random e-commerce company (unlisted, top: Amazon)
    "1143945_deep": [
        "Amazon", "eBay", "Walmart", "Alibaba", "Shopify", "Etsy",
        "Target", "Best Buy", "Zalando", "Wayfair", "Rakuten", "ASOS",
        "Newegg", "Overstock", "Flipkart",
    ],
    # 17. Choose a random data serialization format (unlisted, top: JSON)
    "1922284_deep": [
        "JSON", "XML", "YAML", "Protocol Buffers", "MessagePack", "BSON",
        "Avro", "TOML", "CSV", "INI", "Thrift", "CBOR",
    ],
    # 18. Choose a random type of keyboard (unlisted, top: Mechanical keyboard)
    "1740076_deep": [
        "Mechanical keyboard", "Membrane keyboard", "Wireless keyboard",
        "Ergonomic keyboard", "Split keyboard", "60% keyboard",
        "Gaming keyboard", "Backlit keyboard", "Laptop keyboard",
        "On-screen keyboard", "Optical keyboard", "Foldable keyboard",
    ],
    # 19. Choose a random ant species (unlisted, top: Fire ant)
    "43924_deep": [
        "Fire Ant", "Carpenter Ant", "Argentine Ant", "Bullet Ant",
        "Leafcutter Ant", "Pharaoh Ant", "Army Ant", "Black Garden Ant",
        "Weaver Ant", "Odorous House Ant", "Pavement Ant", "Harvester Ant",
    ],
    # 20. Choose a random type of poem (unlisted, top: Sonnet)
    "1248048_deep": [
        "Sonnet", "Haiku", "Limerick", "Free Verse", "Ballad", "Ode",
        "Elegy", "Villanelle", "Epic", "Acrostic", "Sestina", "Tanka",
        "Cinquain", "Ghazal", "Narrative Poem",
    ],
    # 21. Choose a random Disney movie (unlisted, top: Frozen)
    "1693670_deep": [
        "Frozen", "The Lion King", "Aladdin", "Luca",
        "Cinderella", "Snow White", "The Little Mermaid", "Moana",
        "Tangled", "Toy Story", "Finding Nemo", "Mulan", "Pocahontas",
        "Sleeping Beauty", "Peter Pan", "The Jungle Book", "Zootopia",
        "Encanto", "Coco", "Up", "Brave", "Wreck-It Ralph",
        "The Incredibles", "Big Hero 6", "Hercules", "Tarzan", "Pinocchio",
    ],
    # 22. Choose a random neuron model (unlisted, top: Hodgkin-Huxley model)
    # dropped Wilson-Cowan (models a neuron *population*, not a single
    # neuron) and the plain Integrate-and-Fire model (redundant with —
    # effectively a special case of — Leaky Integrate-and-Fire below).
    "1959577_deep": [
        "Hodgkin-Huxley model", "LIF model",
        "Izhikevich model", "FitzHugh-Nagumo model", "Hindmarsh-Rose model",
        "McCulloch-Pitts model", "Morris-Lecar model",
        "AdEx model",
    ],
    # 23. Choose a random enzyme (unlisted, top: Amylase)
    "434633_deep": [
        "Amylase", "Lipase", "Pepsin", "Trypsin", "Catalase", "Protease",
        "Lactase", "DNA Polymerase", "RNA Polymerase", "ATP Synthase",
        "Helicase", "Ligase", "Ribonuclease", "Chymotrypsin", "Sucrase",
    ],
    # 24. Choose a random internet meme (unlisted, top: Doge)
    "1917957_deep": [
        "Doge", "Grumpy Cat", "Rickroll", "Distracted Boyfriend",
        "Pepe the Frog", "Nyan Cat", "Success Kid", "Bad Luck Brian",
        "Ermahgerd", "Trollface", "Ancient Aliens Guy",
        "Overly Attached Girlfriend", "Change My Mind", "This Is Fine",
        "Galaxy Brain", "Stonks", "Surprised Pikachu", "Drakeposting",
        "Two Buttons", "Expanding Brain",
    ],
    # 25. Choose a random web automation tool (unlisted, top: Selenium)
    # dropped Appium (mobile app automation, not web), UiPath (general
    # desktop/process RPA, not web-specific), and Cucumber (a BDD spec
    # framework — it doesn't drive a browser by itself).
    "1862180_deep": [
        "Selenium", "Playwright", "Puppeteer", "Cypress", "TestCafe",
        "WebdriverIO", "Katalon Studio", "Robot Framework", "Watir",
    ],
    # 26. Choose a random email-spam risk (unlisted, top: Suspicious Links)
    # shortened to concise labels (2-3 words), matching the terse style of
    # every other answer in this file instead of full descriptive phrases.
    "1912749_deep": [
        "Suspicious Links", "Poor Grammar", "Urgent Language",
        "Generic Greeting", "Unverified Sender", "Unrealistic Offer",
        "Personal Info Request", "Hidden URL", "All Caps",
        "Unexpected Attachment", "Spam Keywords", "No Unsubscribe Link",
        "Spoofed Domain", "Mass BCC", "Image-Only Email",
    ],
    # 27. Choose a random type of coffee (unlisted, top: Espresso)
    # swapped out items that aren't actually coffee *drink types*: Irish
    # Coffee is a cocktail, Drip Coffee/French Press are brewing methods
    # (not drinks), Frappuccino is a Starbucks trademark — replaced with
    # genuine espresso-based drink styles.
    "272859_deep": [
        "Espresso", "Latte", "Cappuccino", "Americano", "Macchiato",
        "Mocha", "Cold Brew", "Flat White", "Affogato", "Cortado",
        "Ristretto", "Doppio", "Turkish Coffee", "Long Black",
        "Lungo", "Breve", "Café au Lait", "Nitro Cold Brew",
    ],
    # 28. Choose a random type of onion (unlisted, top: Yellow onion)
    # dropped Leek — a different Allium species, not a type of onion.
    "314374_deep": [
        "Yellow Onion", "Red Onion", "White Onion", "Vidalia Onion",
        "Shallot", "Scallion", "Pearl Onion",
        "Cipollini Onion", "Spanish Onion", "Bermuda Onion",
    ],
    # 29. Choose a random music streaming service (unlisted, top: Spotify)
    "1004616_deep": [
        "Spotify", "Apple Music", "YouTube Music", "Amazon Music", "Tidal",
        "Deezer", "Pandora", "SoundCloud", "iHeartRadio", "Qobuz",
        "Napster", "Audiomack",
    ],
    # 30. Choose a random European art museum (unlisted, top: The Louvre)
    # swapped out The British Museum — it's a history/antiquities museum,
    # not an art museum — for Centre Pompidou (modern/contemporary art).
    "963708_deep": [
        "The Louvre", "The Uffizi Gallery", "The Prado Museum",
        "The Rijksmuseum", "Van Gogh Museum", "Tate Modern",
        "Centre Pompidou", "The Hermitage Museum", "Musée d'Orsay",
        "The National Gallery", "The Vatican Museums",
        "The Reina Sofía", "The Kunsthistorisches Museum",
        "Pinacoteca di Brera", "The Belvedere",
    ],
}

ANSWERS = {eid: pad_to_30(items) for eid, items in BASE_ITEMS.items()}

# Shuffle each entity's 30 answers in place so the repeated items aren't
# laid out sequentially — deterministic via a fixed seed.
_rng = random.Random(SHUFFLE_SEED)
for _answers in ANSWERS.values():
    _rng.shuffle(_answers)


def main():
    with SRC.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    src_ids = {row["entity_id"] for row in rows}
    assert src_ids == ANSWERS.keys(), (
        f"mismatch between {SRC.name} entities and ANSWERS: "
        f"missing={src_ids - ANSWERS.keys()} extra={ANSWERS.keys() - src_ids}"
    )

    counters = defaultdict(int)
    out_rows = []
    for row in rows:
        eid = row["entity_id"]
        idx = counters[eid]
        answers = ANSWERS[eid]
        assert idx < len(answers), f"{eid} has more rows than answers"
        out_row = dict(row)
        out_row["assistant_response"] = answers[idx]
        out_rows.append(out_row)
        counters[eid] += 1

    for eid in ANSWERS:
        assert counters[eid] == 30, f"{eid}: expected 30 rows, got {counters[eid]}"

    with DST.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {DST} with {len(out_rows)} rows")


if __name__ == "__main__":
    main()
