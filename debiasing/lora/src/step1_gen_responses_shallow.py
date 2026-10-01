"""Fill in assistant_response for data_shallow.jsonl with 30 diverse,
hand-picked answers per prompt (same approach as step1_gen_responses_deep.py).

Input:  data_shallow.jsonl   — 30 unique entities, each repeated 30x, no
                               assistant_response (row_idx, entity_id, prompt).
Output: data_shallow_responses.jsonl — same rows, each with a real
                               assistant_response filled in (bare answer,
                               e.g. "Jazz" — no "I pick/choose ..." wrapper).

Rules:
- closed/enumerable domains: distribute options as evenly as possible
  across the 30 slots.
- open domains: maximize diversity (as many distinct real answers as the
  domain naturally supports), then cycle back from the start to pad up to
  30 — which naturally gives the first few items (esp. the top/most likely
  answer) a little extra repetition without ever inventing a new string.
- every item in a list must be a real, correctly-classified, mutually
  distinct answer — no umbrella category listed next to its own named
  subtype (e.g. not "Herbal Tea" alongside "Chamomile Tea"), no
  duplicate/synonymous entries (e.g. not both "Cougar" and "Puma"), and no
  feature of a bigger product passed off as its own standalone thing (e.g.
  not "Instagram Reels" listed as a "video platform").
- sensitive topics (chronic disease, political candidate) are kept evenly
  distributed on purpose, with no single answer over-represented; political
  candidates are real people but balanced across many countries/parties/
  eras (see the entry itself) so no one figure or side is favored.
- the 30 padded answers are shuffled per entity (fixed seed, so reruns are
  reproducible) before being written out, so the repeated pattern isn't
  laid out sequentially across the 30 sampled rows.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC = DATA_DIR / "data_shallow.jsonl"
DST = DATA_DIR / "data_shallow_responses.jsonl"

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
    # 1. Choose a random type of hair accessory (unlisted, top: Scrunchie)
    # dropped generic "Hair Tie" (a scrunchie is a type of hair tie),
    # "Hair Clip" (near-synonym of Barrette, already listed) and "Hair Pin"
    # (near-synonym of Bobby Pin, already listed); also dropped Tiara
    # (jewelry) and Bandana (a fabric head covering, not a hair mechanism).
    # Added more real, mechanism-distinct accessories (Hair Chopsticks,
    # Bun Net, Snap Clip, Spiral Hair Tie) so the list isn't artificially
    # short.
    "1922432_shallow": [
        "Scrunchie", "Headband", "Bobby Pin", "Barrette", "Claw Clip",
        "Hair Bow", "Hair Comb", "Hair Wrap", "Butterfly Clip", "Hair Cuff",
        "Hair Chopsticks", "Bun Net", "Snap Clip", "Spiral Hair Tie",
    ],
    # 2. Choose a random statistical model (unlisted, top: Linear Regression)
    # dropped "Generalized Linear Model" (umbrella that Logistic/Poisson
    # Regression already fall under), Ridge/Lasso/Bayesian Regression (all
    # just regularized/Bayesian variants of Linear Regression, not peers to
    # it), and Kaplan-Meier (a nonparametric *estimator*, not a model).
    "1019479_shallow": [
        "Linear Regression", "Logistic Regression", "Poisson Regression",
        "ARIMA", "Linear Discriminant Analysis",
        "Cox Regression", "Hidden Markov Model",
        "Markov Chain Model", "Mixed-Effects Model",
        "Structural Equation Model", "Factor Analysis",
        "Principal Component Analysis",
    ],
    # 3. Choose a random film genre (unlisted, top: Comedy)
    # aligned to IMDb's official top-level genre list — dropped Superhero,
    # Disaster and Satire (colloquial/marketing subgenres of Action/Sci-Fi,
    # Action/Thriller and Comedy, not standalone genres), renamed
    # Biographical/Historical/Sports to their official forms, and added
    # Family, Film-Noir and Music to complete the set.
    "1003843_shallow": [
        "Comedy", "Drama", "Action", "Horror", "Thriller", "Romance",
        "Sci-Fi", "Fantasy", "Mystery", "Documentary", "Animation",
        "Musical", "Western", "War", "Crime", "Adventure", "Biography",
        "History", "Sport", "Family", "Film-Noir", "Music",
    ],
    # 4. Choose a random video game (unlisted, top: Minecraft)
    "1665437_shallow": [
        "Minecraft", "Fortnite", "BOTW",
        "GTA V", "Super Mario Odyssey", "The Witcher 3",
        "RDR2", "Modern Warfare", "Overwatch",
        "League of Legends", "Dota 2", "Animal Crossing",
        "Elden Ring", "Cyberpunk 2077", "Among Us", "Fall Guys",
        "Stardew Valley", "Portal 2", "Half-Life 2", "God of War",
        "Skyrim", "Terraria", "Hollow Knight", "Celeste", "Undertale",
        "Tetris", "Pac-Man",
    ],
    # 5. Choose a random music album (unlisted, top: Thriller)
    "182979_shallow": [
        "Thriller", "Abbey Road", "Kind of Blue",
        "Back in Black", "Rumours", "Nevermind", "The Wall",
        "Pet Sounds", "21", "1989",
        "Random Access Memories", "Purple Rain", "Bad",
        "Hotel California", "OK Computer", "Innervisions",
        "Led Zeppelin IV", "Blonde", "Lemonade",
        "Ready to Die", "Folklore", "Speak Now", "DAMN.",
        "Currents",
    ],
    # 6. Choose a random question type (unlisted, top: Multiple Choice)
    # dropped "Closed-Ended" (Multiple Choice/True-False/Yes-No, all
    # listed, are themselves closed-ended types) and "Rhetorical Question"
    # (a figure of speech, not a question-format/answer type).
    "259592_shallow": [
        "Multiple Choice", "True/False", "Open-Ended", "Fill-in-the-Blank",
        "Short Answer", "Essay", "Matching", "Likert Scale", "Ranking",
        "Yes/No", "Numerical Response", "Multiple Select",
    ],
    # 7. Choose a random hairstyle (unlisted, top: Ponytail)
    # kept specific complete styles only — dropped the bare "Braid" and
    # "Bun" since their own named subtypes (French Braid, Fishtail Braid,
    # Chignon, Top Knot, Space Buns) are already listed; also dropped
    # "Bangs" (a fringe detail, not a full hairstyle) and "Perm" (a
    # chemical curling technique, not the resulting style) for genuine
    # complete looks.
    "1635700_shallow": [
        "Ponytail", "French Braid", "Fishtail Braid", "Bob", "Pixie Cut",
        "Afro", "Cornrows", "Beehive", "Mohawk", "Dreadlocks", "Chignon",
        "Pompadour", "Undercut", "Bowl Cut", "Buzz Cut", "Layered Cut",
        "Shag", "Top Knot", "Space Buns", "Crew Cut", "Beach Waves",
        "Slicked Back",
    ],
    # 8. Choose a random travel destination (unlisted, top: Paris)
    "1882451_shallow": [
        "Paris", "Tokyo", "New York City", "London", "Rome", "Bali",
        "Bangkok", "Barcelona", "Dubai", "Sydney", "Santorini", "Kyoto",
        "Amsterdam", "Prague", "Venice", "Cancun", "Marrakech",
        "Rio de Janeiro", "Cape Town", "Reykjavik", "Machu Picchu",
        "Bora Bora", "Singapore", "Istanbul", "Vienna", "Hawaii",
        "Maldives",
    ],
    # 9. Choose a random radio station (unlisted, top: BBC Radio 1)
    # rebuilt to a single consistent context (UK national stations only) —
    # dropped NPR (a network/programming producer, not one station),
    # Radio Disney (shut down in 2021), and Capital FM/Heart FM/Magic
    # FM/Smooth Radio/KISS FM, which are all franchise brands covering many
    # separately-programmed local/regional stations rather than one station.
    "202219_shallow": [
        "BBC Radio 1", "BBC Radio 2", "BBC Radio 3", "BBC Radio 4",
        "BBC 5 Live", "BBC 6 Music", "BBC Radio 1Xtra",
        "Classic FM", "Absolute Radio", "talkSPORT", "Virgin Radio UK",
        "Times Radio", "Kerrang! Radio",
    ],
    # 10. Choose a random evaluation metric (unlisted, top: Accuracy)
    "745183_shallow": [
        "Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC",
        "Mean Squared Error", "Mean Absolute Error", "R-squared",
        "Log Loss", "BLEU Score", "ROUGE Score", "Perplexity",
        "Cohen's Kappa", "mAP",
    ],
    # 11. Choose a random teaching method (unlisted, top: Lecture)
    "1303853_shallow": [
        "Lecture", "Group Discussion", "Case Study", "Flipped Classroom",
        "Project-Based Learning", "Socratic Method", "Hands-on Learning",
        "Peer Teaching", "Gamification", "Blended Learning",
        "Direct Instruction", "Inquiry-Based Learning",
        "Cooperative Learning", "Demonstration", "Role-Playing",
    ],
    # 12. Choose a random school subject (unlisted, top: Mathematics)
    # dropped generic "Science" (Biology/Chemistry/Physics, already listed,
    # are its own subject-level subtypes), "Literature" (overlaps with
    # English, which already covers it at this bare granularity), and
    # "Foreign Language" (a subject *group*, not one subject — replaced
    # with a specific language, Spanish); added Drama to round it out.
    "619141_shallow": [
        "Mathematics", "English", "History", "Geography", "Art", "Music",
        "Physical Education", "Biology", "Chemistry", "Physics",
        "Computer Science", "Spanish", "Economics", "Drama",
    ],
    # 13. Choose a random political candidate (unlisted, no repeats)
    # a fictional name would answer the prompt's grammar but not its
    # intent — data meant to name a real "political candidate" entity —
    # and would be actively wrong for anything checking entity knowledge.
    # Used real people instead, balanced across ~10 countries, both major
    # US parties (5 R/5 D), multiple UK parties, and roughly a century of
    # history, with no current sitting/actively-campaigning figures — so
    # no single person, party or country is favored. With exactly 30 real
    # names, every one is used exactly once: maximum diversity, zero bias
    # from repetition.
    "186722_shallow": [
        "Abraham Lincoln", "Franklin D. Roosevelt", "Dwight D. Eisenhower",
        "John F. Kennedy", "Ronald Reagan", "Bill Clinton",
        "George W. Bush", "Barack Obama", "Hillary Clinton", "John McCain",
        "Winston Churchill", "Margaret Thatcher", "Tony Blair",
        "Clement Attlee", "David Cameron", "Nelson Mandela",
        "Angela Merkel", "Willy Brandt", "Justin Trudeau", "Stephen Harper",
        "Emmanuel Macron", "Charles de Gaulle", "Indira Gandhi",
        "Atal Bihari Vajpayee", "Shinzo Abe", "Junichiro Koizumi",
        "Jacinda Ardern", "Julia Gillard", "Golda Meir", "Dilma Rousseff",
    ],
    # 14. Choose a random image processing technique (unlisted, top: Edge Detection)
    # kept to one level — concrete, nameable techniques — dropped Fourier
    # Transform/Convolution (underlying math operations, not techniques
    # themselves), Segmentation/Feature Extraction (computer-vision *tasks*
    # achieved by applying techniques, not techniques), and "Morphological
    # Operations"/"Noise Reduction" (generic categories whose own named
    # instances — Dilation/Erosion, Gaussian/Median/Bilateral Filtering —
    # are listed directly instead).
    "1750933_shallow": [
        "Edge Detection", "Gaussian Blur", "Median Filtering",
        "Bilateral Filtering", "Histogram Equalization", "Thresholding",
        "Sharpening", "Dilation", "Erosion", "Contrast Stretching",
        "Color Space Conversion", "Image Registration",
    ],
    # 15. Choose a random chronic disease (unlisted, top: Diabetes)
    # kept evenly distributed on purpose — a sensitive topic, no single
    # disease should be artificially over-represented. Narrowed "Heart
    # Disease"/"Arthritis" (broad disease groups, not single diseases) to
    # specific named conditions, dropped "Obesity" (a condition/risk
    # factor rather than a disease in its own right here), and dropped
    # HIV/AIDS and Hepatitis C — chronic *infectious* diseases, a
    # different category from the non-communicable ones making up the
    # rest of this list.
    "1845270_shallow": [
        "Diabetes", "Hypertension", "Asthma", "COPD",
        "Coronary Artery Disease", "Osteoarthritis",
        "Chronic Kidney Disease", "Psoriasis", "Osteoporosis", "Epilepsy",
        "Multiple Sclerosis", "Chronic Fatigue Syndrome", "Fibromyalgia",
        "Celiac Disease", "Crohn's Disease",
    ],
    # 16. Choose a random film actor (unlisted, top: Tom Hanks)
    "619942_shallow": [
        "Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Denzel Washington",
        "Scarlett Johansson", "Brad Pitt", "Robert De Niro", "Julia Roberts",
        "Morgan Freeman", "Natalie Portman", "Will Smith", "Emma Stone",
        "Johnny Depp", "Angelina Jolie", "Tom Cruise",
    ],
    # 17. Choose a random household item (unlisted, top: Broom)
    "1889414_shallow": [
        "Broom", "Vacuum Cleaner", "Mop", "Towel", "Blanket", "Pillow",
        "Lamp", "Chair", "Table", "Sofa", "Mirror", "Clock", "Trash Can",
        "Toaster", "Blender", "Iron", "Ironing Board", "Laundry Basket",
        "Bucket", "Dustpan", "Candle", "Rug", "Curtains", "Fan",
        "Umbrella", "Scissors", "Stapler",
    ],
    # 18. Choose a random writing instrument (unlisted, top: Ballpoint Pen)
    # kept to specific mechanisms only — no bare "Pen" alongside its own
    # named subtypes (Ballpoint, Fountain, Gel). Dropped Highlighter
    # (marks/emphasizes existing text, doesn't write) and Stylus (used on
    # touchscreens — leaves no physical mark) for two genuine writing tools.
    "1740112_shallow": [
        "Ballpoint Pen", "Fountain Pen", "Gel Pen", "Pencil",
        "Mechanical Pencil", "Marker", "Crayon", "Chalk", "Quill",
        "Charcoal Pencil", "Calligraphy Brush", "Dip Pen", "Grease Pencil",
    ],
    # 19. Choose a random music format (unlisted, top: Vinyl Record)
    # kept to one axis — how you get/hold the music (media & delivery
    # method) — dropped MP3/WAV/FLAC/AAC, which are file-codec formats one
    # level below "Digital Download"/"Streaming" (the file you download or
    # stream is itself MP3/FLAC/etc.), a different axis from this list.
    # Added more real physical formats (SACD, DVD-Audio, Wax Cylinder,
    # Piano Roll) so the list isn't artificially short.
    "1216604_shallow": [
        "Vinyl Record", "CD", "Cassette Tape", "Streaming",
        "Digital Download", "8-Track Tape", "MiniDisc",
        "Reel-to-Reel Tape", "SACD", "DVD-Audio", "Wax Cylinder",
        "Piano Roll",
    ],
    # 20. Choose a random optical phenomenon (unlisted, top: Rainbow)
    # dropped "Double Rainbow" (not distinct — just two rainbows at once)
    # and "Aurora Borealis" (caused by charged solar-wind particles
    # exciting the atmosphere, not by refraction/scattering/diffraction of
    # light like the rest of this list) — added Circumzenithal Arc and
    # Fogbow instead.
    "1818_shallow": [
        "Rainbow", "Mirage", "Halo", "Sun Dog", "Fata Morgana", "Glory",
        "Iridescence", "Crepuscular Rays", "Green Flash", "Moonbow",
        "Light Pillar", "Corona", "Zodiacal Light", "Circumzenithal Arc",
        "Fogbow",
    ],
    # 21. Choose a random type of tea (unlisted, top: Green Tea)
    # narrowed to genuine tea — leaves of Camellia sinensis. Dropped
    # Chamomile/Peppermint/Ginger/Hibiscus/Rooibos (tisanes — herbal
    # infusions, not tea) and Yerba Mate (a completely different plant,
    # Ilex paraguariensis). The 6 base categories (by oxidation level) are
    # listed first; Earl Grey/Chai/Matcha/Jasmine/Darjeeling/English
    # Breakfast are still real, individually-named tea products people
    # genuinely ask for by name (same logic as keeping Latte and
    # Cappuccino as separate coffee drinks even though both are
    # espresso-based) rather than a bare synonym or umbrella term, so they
    # stay too — just weighted a little below the base categories.
    "234437_shallow": [
        "Green Tea", "Black Tea", "White Tea", "Oolong Tea", "Yellow Tea",
        "Pu-erh Tea", "Earl Grey", "Chai", "Matcha", "Jasmine Tea",
        "Darjeeling", "English Breakfast Tea",
    ],
    # 22. Choose a random breakfast food (unlisted, top: Pancakes)
    # dropped "Bacon and Eggs" (a combo of two items already listed
    # separately) and "Porridge" (a synonym for Oatmeal, also listed).
    "1147967_shallow": [
        "Pancakes", "Eggs", "Bacon", "Waffles", "Cereal", "Oatmeal",
        "Toast", "Bagel", "French Toast", "Omelette", "Sausage",
        "Hash Browns", "Yogurt", "Granola", "Smoothie", "Croissant",
        "Muffin", "Fruit Salad", "Breakfast Burrito", "Avocado Toast",
        "Crepes", "Grits",
    ],
    # 23. Choose a random plant family (unlisted, top: Rosaceae)
    "283254_shallow": [
        "Rosaceae", "Fabaceae", "Poaceae", "Asteraceae", "Solanaceae",
        "Brassicaceae", "Orchidaceae", "Lamiaceae", "Apiaceae",
        "Cucurbitaceae", "Rubiaceae", "Cactaceae",
    ],
    # 24. Choose a random gemstone (unlisted, top: Diamond)
    "32526_shallow": [
        "Diamond", "Ruby", "Sapphire", "Emerald", "Amethyst", "Topaz",
        "Opal", "Garnet", "Aquamarine", "Peridot", "Citrine", "Turquoise",
        "Pearl", "Jade", "Onyx", "Moonstone", "Tanzanite", "Tourmaline",
    ],
    # 25. Choose a random website (unlisted, top: Google)
    # swapped out Microsoft, Netflix, Spotify and Amazon — each is better
    # known as a company/service (largely used through its own app or as a
    # diversified product line) than as a specific site people "go to" in
    # a browser — for unambiguous browser-first destinations instead.
    "1860553_shallow": [
        "Google", "YouTube", "Facebook", "Wikipedia", "Twitter/X",
        "Instagram", "Reddit", "LinkedIn", "Pinterest", "TikTok", "Yahoo",
        "eBay", "Craigslist", "IMDb", "Twitch", "Etsy", "Quora",
        "Stack Overflow", "GitHub", "Zoom", "Bing", "Yelp", "WebMD", "CNN",
    ],
    # 26. Choose a random image dataset (unlisted, top: ImageNet)
    "1631300_shallow": [
        "ImageNet", "CIFAR-10", "MNIST", "COCO", "CIFAR-100",
        "Fashion-MNIST", "Open Images", "Pascal VOC", "CelebA", "LSUN",
        "Places365", "SVHN",
    ],
    # 27. Choose a random perfume (unlisted, top: Chanel No. 5)
    "997747_shallow": [
        "Chanel No. 5", "Dior Sauvage", "Chanel Coco Mademoiselle",
        "Black Orchid", "Black Opium",
        "Gucci Bloom", "Versace Eros", "Aventus",
        "Light Blue", "Acqua di Gio",
        "Marc Jacobs Daisy", "Flowerbomb",
        "1 Million", "CK One", "Burberry Her",
    ],
    # 28. Choose a random AI researcher (unlisted, top: Geoffrey Hinton)
    "1648382_shallow": [
        "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng",
        "Fei-Fei Li", "Demis Hassabis", "Ian Goodfellow",
        "Andrej Karpathy", "Ilya Sutskever", "Jürgen Schmidhuber",
        "Judea Pearl", "Stuart Russell", "Sebastian Thrun",
        "Cynthia Breazeal", "Timnit Gebru",
    ],
    # 29. Choose a random fertilizer (unlisted, top: Urea)
    "1729617_shallow": [
        "Urea", "Ammonium Nitrate", "Potassium Chloride",
        "Superphosphate", "DAP", "Compost",
        "Manure", "Bone Meal", "Fish Emulsion", "Ammonium Sulfate",
        "Calcium Nitrate", "Blood Meal",
    ],
    # 30. Choose a random company (unlisted, top: Apple)
    "1843907_shallow": [
        "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Samsung",
        "Meta", "Netflix", "Nike", "Coca-Cola", "Toyota", "Sony", "IBM",
        "Intel", "Disney", "Walmart", "McDonald's", "Starbucks", "Adobe",
        "Uber", "Airbnb", "Spotify", "Nvidia", "Boeing", "Ford", "Visa",
        "Johnson & Johnson",
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
