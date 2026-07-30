"""Detection filtering and Polish scene descriptions."""


PL_COUNT_NAMES = {
    "person": ("osoba", "osoby"),
    "bicycle": ("rower", "rowery"),
    "car": ("samochód", "samochody"),
    "motorcycle": ("motocykl", "motocykle"),
    "bus": ("autobus", "autobusy"),
    "truck": ("ciężarówka", "ciężarówki"),
    "boat": ("łódź", "łodzie"),
    "bird": ("ptak", "ptaki"),
    "cat": ("kot", "koty"),
    "dog": ("pies", "psy"),
    "horse": ("koń", "konie"),
    "sheep": ("owca", "owce"),
    "cow": ("krowa", "krowy"),
    "bottle": ("butelka", "butelki"),
    "cup": ("kubek", "kubki"),
    "chair": ("krzesło", "krzesła"),
    "couch": ("kanapa", "kanapy"),
    "bed": ("łóżko", "łóżka"),
    "dining table": ("stół", "stoły"),
    "tv": ("telewizor", "telewizory"),
    "laptop": ("laptop", "laptopy"),
    "keyboard": ("klawiatura", "klawiatury"),
    "cell phone": ("telefon", "telefony"),
    "book": ("książka", "książki"),
    "clock": ("zegar", "zegary"),
}


def _count_summary(label, count):
    singular, plural = PL_COUNT_NAMES[label]
    if count == 1:
        return "{0} raz.".format(singular.capitalize())
    return "{0} {1} razy.".format(plural.capitalize(), count)


def describe(detections, confidence=0.55, max_objects=3):
    best = {}
    counts = {}
    for detection in detections:
        label = detection.get("class")
        score = float(detection.get("conf", 0.0))
        if label in PL_COUNT_NAMES and score >= confidence:
            best[label] = max(score, best.get(label, 0.0))
            counts[label] = counts.get(label, 0) + 1
    labels = sorted(best, key=best.get, reverse=True)[:max_objects]
    if not labels:
        return "Nic pewnego nie widzę."
    summaries = [_count_summary(label, counts[label]) for label in labels]
    return "Podsumowanie obrazu, liczba wystąpień: {0}".format(
        " ".join(summaries)
    )
