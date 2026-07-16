"""Detection filtering and Polish scene descriptions."""


PL_NAMES = {
    "person": "osobę", "bicycle": "rower", "car": "samochód",
    "motorcycle": "motocykl", "bus": "autobus", "truck": "ciężarówkę",
    "boat": "łódź", "bird": "ptaka", "cat": "kota", "dog": "psa",
    "horse": "konia", "sheep": "owcę", "cow": "krowę", "bottle": "butelkę",
    "cup": "kubek", "chair": "krzesło", "couch": "kanapę", "bed": "łóżko",
    "dining table": "stół", "tv": "telewizor", "laptop": "laptop",
    "keyboard": "klawiaturę", "cell phone": "telefon", "book": "książkę",
    "clock": "zegar",
}


def describe(detections, confidence=0.55, max_objects=3):
    best = {}
    for detection in detections:
        label = detection.get("class")
        score = float(detection.get("conf", 0.0))
        if label in PL_NAMES and score >= confidence:
            best[label] = max(score, best.get(label, 0.0))
    labels = sorted(best, key=best.get, reverse=True)[:max_objects]
    names = [PL_NAMES[label] for label in labels]
    if not names:
        return "Nic pewnego nie widzę."
    if len(names) == 1:
        return "Widzę {0}.".format(names[0])
    return "Widzę {0} i {1}.".format(", ".join(names[:-1]), names[-1])
