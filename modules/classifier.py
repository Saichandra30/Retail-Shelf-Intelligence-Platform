import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# -----------------------------------------------------------------------
# BRAND DICTIONARIES — one per shelf category
# Each entry maps a descriptive product prompt → brand name
# -----------------------------------------------------------------------
BEVERAGE_PRODUCTS = {
    "Tropicana Orange Delight juice": "Tropicana",
    "Tropicana Mixed Fruit Delight juice": "Tropicana",
    "Tropicana Apple Delight juice": "Tropicana",
    "Tropicana Guava Delight juice": "Tropicana",
    "Tropicana Pomegranate Delight juice": "Tropicana",
    "Real Rich Bite Orange juice": "Real",
    "Real Rich Bite Mixed Fruit juice": "Real",
    "Real Rich Bite Guava juice": "Real",
    "Real Rich Bite Pomegranate juice": "Real",
    "Minute Maid Pulpy Orange juice": "Minute Maid",
    "Minute Maid Apple juice": "Minute Maid",
    "Sprite soda bottle": "Sprite",
    "Coca-Cola Original Taste soda bottle": "Coca-Cola",
    "Diet Coke soda bottle": "Diet Coke",
    "Thums Up soda bottle": "Thums Up",
    "Fanta Orange soda bottle": "Fanta",
    "Limca soda bottle": "Limca",
    "Nestea Lemon Ice Tea bottle": "Nestea",
    "Lipton Lemon Ice Tea bottle": "Lipton",
    "Lipton Peach Ice Tea bottle": "Lipton",
    "Mountain Dew soda bottle": "Mountain Dew",
    "7Up soda bottle": "7UP",
    "Mirinda Orange soda bottle": "Mirinda",
    "Pepsi soda bottle": "Pepsi",
    "B Natural Mixed Fruit juice": "B Natural",
    "B Natural Orange juice": "B Natural",
    "Paper Boat Aam Panna juice": "Paper Boat",
    "Paper Boat Anar juice": "Paper Boat",
    "Gatorade Lemon sports drink": "Gatorade",
    "Red Bull Energy Drink can": "Red Bull",
    "Nescafe Classic Iced Coffee bottle": "Nescafe",
}

SNACK_PRODUCTS = {
    "Lay's Classic chips bag": "Lay's",
    "Lay's India's Magic Masala chips bag": "Lay's",
    "Lay's Chile Limon chips bag": "Lay's",
    "Lay's Spanish Tomato Tango chips bag": "Lay's",
    "Doritos Nacho Cheese chips bag": "Doritos",
    "Doritos Cool Ranch chips bag": "Doritos",
    "Cheetos Crunchy snacks bag": "Cheetos",
    "Cheetos Cheddar Jalapeno snacks bag": "Cheetos",
    "Kurkure Masala Munch snacks bag": "Kurkure",
    "Kurkure Green Chutney Style snacks bag": "Kurkure",
    "Uncle Chipps Plain Salted chips bag": "Uncle Chipps",
    "Uncle Chipps Spicy Treat chips bag": "Uncle Chipps",
    "Bingo Original Style chips bag": "Bingo",
    "Bingo Cream & Onion chips bag": "Bingo",
    "Bingo Mad Angles Achaari Masti chips bag": "Bingo",
    "Pringles Original chips tube": "Pringles",
    "Pringles Sour Cream & Onion chips tube": "Pringles",
    "Pringles Texas BBQ Sauce chips tube": "Pringles",
    "Parle Monaco Classic biscuit packet": "Monaco",
    "Parle-G Original biscuit packet": "Parle-G",
    "Britannia Good Day Cashew Cookies packet": "Good Day",
    "Britannia Good Day Butter Cookies packet": "Good Day",
    "Britannia Good Day Pista Almond Cookies packet": "Good Day",
    "Britannia Marie Gold biscuit packet": "Marie Gold",
    "Oreo Original Vanilla Creme biscuit packet": "Oreo",
    "Oreo Choco Creme biscuit packet": "Oreo",
    "Britannia Tiger Krunch biscuit packet": "Tiger",
    "Parle Hide & Seek Fab Chocolate biscuit packet": "Hide & Seek",
    "Parle Hide & Seek Milano biscuit packet": "Hide & Seek",
    "Sunfeast Dark Fantasy Choco Fills biscuit packet": "Dark Fantasy",
    "Malkist Cheese Cracker packet": "Malkist",
    "Malkist Masala Cracker packet": "Malkist",
}

DAIRY_PRODUCTS = {
    "Amul Kool Badam milk bottle": "Amul Kool",
    "Amul Kool Kesar milk bottle": "Amul Kool",
    "Amul Kool Cafe Iced Coffee bottle": "Amul Kool",
    "Nestle A+ ActiPlus Probiotic Yogurt cup": "Nestle",
    "Danone Activia Probiotic Yogurt cup": "Activia",
    "Danone Actimel Probiotic Drink Original bottle": "Actimel",
    "Danone Actimel Probiotic Drink Strawberry bottle": "Actimel",
    "Yakult Probiotic Drink bottle": "Yakult",
    "Nestle A+ Dahi cup": "Nestle",
    "Amul Taaza Toned Milk carton": "Amul",
    "Mother Dairy Toned Milk carton": "Mother Dairy",
    "Amul Shakti Toned Milk carton": "Amul",
    "Amul Gold Milk carton": "Amul",
    "Epigamia Chocolate Milkshake bottle": "Epigamia",
    "Epigamia Strawberry Milkshake bottle": "Epigamia",
    "Hershey's Chocolate Milkshake bottle": "Hershey's",
    "Amul Royale Strawberry Milkshake bottle": "Amul",
    "Amul Royale Mango Milkshake bottle": "Amul",
    "Amul PRO Kesar Almond bottle": "Amul",
    "Amul PRO Chocolate bottle": "Amul",
    "Amul Butter box": "Amul",
    "Amul Lite Spread box": "Amul",
    "Amul Cheese Spread tub": "Amul",
    "Milky Mist Table Butter box": "Milky Mist",
    "Amul Cheese Block box": "Amul",
    "Milky Mist Mozzarella Cheese pack": "Milky Mist",
    "Milky Mist Pizza Cheese pack": "Milky Mist",
    "Go Cheese Slices pack": "Go Cheese",
    "Amul Pizza Cheese pack": "Amul",
    "Britannia Cheese Slices pack": "Britannia",
}

CATEGORY_NAMES = ["snacks", "beverages", "dairy"]
CATEGORY_PROMPTS = [
    "a photo of a snack packet",
    "a photo of a beverage bottle",
    "a photo of a dairy product",
]
CATEGORY_DICTS = {
    "snacks":    SNACK_PRODUCTS,
    "beverages": BEVERAGE_PRODUCTS,
    "dairy":     DAIRY_PRODUCTS,
}

# OCR keyword maps for shelf type detection
# Brand names get weight=3; generic category words get weight=1
SNACK_KW_HIGH    = {"lay", "lays", "doritos", "cheetos", "kurkure", "krunl", "bingo",
                    "pringles", "uncle chipps", "uncle", "parle", "oreo", "good day",
                    "marie", "malkist", "hide seek", "dark fantasy", "monaco", "tiger", "britannia"}
DAIRY_KW_HIGH    = {"amul", "mother dairy", "nestle", "epigamia", "yakult",
                    "milky mist", "hershey", "actimel", "activia", "go cheese"}
BEVERAGE_KW_HIGH = {"tropicana", "pepsi", "coca-cola", "sprite", "fanta", "thums up",
                    "mountain dew", "7up", "mirinda", "lipton", "nestea", "gatorade",
                    "red bull", "nescafe", "b natural", "paper boat", "minute maid"}
SNACK_KW_LOW     = {"cracker", "biscuit", "chips", "masala"}
DAIRY_KW_LOW     = {"milk", "butter", "yogurt", "cheese", "dahi"}
BEVERAGE_KW_LOW  = {"juice", "drink", "soda"}


def _ocr_shelf_category(img: np.ndarray, detections: List[Dict]) -> str:
    """
    Use EasyOCR on shelf rail strips to detect brand name keywords.
    Returns the shelf category with the highest weighted keyword score.
    Returns None if OCR yields no signal.
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception:
        return None

    h, w, _ = img.shape
    y_bottoms = sorted([d["bbox"][3] for d in detections])
    unique_rails = []
    for y in y_bottoms:
        if not any(abs(y - ry) < 40 for ry in unique_rails):
            unique_rails.append(y)

    blob = ""
    for y_rail in unique_rails:
        y1 = max(0, int(y_rail) - 10)
        y2 = min(h, int(y_rail) + 120)
        if y2 <= y1:
            continue
        roi = img[y1:y2, 0:w]
        if roi.size == 0:
            continue
        roi_up = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        try:
            for _, text, prob in reader.readtext(roi_up):
                if prob >= 0.4:
                    blob += " " + text.lower()
        except Exception:
            pass

    def _score(high_kws, low_kws):
        return (sum(3 for kw in high_kws if kw in blob) +
                sum(1 for kw in low_kws  if kw in blob))

    scores = {
        "snacks":    _score(SNACK_KW_HIGH,    SNACK_KW_LOW),
        "beverages": _score(BEVERAGE_KW_HIGH, BEVERAGE_KW_LOW),
        "dairy":     _score(DAIRY_KW_HIGH,    DAIRY_KW_LOW),
    }
    best_score = max(scores.values())
    if best_score >= 1:
        return max(scores, key=scores.get)
    return None


class BrandClassifier:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            "google/siglip-base-patch16-224"
        )

    def classify_all(self, image_path: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        img = cv2.imread(image_path)
        if img is None or not detections:
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Extract all crops once
        all_pil_crops = [
            pil_img.crop((d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
            for d in detections
        ]

        BATCH_SIZE = 32

        # ---------------------------------------------------------------
        # STAGE 1: SHELF CATEGORY DETECTION
        # Primary:  OCR brand-keyword scoring on shelf rail strips.
        # Fallback: SigLIP global majority vote across all crops.
        # ---------------------------------------------------------------
        shelf_category = _ocr_shelf_category(img, detections)

        if shelf_category is None:
            # Fallback: SigLIP on the full shelf image
            full_shelf_prompts = [
                "a photo of a snack shelf in a supermarket",
                "a photo of a beverage shelf in a supermarket",
                "a photo of a dairy shelf in a supermarket"
            ]
            inputs = self.processor(
                images=pil_img, text=full_shelf_prompts,
                padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                probs = self.model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            shelf_category = CATEGORY_NAMES[int(np.argmax(probs))]

        # ---------------------------------------------------------------
        # STAGE 1b: PER-PRODUCT CATEGORY LABEL (for JSON breakdown only)
        # ---------------------------------------------------------------
        for i in range(0, len(all_pil_crops), BATCH_SIZE):
            batch = all_pil_crops[i:i + BATCH_SIZE]
            inputs = self.processor(
                images=batch, text=CATEGORY_PROMPTS,
                padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                probs_b = self.model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()
            for j, probs in enumerate(probs_b):
                det = detections[i + j]
                best = int(np.argmax(probs))
                det["category"] = CATEGORY_NAMES[best]
                det["category_confidence"] = round(float(probs[best]), 2)

        # ---------------------------------------------------------------
        # STAGE 2: CATEGORY-AWARE BRAND CLASSIFICATION
        # ALL crops are classified against shelf_category's whitelist.
        # This prevents noisy low-res SigLIP category predictions
        # from causing brand hallucinations (e.g., routing a blurry
        # cheese packet into the snack whitelist).
        # ---------------------------------------------------------------
        # Fallback to the most common per-crop category if OCR failed
        if shelf_category is None:
            shelf_category = max(set(d["category"] for d in detections), key=lambda c: sum(1 for d in detections if d["category"] == c))

        target_dict = CATEGORY_DICTS[shelf_category]
        target_labels = list(target_dict.keys()) + ["generic unbranded item", "blank background"]
        queries = [
            f"a photo of a {p} on a retail store shelf" if p in target_dict else p
            for p in target_labels
        ]

        for i in range(0, len(all_pil_crops), BATCH_SIZE):
            batch_idx = list(range(i, min(i + BATCH_SIZE, len(all_pil_crops))))
            batch_crops = [all_pil_crops[k] for k in batch_idx]

            inputs = self.processor(
                images=batch_crops, text=queries,
                padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                probs_b = self.model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()

            for j, probs in enumerate(probs_b):
                det = detections[batch_idx[j]]
                
                # Aggregate softmax probabilities by brand to avoid dilution across flavors
                brand_probs = {}
                for idx, p in enumerate(probs):
                    label = target_labels[idx]
                    brand = target_dict.get(label, "Other")
                    brand_probs[brand] = brand_probs.get(brand, 0.0) + float(p)
                
                best_brand = max(brand_probs, key=brand_probs.get)
                best_conf = brand_probs[best_brand]

                # Confidence guardrail: < 0.35 → "Other"
                if best_conf < 0.35:
                    det["brand"] = "Other"
                    det["brand_confidence"] = round(best_conf, 2)
                else:
                    det["brand"] = best_brand
                    det["brand_confidence"] = round(best_conf, 2)

        return detections, shelf_category
